using System;
using GhostGPU.Core;
using GhostGPU.Validation.Models;

namespace GhostGPU.Validation.Tests
{
    public static class MemoryGuardTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestMemoryLimitBytesGuard(recordResult);
            TestMaximumNearestPairsGuard(recordResult);
            TestClosestPointsPartialAllocationFailureRollback(recordResult);
            TestInvalidEnumsRejection(session, recordResult);
            TestSessionDisposeMetricsZeroAndInvalidation(recordResult);
        }

        private static void TestMemoryLimitBytesGuard(Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Create a restricted session with only 1024 bytes memory limit
                var options = new SessionOptions
                {
                    Backend = GpuBackend.CpuDebug,
                    MemoryLimitBytes = 1024
                };

                using var restrictedSession = SessionFactory.CreateSession(options);

                // 100 V3 points = 100 * 24 = 2400 bytes, which exceeds 1024 bytes
                var points = new V3[100];
                for (int i = 0; i < points.Length; i++)
                    points[i] = new V3(i, i, i);

                bool threw = false;
                try
                {
                    // Device budget guard: explicit device residency (Auto keeps small uploads host-side, outside the device budget).
                    using var buf = restrictedSession.UploadPoints(points, ElementType.V3D, Residency.Device);
                }
                catch (InvalidOperationException)
                {
                    threw = true;
                }
                catch (OutOfMemoryException)
                {
                    threw = true;
                }

                if (!threw)
                    throw new Exception("SessionOptions.MemoryLimitBytes violation did not throw!");

                record(new TestResultItem
                {
                    Name = "Guard_MemoryLimitBytes_Enforced",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Allocation exceeding MemoryLimitBytes throws InvalidOperationException/OutOfMemoryException."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Guard_MemoryLimitBytes_Enforced",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestMaximumNearestPairsGuard(Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Create session with MaximumNearestPairs limit = 100
                var options = new SessionOptions
                {
                    Backend = GpuBackend.CpuDebug,
                    MaximumNearestPairs = 100
                };

                using var restrictedSession = SessionFactory.CreateSession(options);

                // 20 queries * 10 targets = 200 pairs, exceeding limit 100
                var queries = new V3[20];
                var targets = new V3[10];

                using var qBuf = restrictedSession.UploadPoints(queries);
                using var tBuf = restrictedSession.UploadPoints(targets);

                bool threw = false;
                try
                {
                    using var res = restrictedSession.ClosestPoints(qBuf, tBuf);
                }
                catch (InvalidOperationException)
                {
                    threw = true;
                }
                catch (ArgumentException)
                {
                    threw = true;
                }

                if (!threw)
                    throw new Exception("Exceeding MaximumNearestPairs limit did not throw!");

                record(new TestResultItem
                {
                    Name = "Guard_MaximumNearestPairs_Enforced",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Pairwise work exceeding MaximumNearestPairs throws without allocating full distance matrix."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Guard_MaximumNearestPairs_Enforced",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestClosestPointsPartialAllocationFailureRollback(Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Regression test required by Task #80:
                // Input uploads fit within MemoryLimitBytes, but the 3 ClosestPoints outputs
                // (points 24B, indices 4B, distances 8B) exceed the budget.
                // After exception:
                // 1) ActiveBytes and LiveAllocations must roll back to exact values before the failed call.
                // 2) Subsequent valid operations on the session must succeed normally without state corruption.

                // Query: 50 points * 24B = 1200B
                // Target: 10 points * 24B = 240B
                // Total input = 1440B
                // Outputs require: 50*24 (1200B) + 50*4 (200B) + 50*8 (400B) = 1800B
                // Total required with outputs = 1440 + 1800 = 3240B.
                // We set MemoryLimitBytes = 2500B:
                // Inputs (1440B) comfortably fit in 2500B.
                // Outputs (1800B) will breach the 2500B limit (1440 + 1800 = 3240 > 2500).
                var options = new SessionOptions
                {
                    Backend = GpuBackend.CpuDebug,
                    MemoryLimitBytes = 2500
                };

                using var session = SessionFactory.CreateSession(options);

                var queries = new V3[50];
                for (int i = 0; i < queries.Length; i++) queries[i] = new V3(i, 0, 0);

                var targets = new V3[10];
                for (int i = 0; i < targets.Length; i++) targets[i] = new V3(i * 5, 0, 0);

                // Inputs explicitly device-resident so the before/after measurement isolates the op's own outputs
                // (with Auto, ClosestPoints would legitimately promote and cache the inputs on the device).
                using var qBuf = session.UploadPoints(queries, ElementType.V3D, Residency.Device);
                using var tBuf = session.UploadPoints(targets, ElementType.V3D, Residency.Device);

                long activeBytesBefore = session.Metrics.ActiveBytes;
                long liveAllocsBefore = session.Metrics.LiveAllocations;

                bool threw = false;
                try
                {
                    using var res = session.ClosestPoints(qBuf, tBuf);
                }
                catch (InvalidOperationException)
                {
                    threw = true;
                }
                catch (OutOfMemoryException)
                {
                    threw = true;
                }

                if (!threw)
                    throw new Exception("ClosestPoints did not throw when output memory exceeded MemoryLimitBytes!");

                long activeBytesAfter = session.Metrics.ActiveBytes;
                long liveAllocsAfter = session.Metrics.LiveAllocations;

                if (activeBytesAfter != activeBytesBefore)
                {
                    throw new Exception($"ActiveBytes leaked on failed allocation: before={activeBytesBefore}, after={activeBytesAfter} (delta {activeBytesAfter - activeBytesBefore})");
                }

                if (liveAllocsAfter != liveAllocsBefore)
                {
                    throw new Exception($"LiveAllocations leaked on failed allocation: before={liveAllocsBefore}, after={liveAllocsAfter}");
                }

                // Subsequent valid operation must succeed:
                // Upload a small array that fits in remaining budget (2500 - 1440 = 1060B available)
                var smallScalars = new[] { 1.23, 4.56, 7.89 };
                using var smallBuf = session.UploadScalars(smallScalars);
                var downloadedSmall = session.DownloadScalars(smallBuf);

                AssertHelper.AssertClose(downloadedSmall[0], 1.23, 1e-9, "Post-recovery scalar 0");
                AssertHelper.AssertClose(downloadedSmall[2], 7.89, 1e-9, "Post-recovery scalar 2");

                record(new TestResultItem
                {
                    Name = "Guard_ClosestPoints_PartialAllocationFailure_Rollback",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Partial allocation failure in ClosestPoints rolled back ActiveBytes/LiveAllocations cleanly; subsequent operations succeeded."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Guard_ClosestPoints_PartialAllocationFailure_Rollback",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestInvalidEnumsRejection(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                using var a = session.UploadScalars(new[] { 1.0, 2.0 });
                using var b = session.UploadScalars(new[] { 3.0, 4.0 });

                bool threwMath = false;
                try
                {
                    session.ScalarMath(a, b, (ScalarOperation)9999);
                }
                catch (ArgumentOutOfRangeException)
                {
                    threwMath = true;
                }
                catch (ArgumentException)
                {
                    threwMath = true;
                }

                if (!threwMath)
                    throw new Exception("Invalid ScalarOperation enum did not throw ArgumentOutOfRangeException.");

                bool threwMap = false;
                try
                {
                    session.Map(a, (MapFunction)9999);
                }
                catch (ArgumentOutOfRangeException)
                {
                    threwMap = true;
                }
                catch (ArgumentException)
                {
                    threwMap = true;
                }

                if (!threwMap)
                    throw new Exception("Invalid MapFunction enum did not throw ArgumentOutOfRangeException.");

                record(new TestResultItem
                {
                    Name = "Guard_InvalidEnums_RejectedCleanly",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Invalid ScalarOperation and MapFunction enum values rejected with ArgumentOutOfRangeException."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Guard_InvalidEnums_RejectedCleanly",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestSessionDisposeMetricsZeroAndInvalidation(Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var session = SessionFactory.CreateSession(new SessionOptions { Backend = GpuBackend.CpuDebug });

                var pBuf = session.UploadPoints(new[] { new V3(1, 2, 3) }, ElementType.V3D, Residency.Device);
                var sBuf = session.UploadScalars(new[] { 42.0 }, ElementType.F64, Residency.Device);
                var hBuf = session.UploadScalars(new[] { 7.0 }, ElementType.F64, Residency.Host);

                if (session.Metrics.ActiveBytes <= 0 || session.Metrics.LiveAllocations <= 0)
                    throw new Exception("ActiveBytes or LiveAllocations not positive before Dispose.");

                session.Dispose();

                if (session.Metrics.ActiveBytes != 0)
                    throw new Exception($"ActiveBytes must be 0 after session Dispose, got {session.Metrics.ActiveBytes}");

                if (session.Metrics.LiveAllocations != 0)
                    throw new Exception($"LiveAllocations must be 0 after session Dispose, got {session.Metrics.LiveAllocations}");

                if (session.Metrics.HostResidentBytes != 0)
                    throw new Exception($"HostResidentBytes must be 0 after session Dispose, got {session.Metrics.HostResidentBytes}");

                if (!hBuf.IsDisposed || !pBuf.IsDisposed)
                    throw new Exception("Host- and device-resident buffers must report IsDisposed after session Dispose.");

                bool threwDownload = false;
                try
                {
                    session.DownloadScalars(sBuf);
                }
                catch (ObjectDisposedException)
                {
                    threwDownload = true;
                }
                catch (InvalidOperationException)
                {
                    threwDownload = true;
                }

                if (!threwDownload)
                    throw new Exception("Operations after session.Dispose() must throw ObjectDisposedException.");

                record(new TestResultItem
                {
                    Name = "Guard_SessionDispose_MetricsZero_BuffersInvalidated",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "session.Dispose() zeroes ActiveBytes/LiveAllocations and invalidates all handles."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Guard_SessionDispose_MetricsZero_BuffersInvalidated",
                    Category = "MemoryGuard",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
