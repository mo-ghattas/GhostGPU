using System;
using GhostGPU.Core;
using GhostGPU.Validation.Models;

namespace GhostGPU.Validation.Tests
{
    public static class LeaseLifecycleTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestRetainIndependentLease(session, recordResult);
            TestDoubleDisposeSafety(session, recordResult);
            TestDisposedBufferRejection(session, recordResult);
            TestForeignSessionBufferRejection(session, recordResult);
            TestSessionResetInvalidatesBuffers(session, recordResult);
        }

        private static void TestRetainIndependentLease(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var input = new[] { new V3(1, 2, 3), new V3(4, 5, 6) };
                var bufA = session.UploadPoints(input);
                var bufB = bufA.Retain();

                if (bufB.Count != bufA.Count)
                    throw new Exception("Retained buffer count mismatch.");

                if (bufB.IsDisposed)
                    throw new Exception("Retained buffer marked disposed.");

                // Dispose original lease A
                bufA.Dispose();
                if (!bufA.IsDisposed)
                    throw new Exception("Buffer A should be marked disposed.");

                // Lease B must remain completely valid and intact
                if (bufB.IsDisposed)
                    throw new Exception("Buffer B should NOT be disposed after A was disposed.");

                var downloadedB = session.DownloadPoints(bufB);
                if (downloadedB.Length != 2 || downloadedB[0].X != 1 || downloadedB[1].Z != 6)
                    throw new Exception("Downloaded data from retained buffer B corrupted after A disposed.");

                // Now dispose B
                bufB.Dispose();
                if (!bufB.IsDisposed)
                    throw new Exception("Buffer B should be marked disposed.");

                record(new TestResultItem
                {
                    Name = "Lease_Retain_IndependentLifetime",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Retained lease remains fully accessible after original lease is disposed."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Lease_Retain_IndependentLifetime",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestDoubleDisposeSafety(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var buf = session.UploadScalars(new[] { 1.0, 2.0 });
                buf.Dispose();
                buf.Dispose(); // second dispose must be a harmless no-op
                buf.Dispose(); // third dispose

                record(new TestResultItem
                {
                    Name = "Lease_DoubleDispose_SafeNoOp",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Multiple Dispose() calls on the same handle do not throw."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Lease_DoubleDispose_SafeNoOp",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestDisposedBufferRejection(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var buf = session.UploadPoints(new[] { new V3(0, 0, 0) });
                buf.Dispose();

                bool threw = false;
                try
                {
                    session.DownloadPoints(buf);
                }
                catch (ObjectDisposedException)
                {
                    threw = true;
                }
                catch (InvalidOperationException)
                {
                    threw = true;
                }

                if (!threw)
                    throw new Exception("Using a disposed buffer must throw ObjectDisposedException or InvalidOperationException.");

                record(new TestResultItem
                {
                    Name = "Lease_DisposedBuffer_RejectedOnUse",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Operations on disposed handles are rejected."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Lease_DisposedBuffer_RejectedOnUse",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestForeignSessionBufferRejection(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Create a second session to produce a foreign buffer
                using var foreignSession = SessionFactory.CreateNaiveCpuReference();
                using var foreignBuf = foreignSession.UploadVectors(new[] { new V3(1, 0, 0) });

                bool threw = false;
                try
                {
                    using var res = session.Unitize(foreignBuf);
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
                    throw new Exception("Foreign session buffer must be rejected before kernel dispatch.");

                record(new TestResultItem
                {
                    Name = "Lease_ForeignSessionBuffer_Rejected",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Buffers belonging to other sessions are rejected with clear exception."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Lease_ForeignSessionBuffer_Rejected",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestSessionResetInvalidatesBuffers(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var buf = session.UploadScalars(new[] { 10.0, 20.0 });
                session.Reset();

                bool threw = false;
                try
                {
                    session.DownloadScalars(buf);
                }
                catch (ObjectDisposedException)
                {
                    threw = true;
                }
                catch (InvalidOperationException)
                {
                    threw = true;
                }

                if (!threw)
                    throw new Exception("Buffer created before Reset() must be invalidated after Reset().");

                record(new TestResultItem
                {
                    Name = "Lease_SessionReset_InvalidatesOldBuffers",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Session.Reset() cleanly invalidates all outstanding buffer handles."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Lease_SessionReset_InvalidatesOldBuffers",
                    Category = "LeaseLifecycle",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
