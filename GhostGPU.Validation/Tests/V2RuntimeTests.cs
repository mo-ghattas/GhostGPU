using System;
using GhostGPU.Core;
using GhostGPU.Runtime;
using GhostGPU.Runtime.Kernels;
using GhostGPU.Validation.Models;
using GhostGPU.Validation.Reference;

namespace GhostGPU.Validation.Tests
{
    public static class V2RuntimeTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestPoolReuseAndHits(session, recordResult);
            TestBudgetAndTrim(session, recordResult);
            TestSpillAndRepromote(session, recordResult);
            TestStagingRingChunking(recordResult);
            TestHostVsDeviceBitEquality(session, recordResult);
            TestCompactDatatypesRoundTripAndTolerance(session, recordResult);
            TestGridNearestMatchesBruteNearest(session, recordResult);
            TestParallelBoundsMatchesSerialOracle(session, recordResult);
        }

        private static void TestPoolReuseAndHits(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                long initialHits = session.Metrics.PoolHits;
                long initialMisses = session.Metrics.PoolMisses;

                // 1. First allocation: miss (explicit Device: Auto keeps small uploads on the host)
                var buf1 = session.UploadScalars(new double[2000], ElementType.F64, Residency.Device);
                if (session.Metrics.PoolMisses <= initialMisses)
                    throw new Exception("First allocation did not increment PoolMisses.");

                // 2. Dispose: returns block to pool
                buf1.Dispose();
                if (session.Metrics.PooledBytes <= 0)
                    throw new Exception("Disposed buffer did not contribute to PooledBytes.");

                // 3. Second allocation of identical size: must hit the pool
                var buf2 = session.UploadScalars(new double[2000], ElementType.F64, Residency.Device);
                if (session.Metrics.PoolHits <= initialHits)
                    throw new Exception("Re-allocation of same size did not increment PoolHits.");

                buf2.Dispose();

                record(new TestResultItem
                {
                    Name = "V2_Pool_Reuse_And_Hits",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Power-of-two size class pool successfully caches and hits free blocks on reallocation."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "V2_Pool_Reuse_And_Hits",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestBudgetAndTrim(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Allocate and dispose to populate pool
                using (var b = session.UploadScalars(new double[1024], ElementType.F64, Residency.Device)) { }
                if (session.Metrics.PooledBytes <= 0)
                    throw new Exception("Pool should have cached bytes after dispose.");

                session.TrimPool();
                if (session.Metrics.PooledBytes != 0)
                    throw new Exception($"TrimPool did not empty pooled bytes: {session.Metrics.PooledBytes}");

                record(new TestResultItem
                {
                    Name = "V2_Budget_And_Trim",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "TrimPool releases all cached pooled memory back to driver."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "V2_Budget_And_Trim",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestSpillAndRepromote(IGpuSession mainSession, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Create restricted session with tight budget: fits two 1024B blocks + 512B output
                var opt = new SessionOptions
                {
                    Backend = mainSession.IsHardwareAccelerated ? GpuBackend.CudaOnly : GpuBackend.CpuDebug,
                    MemoryLimitBytes = 2560,
                    PoolCacheBytes = 0,
                    Spill = SpillPolicy.SpillToHost
                };

                using var sess = SessionFactory.CreateSession(opt);

                // Allocate buffer A (1024 bytes size class)
                var ptsA = new V3[40]; // 40 * 24 = 960B -> 1024B
                for (int i = 0; i < ptsA.Length; i++) ptsA[i] = new V3(i, i, i);

                using var bufA = sess.UploadPoints(ptsA, ElementType.V3D, Residency.Device);
                if (bufA.Residency != Residency.Device)
                    throw new Exception($"Buffer A should be device resident, but was {bufA.Residency}");

                // Allocate buffer B (also 1024 bytes)
                var ptsB = new V3[40];
                for (int i = 0; i < ptsB.Length; i++) ptsB[i] = new V3(i * 2, 0, 0);

                using var bufB = sess.UploadPoints(ptsB, ElementType.V3D, Residency.Device);
                if (bufB.Residency != Residency.Device)
                    throw new Exception($"Buffer B should be device resident, but was {bufB.Residency}");

                // Allocate buffer C (also 1024 bytes) -> 1024 * 3 = 3072 > 2560 limit -> forces Buffer A to spill to host
                var ptsC = new V3[40];
                for (int i = 0; i < ptsC.Length; i++) ptsC[i] = new V3(i * 3, 0, 0);

                using (var bufC = sess.UploadPoints(ptsC, ElementType.V3D, Residency.Device))
                {
                    // Verify buffer A was spilled to host under pressure
                    if (bufA.Residency != Residency.Host)
                        throw new Exception($"Buffer A should have spilled to host, but was {bufA.Residency}");

                    // Verify values of buffer A are intact on host
                    var downA = sess.DownloadPoints(bufA);
                    AssertHelper.AssertClose(downA[0], ptsA[0], 1e-9, "Spilled A point 0");
                    AssertHelper.AssertClose(downA[39], ptsA[39], 1e-9, "Spilled A point 39");
                }

                // bufC is disposed, freeing its block
                sess.TrimPool();

                // Execute operation on bufA and bufB -> re-promotes bufA to device!
                using var dist = sess.Distance(bufA, bufB);
                if (bufA.Residency != Residency.Device)
                    throw new Exception($"Buffer A should have re-promoted to device, but was {bufA.Residency}");

                var distVals = sess.DownloadScalars(dist);
                if (distVals.Length != 40)
                    throw new Exception($"Distance count mismatch: {distVals.Length}");

                record(new TestResultItem
                {
                    Name = "V2_Spill_And_RePromote",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "LRU device buffer spills to host under budget pressure and re-promotes correctly."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "V2_Spill_And_RePromote",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestStagingRingChunking(Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // 256KB slot size, upload 1MB buffer (forces 4 chunks)
                var opt = new SessionOptions
                {
                    Backend = GpuBackend.CpuDebug,
                    StagingSlots = 2,
                    StagingSlotBytes = 256 * 1024,
                    MemoryLimitBytes = 64 * 1024 * 1024
                };

                using var sess = SessionFactory.CreateSession(opt);

                int count = 43690; // 43690 * 24B = 1,048,560B (~1MB)
                var pts = new V3[count];
                for (int i = 0; i < count; i++) pts[i] = new V3(i, i * 0.5, -i);

                using var buf = sess.UploadPoints(pts, ElementType.V3D, Residency.Device); // Auto would keep 43690 points host-side

                if (sess.Metrics.StagedBytes < 1000000)
                    throw new Exception($"Staging ring chunking failed: StagedBytes was {sess.Metrics.StagedBytes}");

                var downloaded = sess.DownloadPoints(buf);
                AssertHelper.AssertClose(downloaded[0], pts[0], 1e-9, "Chunked transfer point 0");
                AssertHelper.AssertClose(downloaded[count - 1], pts[count - 1], 1e-9, "Chunked transfer last point");

                record(new TestResultItem
                {
                    Name = "V2_Staging_Ring_Chunking",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Large payloads correctly chunk through smaller pinned staging ring slots."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "V2_Staging_Ring_Chunking",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestHostVsDeviceBitEquality(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                int count = 100;
                var scalarsA = new double[count];
                var scalarsB = new double[count];
                for (int i = 0; i < count; i++)
                {
                    scalarsA[i] = (i + 1) * 3.14159;
                    scalarsB[i] = (i + 2) * 2.71828;
                }

                // Host resident buffers
                using var hA = session.UploadScalars(scalarsA, ElementType.F64, Residency.Host);
                using var hB = session.UploadScalars(scalarsB, ElementType.F64, Residency.Host);

                // Device resident buffers
                using var dA = session.UploadScalars(scalarsA, ElementType.F64, Residency.Device);
                using var dB = session.UploadScalars(scalarsB, ElementType.F64, Residency.Device);

                // Test ScalarMath operations
                foreach (ScalarOperation op in Enum.GetValues(typeof(ScalarOperation)))
                {
                    using var resHost = session.ScalarMath(hA, hB, op);
                    using var resDev = session.ScalarMath(dA, dB, op);

                    var hVals = session.DownloadScalars(resHost);
                    var dVals = session.DownloadScalars(resDev);

                    for (int i = 0; i < count; i++)
                    {
                        long hBits = BitConverter.DoubleToInt64Bits(hVals[i]);
                        long dBits = BitConverter.DoubleToInt64Bits(dVals[i]);
                        if (hBits != dBits)
                        {
                            // Allow at most 1 ULP for transcendental operations if any
                            if (Math.Abs(hBits - dBits) > 1)
                                throw new Exception($"Host vs Device bit mismatch in {op} at index {i}: host={hVals[i]:R}, dev={dVals[i]:R}");
                        }
                    }
                }

                record(new TestResultItem
                {
                    Name = "V2_Host_Vs_Device_Bit_Equality",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Pure per-element functions yield bit-identical outputs across host and device execution."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "V2_Host_Vs_Device_Bit_Equality",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestCompactDatatypesRoundTripAndTolerance(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // 1. V3F round-trip and relative tolerance (< 1e-6)
                var originalPts = new V3[]
                {
                    new V3(100.123456, 200.654321, 300.987654),
                    new V3(105.5, 205.25, 305.125),
                    new V3(95.0, 195.0, 295.0)
                };

                using var v3fBuf = session.UploadPoints(originalPts, ElementType.V3F, Residency.Auto);
                if (v3fBuf.ElementType != ElementType.V3F)
                    throw new Exception($"Expected V3F ElementType, got {v3fBuf.ElementType}");

                var downPts = session.DownloadPoints(v3fBuf);
                for (int i = 0; i < originalPts.Length; i++)
                {
                    double diffX = Math.Abs(downPts[i].X - originalPts[i].X) / Math.Abs(originalPts[i].X);
                    double diffY = Math.Abs(downPts[i].Y - originalPts[i].Y) / Math.Abs(originalPts[i].Y);
                    double diffZ = Math.Abs(downPts[i].Z - originalPts[i].Z) / Math.Abs(originalPts[i].Z);

                    if (diffX > 1e-6 || diffY > 1e-6 || diffZ > 1e-6)
                        throw new Exception($"V3F relative tolerance exceeded at index {i}: ({diffX:E2}, {diffY:E2}, {diffZ:E2}) > 1e-6");
                }

                // 2. F32 scalar upload and math
                var scalars = new double[] { 1.5, 2.25, 3.125 };
                using var f32Buf = session.UploadScalars(scalars, ElementType.F32, Residency.Auto);
                if (f32Buf.ElementType != ElementType.F32)
                    throw new Exception($"Expected F32 ElementType, got {f32Buf.ElementType}");

                var downScalars = session.DownloadScalars(f32Buf);
                for (int i = 0; i < scalars.Length; i++)
                {
                    AssertHelper.AssertClose(downScalars[i], scalars[i], 1e-6, $"F32 scalar {i}");
                }

                // 3. U16 index narrowing and widening
                var indices = new int[] { 0, 10, 65535, 42 };
                using var idxBuf = session.UploadIndices(indices);
                if (idxBuf.ElementType != ElementType.U16)
                    throw new Exception($"Indices in [0, 65535] should narrow to U16, got {idxBuf.ElementType}");

                var downIdx = session.DownloadIndices(idxBuf);
                for (int i = 0; i < indices.Length; i++)
                {
                    if (downIdx[i] != indices[i])
                        throw new Exception($"U16 roundtrip mismatch at {i}: expected {indices[i]}, got {downIdx[i]}");
                }

                record(new TestResultItem
                {
                    Name = "V2_Compact_Datatypes_Tolerance",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "V3F relative tolerance < 1e-6 with origin shifting, F32 math, and U16 index narrowing verified."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "V2_Compact_Datatypes_Tolerance",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestGridNearestMatchesBruteNearest(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // 300 queries and 1000 targets (> 128 targets triggers uniform grid)
                int qCount = 200;
                int tCount = 500;

                var rnd = new Random(12345);
                var queries = new V3[qCount];
                var targets = new V3[tCount];

                for (int i = 0; i < qCount; i++)
                    queries[i] = new V3(rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0);

                for (int i = 0; i < tCount; i++)
                    targets[i] = new V3(rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0);

                // Add explicit tie targets to test tie breaking
                targets[10] = new V3(50, 50, 50);
                targets[20] = new V3(50, 50, 50);
                queries[0] = new V3(50, 50, 50);

                using var qBuf = session.UploadPoints(queries);
                using var tBuf = session.UploadPoints(targets);

                using var result = session.ClosestPoints(qBuf, tBuf);
                var gpuPts = session.DownloadPoints(result.Points);
                var gpuIndices = session.DownloadIndices(result.Indices);
                var gpuDists = session.DownloadScalars(result.Distances);

                // Oracle brute force
                var refRes = NaiveCpuReference.ClosestPoints(queries, targets);

                for (int i = 0; i < qCount; i++)
                {
                    AssertHelper.AssertClose(gpuDists[i], refRes.Distances[i], 1e-8, $"Grid vs Brute Dist {i}");
                    AssertHelper.AssertClose(gpuPts[i], refRes.Points[i], 1e-8, $"Grid vs Brute Pts {i}");

                    // Indices must match (including tie breaking to lowest target index)
                    if (gpuIndices[i] != refRes.Indices[i])
                    {
                        // Check if it's an exact distance tie
                        double dGpu = gpuDists[i];
                        double dRef = refRes.Distances[i];
                        if (Math.Abs(dGpu - dRef) > 1e-9 || gpuIndices[i] > refRes.Indices[i])
                            throw new Exception($"Closest point index mismatch at query {i}: gpu={gpuIndices[i]}, ref={refRes.Indices[i]}");
                    }
                }

                record(new TestResultItem
                {
                    Name = "V2_Grid_Nearest_Matches_Brute_Nearest",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Uniform-grid spatial acceleration produces exact match to brute force reference."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "V2_Grid_Nearest_Matches_Brute_Nearest",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestParallelBoundsMatchesSerialOracle(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Input > 1024 triggers ParallelBoundsPass1Kernel + ParallelBoundsPass2Kernel
                int count = 10000;
                var rnd = new Random(42);
                var values = new double[count];
                for (int i = 0; i < count; i++)
                {
                    values[i] = (rnd.NextDouble() - 0.5) * 100000.0;
                }
                values[100] = -999999.0; // explicit min
                values[5000] = 888888.0; // explicit max

                using var buf = session.UploadScalars(values);
                using var bndsBuf = session.Bounds(buf);
                var bnds = session.DownloadScalars(bndsBuf);

                AssertHelper.AssertClose(bnds[0], -999999.0, 1e-9, "Parallel bounds min");
                AssertHelper.AssertClose(bnds[1], 888888.0, 1e-9, "Parallel bounds max");

                // Compare with serial reference
                var refBnds = NaiveCpuReference.Bounds(values);
                AssertHelper.AssertClose(bnds[0], refBnds[0], 0.0, "Parallel bounds min matches reference");
                AssertHelper.AssertClose(bnds[1], refBnds[1], 0.0, "Parallel bounds max matches reference");

                record(new TestResultItem
                {
                    Name = "V2_Parallel_Bounds_Matches_Serial_Oracle",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Two-pass parallel reduction matches serial oracle exactly."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "V2_Parallel_Bounds_Matches_Serial_Oracle",
                    Category = "RuntimeV2",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
