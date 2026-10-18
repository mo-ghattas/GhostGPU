using System;
using System.Collections.Generic;
using System.Diagnostics;
using GhostGPU.Core;
using GhostGPU.Validation.Models;
using GhostGPU.Validation.Reference;

namespace GhostGPU.Validation.Benchmarks
{
    public static class BoundedBenchmarkRunner
    {
        public static List<BenchmarkResultItem> RunBenchmarks(IGpuSession session)
        {
            var results = new List<BenchmarkResultItem>();

            results.Add(BenchmarkClosestPoints(session, 1000, 1000));
            results.Add(BenchmarkDistance(session, 10000));
            results.Add(BenchmarkMove(session, 10000));
            results.Add(BenchmarkScalarMath(session, 20000));
            results.Add(BenchmarkRemap(session, 10000));
            results.Add(BenchmarkColourRamp(session, 10000));

            return results;
        }

        private static BenchmarkResultItem BenchmarkClosestPoints(IGpuSession session, int queryCount, int targetCount)
        {
            var rnd = new Random(42);
            var queries = new V3[queryCount];
            var targets = new V3[targetCount];

            for (int i = 0; i < queryCount; i++)
                queries[i] = new V3(rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0);

            for (int i = 0; i < targetCount; i++)
                targets[i] = new V3(rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0);

            // 1. CPU Reference Run
            var cpuSw = Stopwatch.StartNew();
            var (cpuPts, cpuIdx, cpuDst) = NaiveCpuReference.ClosestPoints(queries, targets);
            cpuSw.Stop();
            double cpuDuration = cpuSw.Elapsed.TotalMilliseconds;

            // 2. Device Run (Upload, Kernel, Sync)
            using var qBuf = session.UploadPoints(queries);
            using var tBuf = session.UploadPoints(targets);

            // Warm up
            using (var warm = session.ClosestPoints(qBuf, tBuf)) { session.Synchronize(); }

            var devSw = Stopwatch.StartNew();
            using var devRes = session.ClosestPoints(qBuf, tBuf);
            session.Synchronize();
            devSw.Stop();
            double devDuration = devSw.Elapsed.TotalMilliseconds;

            // Download and compare precision
            var gpuPts = session.DownloadPoints(devRes.Points);
            var gpuIdx = session.DownloadIndices(devRes.Indices);
            var gpuDst = session.DownloadScalars(devRes.Distances);

            double maxDiff = 0.0;
            for (int i = 0; i < queryCount; i++)
            {
                double dDiff = Math.Abs(gpuDst[i] - cpuDst[i]);
                if (dDiff > maxDiff) maxDiff = dDiff;

                // When distances match, indices must match strict tie breaking
                if (dDiff < 1e-9 && gpuIdx[i] != cpuIdx[i])
                {
                    double ptDist = NaiveCpuReference.ScaledDistance(gpuPts[i], cpuPts[i]);
                    if (ptDist > maxDiff) maxDiff = ptDist;
                }
            }

            double speedup = devDuration > 0.0 ? cpuDuration / devDuration : 1.0;

            return new BenchmarkResultItem
            {
                Operation = $"ClosestPoints_{queryCount}x{targetCount}",
                ElementCount = queryCount,
                TargetCount = targetCount,
                DeviceDurationMs = devDuration,
                NaiveCpuDurationMs = cpuDuration,
                Speedup = speedup,
                MaxAbsoluteDifference = maxDiff,
                CpuReferenceDescription = NaiveCpuReference.Description
            };
        }

        private static BenchmarkResultItem BenchmarkDistance(IGpuSession session, int count)
        {
            var rnd = new Random(42);
            var ptsA = new V3[count];
            var ptsB = new V3[count];

            for (int i = 0; i < count; i++)
            {
                ptsA[i] = new V3(rnd.NextDouble() * 50.0, rnd.NextDouble() * 50.0, rnd.NextDouble() * 50.0);
                ptsB[i] = new V3(rnd.NextDouble() * 50.0, rnd.NextDouble() * 50.0, rnd.NextDouble() * 50.0);
            }

            var cpuSw = Stopwatch.StartNew();
            var cpuDst = NaiveCpuReference.Distance(ptsA, ptsB);
            cpuSw.Stop();
            double cpuDuration = cpuSw.Elapsed.TotalMilliseconds;

            using var aBuf = session.UploadPoints(ptsA);
            using var bBuf = session.UploadPoints(ptsB);

            // Warm up
            using (var warm = session.Distance(aBuf, bBuf)) { session.Synchronize(); }

            var devSw = Stopwatch.StartNew();
            using var devBuf = session.Distance(aBuf, bBuf);
            session.Synchronize();
            devSw.Stop();
            double devDuration = devSw.Elapsed.TotalMilliseconds;

            var gpuDst = session.DownloadScalars(devBuf);
            double maxDiff = 0.0;
            for (int i = 0; i < count; i++)
            {
                double diff = Math.Abs(gpuDst[i] - cpuDst[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            double speedup = devDuration > 0.0 ? cpuDuration / devDuration : 1.0;

            return new BenchmarkResultItem
            {
                Operation = $"Distance_Pairwise_{count}",
                ElementCount = count,
                TargetCount = count,
                DeviceDurationMs = devDuration,
                NaiveCpuDurationMs = cpuDuration,
                Speedup = speedup,
                MaxAbsoluteDifference = maxDiff,
                CpuReferenceDescription = NaiveCpuReference.Description
            };
        }

        private static BenchmarkResultItem BenchmarkMove(IGpuSession session, int count)
        {
            var rnd = new Random(42);
            var pts = new V3[count];
            var vecs = new V3[count];

            for (int i = 0; i < count; i++)
            {
                pts[i] = new V3(rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0, rnd.NextDouble() * 100.0);
                vecs[i] = new V3(rnd.NextDouble() * 10.0, rnd.NextDouble() * 10.0, rnd.NextDouble() * 10.0);
            }

            var cpuSw = Stopwatch.StartNew();
            var cpuPts = NaiveCpuReference.Move(pts, vecs);
            cpuSw.Stop();
            double cpuDuration = cpuSw.Elapsed.TotalMilliseconds;

            using var pBuf = session.UploadPoints(pts);
            using var vBuf = session.UploadVectors(vecs);

            // Warm up
            using (var warm = session.Move(pBuf, vBuf)) { session.Synchronize(); }

            var devSw = Stopwatch.StartNew();
            using var devBuf = session.Move(pBuf, vBuf);
            session.Synchronize();
            devSw.Stop();
            double devDuration = devSw.Elapsed.TotalMilliseconds;

            var gpuPts = session.DownloadPoints(devBuf);
            double maxDiff = 0.0;
            for (int i = 0; i < count; i++)
            {
                double d = NaiveCpuReference.ScaledDistance(gpuPts[i], cpuPts[i]);
                if (d > maxDiff) maxDiff = d;
            }

            double speedup = devDuration > 0.0 ? cpuDuration / devDuration : 1.0;

            return new BenchmarkResultItem
            {
                Operation = $"Move_PointsAndVectors_{count}",
                ElementCount = count,
                TargetCount = count,
                DeviceDurationMs = devDuration,
                NaiveCpuDurationMs = cpuDuration,
                Speedup = speedup,
                MaxAbsoluteDifference = maxDiff,
                CpuReferenceDescription = NaiveCpuReference.Description
            };
        }

        private static BenchmarkResultItem BenchmarkScalarMath(IGpuSession session, int count)
        {
            var rnd = new Random(42);
            var a = new double[count];
            var b = new double[count];

            for (int i = 0; i < count; i++)
            {
                a[i] = rnd.NextDouble() * 100.0 + 1.0;
                b[i] = rnd.NextDouble() * 100.0 + 1.0;
            }

            var cpuSw = Stopwatch.StartNew();
            var cpuRes = NaiveCpuReference.ScalarMath(a, b, ScalarOperation.Multiply);
            cpuSw.Stop();
            double cpuDuration = cpuSw.Elapsed.TotalMilliseconds;

            using var aBuf = session.UploadScalars(a);
            using var bBuf = session.UploadScalars(b);

            // Warm up
            using (var warm = session.ScalarMath(aBuf, bBuf, ScalarOperation.Multiply)) { session.Synchronize(); }

            var devSw = Stopwatch.StartNew();
            using var devBuf = session.ScalarMath(aBuf, bBuf, ScalarOperation.Multiply);
            session.Synchronize();
            devSw.Stop();
            double devDuration = devSw.Elapsed.TotalMilliseconds;

            var gpuRes = session.DownloadScalars(devBuf);
            double maxDiff = 0.0;
            for (int i = 0; i < count; i++)
            {
                double d = Math.Abs(gpuRes[i] - cpuRes[i]);
                if (d > maxDiff) maxDiff = d;
            }

            double speedup = devDuration > 0.0 ? cpuDuration / devDuration : 1.0;

            return new BenchmarkResultItem
            {
                Operation = $"ScalarMath_Multiply_{count}",
                ElementCount = count,
                TargetCount = count,
                DeviceDurationMs = devDuration,
                NaiveCpuDurationMs = cpuDuration,
                Speedup = speedup,
                MaxAbsoluteDifference = maxDiff,
                CpuReferenceDescription = NaiveCpuReference.Description
            };
        }

        private static BenchmarkResultItem BenchmarkRemap(IGpuSession session, int count)
        {
            var rnd = new Random(42);
            var s = new double[count];
            for (int i = 0; i < count; i++)
                s[i] = rnd.NextDouble() * 100.0;

            var bounds = new[] { 0.0, 100.0 };

            var cpuSw = Stopwatch.StartNew();
            var cpuRes = NaiveCpuReference.Remap(s, bounds, 10.0, 50.0, true);
            cpuSw.Stop();
            double cpuDuration = cpuSw.Elapsed.TotalMilliseconds;

            using var sBuf = session.UploadScalars(s);
            using var bBuf = session.UploadScalars(bounds);

            // Warm up
            using (var warm = session.Remap(sBuf, bBuf, 10.0, 50.0, true)) { session.Synchronize(); }

            var devSw = Stopwatch.StartNew();
            using var devBuf = session.Remap(sBuf, bBuf, 10.0, 50.0, true);
            session.Synchronize();
            devSw.Stop();
            double devDuration = devSw.Elapsed.TotalMilliseconds;

            var gpuRes = session.DownloadScalars(devBuf);
            double maxDiff = 0.0;
            for (int i = 0; i < count; i++)
            {
                double d = Math.Abs(gpuRes[i] - cpuRes[i]);
                if (d > maxDiff) maxDiff = d;
            }

            double speedup = devDuration > 0.0 ? cpuDuration / devDuration : 1.0;

            return new BenchmarkResultItem
            {
                Operation = $"Remap_{count}",
                ElementCount = count,
                TargetCount = 2,
                DeviceDurationMs = devDuration,
                NaiveCpuDurationMs = cpuDuration,
                Speedup = speedup,
                MaxAbsoluteDifference = maxDiff,
                CpuReferenceDescription = NaiveCpuReference.Description
            };
        }

        private static BenchmarkResultItem BenchmarkColourRamp(IGpuSession session, int count)
        {
            var rnd = new Random(42);
            var s = new double[count];
            for (int i = 0; i < count; i++)
                s[i] = rnd.NextDouble();

            uint low = 0xFF102030;
            uint high = 0xFFF0E0D0;

            var cpuSw = Stopwatch.StartNew();
            var cpuRes = NaiveCpuReference.ColourRamp(s, low, high);
            cpuSw.Stop();
            double cpuDuration = cpuSw.Elapsed.TotalMilliseconds;

            using var sBuf = session.UploadScalars(s);

            // Warm up
            using (var warm = session.ColourRamp(sBuf, low, high)) { session.Synchronize(); }

            var devSw = Stopwatch.StartNew();
            using var devBuf = session.ColourRamp(sBuf, low, high);
            session.Synchronize();
            devSw.Stop();
            double devDuration = devSw.Elapsed.TotalMilliseconds;

            var gpuRes = session.DownloadColours(devBuf);
            double maxDiff = 0.0;
            for (int i = 0; i < count; i++)
            {
                if (gpuRes[i] != cpuRes[i])
                {
                    double d = Math.Abs((double)gpuRes[i] - cpuRes[i]);
                    if (d > maxDiff) maxDiff = d;
                }
            }

            double speedup = devDuration > 0.0 ? cpuDuration / devDuration : 1.0;

            return new BenchmarkResultItem
            {
                Operation = $"ColourRamp_{count}",
                ElementCount = count,
                TargetCount = 1,
                DeviceDurationMs = devDuration,
                NaiveCpuDurationMs = cpuDuration,
                Speedup = speedup,
                MaxAbsoluteDifference = maxDiff,
                CpuReferenceDescription = NaiveCpuReference.Description
            };
        }
    }
}
