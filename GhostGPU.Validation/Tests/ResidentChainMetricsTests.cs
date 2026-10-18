using System;
using GhostGPU.Core;
using GhostGPU.Validation.Models;

namespace GhostGPU.Validation.Tests
{
    public static class ResidentChainMetricsTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestResidentPipelineZeroIntermediateDownloads(session, recordResult);
        }

        private static void TestResidentPipelineZeroIntermediateDownloads(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                long downloadedBefore = session.Metrics.DownloadedBytes;

                // Build a 6-stage resident pipeline:
                // 1. Upload points & vectors
                var pts = new[] { new V3(0, 0, 0), new V3(1, 2, 3), new V3(4, 5, 6) };
                var vecs = new[] { new V3(1, 1, 1) };
                var targets = new[] { new V3(0, 0, 0), new V3(10, 10, 10) };

                using var pBuf = session.UploadPoints(pts);
                using var vBuf = session.UploadVectors(vecs);
                using var tBuf = session.UploadPoints(targets);

                // 2. Move points by vector
                using var movedBuf = session.Move(pBuf, vBuf);

                // 3. Closest point queries
                using var nearestRes = session.ClosestPoints(movedBuf, tBuf);

                // 4. Pairwise distances
                using var distBuf = session.Distance(movedBuf, tBuf);

                // 5. Bounds of distance
                using var boundsBuf = session.Bounds(distBuf);

                // 6. Remap distances
                using var remapBuf = session.Remap(distBuf, boundsBuf, 0.0, 1.0, true);

                // 7. Colour ramp
                using var colourBuf = session.ColourRamp(remapBuf, 0xFF000000, 0xFFFFFFFF);

                // Synchronize device execution
                session.Synchronize();

                long downloadedAfterPipeline = session.Metrics.DownloadedBytes;

                // CRITICAL CONTRACT CHECK: No intermediate downloads allowed
                if (downloadedAfterPipeline != downloadedBefore)
                {
                    throw new Exception($"Resident pipeline violated zero-download contract: DownloadedBytes before={downloadedBefore}, after={downloadedAfterPipeline} (leaked {downloadedAfterPipeline - downloadedBefore} bytes to host)!");
                }

                // Explicit readback boundary: download only final result
                var finalColours = session.DownloadColours(colourBuf);
                long downloadedAfterExplicitReadback = session.Metrics.DownloadedBytes;

                if (downloadedAfterExplicitReadback <= downloadedAfterPipeline)
                {
                    throw new Exception("Explicit DownloadColours did not increment DownloadedBytes counter!");
                }

                if (finalColours.Length != 3)
                {
                    throw new Exception($"Expected 3 colours, got {finalColours.Length}");
                }

                record(new TestResultItem
                {
                    Name = "ResidentPipeline_ZeroIntermediateDownloads_ExplicitBoundary",
                    Category = "ResidentMetrics",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Zero host readbacks across 6-stage resident kernel chain. DownloadedBytes incremented strictly upon explicit download."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ResidentPipeline_ZeroIntermediateDownloads_ExplicitBoundary",
                    Category = "ResidentMetrics",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
