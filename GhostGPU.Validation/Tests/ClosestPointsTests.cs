using System;
using GhostGPU.Core;
using GhostGPU.Validation.Models;
using GhostGPU.Validation.Reference;

namespace GhostGPU.Validation.Tests
{
    public static class ClosestPointsTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestKnownNearestPoints(session, recordResult);
            TestStrictTieBreakingLowestIndex(session, recordResult);
            TestEquidistantSymmetricTies(session, recordResult);
            TestLargeCoordinateOverflow1e155(session, recordResult);
            TestEmptyQueriesHandling(session, recordResult);
            TestNonemptyQueriesWithEmptyTargetsError(session, recordResult);
        }

        private static void TestKnownNearestPoints(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var targets = new[]
                {
                    new V3(0, 0, 0),    // index 0
                    new V3(10, 0, 0),   // index 1
                    new V3(20, 0, 0),   // index 2
                    new V3(30, 0, 0)    // index 3
                };

                var queries = new[]
                {
                    new V3(1, 0, 0),    // nearest to 0 (dist 1)
                    new V3(19, 0, 0),   // nearest to 2 (dist 1)
                    new V3(10, 5, 0),   // nearest to 1 (dist 5)
                    new V3(31, 0, 0)    // nearest to 3 (dist 1)
                };

                using var qBuf = session.UploadPoints(queries);
                using var tBuf = session.UploadPoints(targets);
                using var res = session.ClosestPoints(qBuf, tBuf);

                var outPts = session.DownloadPoints(res.Points);
                var outIdx = session.DownloadIndices(res.Indices);
                var outDst = session.DownloadScalars(res.Distances);

                if (outPts.Length != 4 || outIdx.Length != 4 || outDst.Length != 4)
                    throw new Exception("Output buffer lengths do not match query count.");

                int[] expectedIndices = { 0, 2, 1, 3 };
                double[] expectedDistances = { 1.0, 1.0, 5.0, 1.0 };

                for (int i = 0; i < 4; i++)
                {
                    if (outIdx[i] != expectedIndices[i])
                        throw new Exception($"Query {i} returned target index {outIdx[i]}, expected {expectedIndices[i]}");

                    AssertHelper.AssertClose(outDst[i], expectedDistances[i], 1e-9, $"Query {i} distance");
                    AssertHelper.AssertClose(outPts[i], targets[expectedIndices[i]], 1e-9, $"Query {i} point");
                }

                record(new TestResultItem
                {
                    Name = "ClosestPoints_KnownTargets",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Exact match on known targets, indices and distances."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ClosestPoints_KnownTargets",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestStrictTieBreakingLowestIndex(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Target 0 at (0,0,0) and Target 1 at (10,0,0)
                // Query at (5,0,0) has EXACT equal distance 5.0 to both targets
                var targets = new[] { new V3(0, 0, 0), new V3(10, 0, 0) };
                var queries = new[] { new V3(5, 0, 0) };

                using var qBuf = session.UploadPoints(queries);
                using var tBuf = session.UploadPoints(targets);
                using var res = session.ClosestPoints(qBuf, tBuf);

                var outIdx = session.DownloadIndices(res.Indices);
                var outDst = session.DownloadScalars(res.Distances);

                // REA strict less-than scan rule: lowest index 0 MUST win
                if (outIdx[0] != 0)
                    throw new Exception($"Exact distance tie failed: expected target index 0, got {outIdx[0]}");

                AssertHelper.AssertClose(outDst[0], 5.0, 1e-9, "Tie distance");

                record(new TestResultItem
                {
                    Name = "ClosestPoints_StrictTieBreaking_LowestIndex",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Exact distance tie selects lowest target index (0 over 1)."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ClosestPoints_StrictTieBreaking_LowestIndex",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestEquidistantSymmetricTies(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // 4 equidistant targets around origin: all at distance 10.0 from origin
                var targets = new[]
                {
                    new V3(10, 0, 0),   // index 0
                    new V3(-10, 0, 0),  // index 1
                    new V3(0, 10, 0),   // index 2
                    new V3(0, -10, 0)   // index 3
                };
                var queries = new[] { new V3(0, 0, 0) };

                using var qBuf = session.UploadPoints(queries);
                using var tBuf = session.UploadPoints(targets);
                using var res = session.ClosestPoints(qBuf, tBuf);

                var outIdx = session.DownloadIndices(res.Indices);
                var outDst = session.DownloadScalars(res.Distances);

                if (outIdx[0] != 0)
                    throw new Exception($"4-way tie failed: expected index 0, got {outIdx[0]}");

                AssertHelper.AssertClose(outDst[0], 10.0, 1e-9, "4-way tie distance");

                record(new TestResultItem
                {
                    Name = "ClosestPoints_EquidistantSymmetricTies",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "4-way symmetrical tie selects index 0."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ClosestPoints_EquidistantSymmetricTies",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestLargeCoordinateOverflow1e155(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Coordinates near 1e155: squaring directly produces 1e310 (+Infinity overflow)
                // Scaled length avoids squared overflow per Rhino Vector3d.GetLengthHelper REA evidence
                double scale = 1e155;
                var targets = new[]
                {
                    new V3(scale, scale, scale),              // index 0: dist 0.0
                    new V3(scale + 10.0, scale, scale)       // index 1: dist 10.0
                };
                var queries = new[]
                {
                    new V3(scale, scale, scale)
                };

                using var qBuf = session.UploadPoints(queries);
                using var tBuf = session.UploadPoints(targets);
                using var res = session.ClosestPoints(qBuf, tBuf);

                var outIdx = session.DownloadIndices(res.Indices);
                var outDst = session.DownloadScalars(res.Distances);

                AssertHelper.AssertFinite(outDst[0], "1e155 distance");

                if (outIdx[0] != 0)
                    throw new Exception($"Expected index 0 at 1e155 scale, got {outIdx[0]}");

                AssertHelper.AssertClose(outDst[0], 0.0, 1e-9, "1e155 distance to self");

                record(new TestResultItem
                {
                    Name = "ClosestPoints_LargeCoordinate_1e155_NoOverflow",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Successfully resolved nearest point at 1e155 scale without FP64 overflow."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ClosestPoints_LargeCoordinate_1e155_NoOverflow",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestEmptyQueriesHandling(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var targets = new[] { new V3(1, 2, 3) };
                using var qBuf = session.UploadPoints(Array.Empty<V3>());
                using var tBuf = session.UploadPoints(targets);
                using var res = session.ClosestPoints(qBuf, tBuf);

                var outPts = session.DownloadPoints(res.Points);
                var outIdx = session.DownloadIndices(res.Indices);
                var outDst = session.DownloadScalars(res.Distances);

                if (outPts.Length != 0 || outIdx.Length != 0 || outDst.Length != 0)
                    throw new Exception("Empty queries must yield empty outputs.");

                record(new TestResultItem
                {
                    Name = "ClosestPoints_EmptyQueries_ReturnsEmpty",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Empty query buffer produces 3 empty output buffers."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ClosestPoints_EmptyQueries_ReturnsEmpty",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestNonemptyQueriesWithEmptyTargetsError(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var queries = new[] { new V3(1, 2, 3) };
                using var qBuf = session.UploadPoints(queries);
                using var tBuf = session.UploadPoints(Array.Empty<V3>());

                bool threw = false;
                try
                {
                    using var res = session.ClosestPoints(qBuf, tBuf);
                }
                catch (ArgumentException)
                {
                    threw = true;
                }
                catch (InvalidOperationException)
                {
                    threw = true;
                }

                if (!threw)
                    throw new Exception("Nonempty queries with empty targets MUST throw an error.");

                record(new TestResultItem
                {
                    Name = "ClosestPoints_NonemptyQueries_EmptyTargets_ThrowsError",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Contract enforced: nonempty queries with empty targets throws error."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ClosestPoints_NonemptyQueries_EmptyTargets_ThrowsError",
                    Category = "ClosestPoints",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
