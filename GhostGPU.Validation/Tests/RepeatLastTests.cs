using System;
using GhostGPU.Core;
using GhostGPU.Validation.Models;

namespace GhostGPU.Validation.Tests
{
    public static class RepeatLastTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestScalarMathRepeatLast(session, recordResult);
            TestMoveRepeatLast(session, recordResult);
            TestDistanceRepeatLast(session, recordResult);
            TestAmplitudeRepeatLast(session, recordResult);
            TestEitherEmptyGivesEmpty(session, recordResult);
        }

        private static void TestScalarMathRepeatLast(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // A has 4 elements, B has 2 elements: B repeats last element (20.0)
                var a = new[] { 1.0, 2.0, 3.0, 4.0 };
                var b = new[] { 10.0, 20.0 };

                using var bufA = session.UploadScalars(a);
                using var bufB = session.UploadScalars(b);
                using var sum = session.ScalarMath(bufA, bufB, ScalarOperation.Add);

                var result = session.DownloadScalars(sum);
                if (result.Length != 4)
                    throw new Exception($"Expected 4 elements, got {result.Length}");

                double[] expected = { 11.0, 22.0, 23.0, 24.0 };
                for (int i = 0; i < 4; i++)
                {
                    if (Math.Abs(result[i] - expected[i]) > 1e-9)
                        throw new Exception($"At index {i}: expected {expected[i]}, got {result[i]}");
                }

                // Reverse: A has 1 element (5.0), B has 3 elements [1, 2, 3] -> A repeats 5.0
                using var bufA1 = session.UploadScalars(new[] { 5.0 });
                using var bufB3 = session.UploadScalars(new[] { 1.0, 2.0, 3.0 });
                using var prod = session.ScalarMath(bufA1, bufB3, ScalarOperation.Multiply);

                var prodResult = session.DownloadScalars(prod);
                if (prodResult.Length != 3)
                    throw new Exception("Product length mismatch.");

                double[] expectedProd = { 5.0, 10.0, 15.0 };
                for (int i = 0; i < 3; i++)
                {
                    if (Math.Abs(prodResult[i] - expectedProd[i]) > 1e-9)
                        throw new Exception($"Product at index {i}: expected {expectedProd[i]}, got {prodResult[i]}");
                }

                record(new TestResultItem
                {
                    Name = "RepeatLast_ScalarMath_LongestList",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Shorter buffer correctly repeats last element in both left and right positions."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "RepeatLast_ScalarMath_LongestList",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestMoveRepeatLast(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var points = new[]
                {
                    new V3(0, 0, 0),
                    new V3(1, 1, 1),
                    new V3(2, 2, 2)
                };
                var vectors = new[]
                {
                    new V3(10, 0, 0) // Single vector repeated for all 3 points
                };

                using var pBuf = session.UploadPoints(points);
                using var vBuf = session.UploadVectors(vectors);
                using var moved = session.Move(pBuf, vBuf);

                var result = session.DownloadPoints(moved);
                if (result.Length != 3)
                    throw new Exception($"Expected 3 points, got {result.Length}");

                V3[] expected =
                {
                    new V3(10, 0, 0),
                    new V3(11, 1, 1),
                    new V3(12, 2, 2)
                };

                for (int i = 0; i < 3; i++)
                {
                    if (Math.Abs(result[i].X - expected[i].X) > 1e-9 ||
                        Math.Abs(result[i].Y - expected[i].Y) > 1e-9 ||
                        Math.Abs(result[i].Z - expected[i].Z) > 1e-9)
                        throw new Exception($"Moved point {i} mismatch: got ({result[i].X},{result[i].Y},{result[i].Z})");
                }

                record(new TestResultItem
                {
                    Name = "RepeatLast_Move_PointsWithSingleVector",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Single motion vector applied to all input points."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "RepeatLast_Move_PointsWithSingleVector",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestDistanceRepeatLast(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var ptsA = new[] { new V3(0, 0, 0), new V3(3, 0, 0), new V3(0, 4, 0) };
                var ptsB = new[] { new V3(0, 0, 0) }; // origin, repeated for all 3

                using var aBuf = session.UploadPoints(ptsA);
                using var bBuf = session.UploadPoints(ptsB);
                using var distBuf = session.Distance(aBuf, bBuf);

                var dists = session.DownloadScalars(distBuf);
                if (dists.Length != 3)
                    throw new Exception("Distance length mismatch.");

                double[] expected = { 0.0, 3.0, 4.0 };
                for (int i = 0; i < 3; i++)
                {
                    if (Math.Abs(dists[i] - expected[i]) > 1e-9)
                        throw new Exception($"Distance at {i} mismatch: expected {expected[i]}, got {dists[i]}");
                }

                record(new TestResultItem
                {
                    Name = "RepeatLast_Distance_OriginPairing",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "RepeatLast_Distance_OriginPairing",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestAmplitudeRepeatLast(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var vecs = new[] { new V3(10, 0, 0), new V3(0, 20, 0) };
                var mags = new[] { 5.0 }; // repeated for both

                using var vBuf = session.UploadVectors(vecs);
                using var mBuf = session.UploadScalars(mags);
                using var ampBuf = session.Amplitude(vBuf, mBuf);

                var res = session.DownloadVectors(ampBuf);
                if (res.Length != 2)
                    throw new Exception("Amplitude length mismatch.");

                if (Math.Abs(res[0].X - 5.0) > 1e-9 || Math.Abs(res[1].Y - 5.0) > 1e-9)
                    throw new Exception("Amplitude values mismatch.");

                record(new TestResultItem
                {
                    Name = "RepeatLast_Amplitude_SingleMagnitude",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "RepeatLast_Amplitude_SingleMagnitude",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestEitherEmptyGivesEmpty(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                using var nonempty = session.UploadScalars(new[] { 1.0, 2.0, 3.0 });
                using var empty = session.UploadScalars(Array.Empty<double>());

                using var res1 = session.ScalarMath(nonempty, empty, ScalarOperation.Add);
                using var res2 = session.ScalarMath(empty, nonempty, ScalarOperation.Add);
                using var res3 = session.ScalarMath(empty, empty, ScalarOperation.Add);

                if (res1.Count != 0 || res2.Count != 0 || res3.Count != 0)
                    throw new Exception("Either empty buffer in pairwise operation must yield empty result.");

                record(new TestResultItem
                {
                    Name = "RepeatLast_EitherEmpty_YieldsEmpty",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Pairwise operations with either operand empty cleanly return empty buffer."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "RepeatLast_EitherEmpty_YieldsEmpty",
                    Category = "RepeatLast",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
