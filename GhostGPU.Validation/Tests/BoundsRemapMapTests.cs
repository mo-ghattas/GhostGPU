using System;
using GhostGPU.Core;
using GhostGPU.Validation.Models;

namespace GhostGPU.Validation.Tests
{
    public static class BoundsRemapMapTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestBoundsNormalSingletonAndEmpty(session, recordResult);
            TestRemapNormal(session, recordResult);
            TestRemapCollapsedSourceMidpointRea(session, recordResult);
            TestRemapReversedTargetClamping(session, recordResult);
            TestMapFunctions(session, recordResult);
            TestColourRampInterpolationAndNonfinite(session, recordResult);
            TestRepeatColoursCyclicPattern(session, recordResult);
            TestRepeatColoursEmptyError(session, recordResult);
        }

        private static void TestBoundsNormalSingletonAndEmpty(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Normal
                using var bufNormal = session.UploadScalars(new[] { 3.0, -10.0, 25.5, 0.0 });
                using var bNormal = session.Bounds(bufNormal);
                var resNormal = session.DownloadScalars(bNormal);
                if (resNormal.Length != 2 || resNormal[0] != -10.0 || resNormal[1] != 25.5)
                    throw new Exception($"Normal bounds mismatch: expected [-10, 25.5], got [{resNormal[0]}, {resNormal[1]}]");

                // Singleton
                using var bufSingle = session.UploadScalars(new[] { 42.0 });
                using var bSingle = session.Bounds(bufSingle);
                var resSingle = session.DownloadScalars(bSingle);
                if (resSingle.Length != 2 || resSingle[0] != 42.0 || resSingle[1] != 42.0)
                    throw new Exception($"Singleton bounds mismatch: expected [42, 42], got [{resSingle[0]}, {resSingle[1]}]");

                // Empty -> [NaN, NaN]
                using var bufEmpty = session.UploadScalars(Array.Empty<double>());
                using var bEmpty = session.Bounds(bufEmpty);
                var resEmpty = session.DownloadScalars(bEmpty);
                if (resEmpty.Length != 2 || !double.IsNaN(resEmpty[0]) || !double.IsNaN(resEmpty[1]))
                    throw new Exception($"Empty bounds mismatch: expected [NaN, NaN], got [{resEmpty[0]}, {resEmpty[1]}]");

                record(new TestResultItem
                {
                    Name = "Bounds_Normal_Singleton_Empty",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Normal bounds, singleton [x,x], and empty bounds [NaN, NaN] verified."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Bounds_Normal_Singleton_Empty",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestRemapNormal(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var scalars = new[] { 0.0, 5.0, 10.0 };
                var bounds = new[] { 0.0, 10.0 };

                using var sBuf = session.UploadScalars(scalars);
                using var bBuf = session.UploadScalars(bounds);
                using var rBuf = session.Remap(sBuf, bBuf, 100.0, 200.0, false);

                var res = session.DownloadScalars(rBuf);
                if (res.Length != 3) throw new Exception("Length mismatch.");

                AssertHelper.AssertClose(res[0], 100.0, 1e-9, "Remap normal 0");
                AssertHelper.AssertClose(res[1], 150.0, 1e-9, "Remap normal 1");
                AssertHelper.AssertClose(res[2], 200.0, 1e-9, "Remap normal 2");

                record(new TestResultItem
                {
                    Name = "Remap_Normal_DomainMapping",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Remap_Normal_DomainMapping",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestRemapCollapsedSourceMidpointRea(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // CRITICAL REA finding: When source domain is collapsed (min == max),
                // native Remap Numbers returns the TARGET MIDPOINT, not targetMin!
                var scalars = new[] { 5.0, 5.0, 5.0 };
                var collapsedBounds = new[] { 5.0, 5.0 };

                using var sBuf = session.UploadScalars(scalars);
                using var bBuf = session.UploadScalars(collapsedBounds);
                // Target is [10.0, 30.0], midpoint is 20.0
                using var rBuf = session.Remap(sBuf, bBuf, 10.0, 30.0, false);

                var res = session.DownloadScalars(rBuf);
                for (int i = 0; i < res.Length; i++)
                {
                    AssertHelper.AssertClose(res[i], 20.0, 1e-9, $"REA midpoint rule at {i}");
                }

                record(new TestResultItem
                {
                    Name = "Remap_CollapsedSource_MapsToTargetMidpoint_REA",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "REA/ILSpy verified: collapsed source domain maps to target domain midpoint."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Remap_CollapsedSource_MapsToTargetMidpoint_REA",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestRemapReversedTargetClamping(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Target domain reversed: targetMin=100.0, targetMax=0.0, clamp=true
                // Source bounds: [0.0, 10.0]
                // Input: -5.0 (below sMin) -> unclipped 150.0 -> clamped to 100.0 (max(0, 100) = 100)
                // Input: 5.0 (midpoint) -> 50.0
                // Input: 15.0 (above sMax) -> unclipped -50.0 -> clamped to 0.0 (min(0, 100) = 0)
                var scalars = new[] { -5.0, 5.0, 15.0 };
                var bounds = new[] { 0.0, 10.0 };

                using var sBuf = session.UploadScalars(scalars);
                using var bBuf = session.UploadScalars(bounds);

                // Unclipped
                using var unclippedBuf = session.Remap(sBuf, bBuf, 100.0, 0.0, false);
                var unclipped = session.DownloadScalars(unclippedBuf);
                AssertHelper.AssertClose(unclipped[0], 150.0, 1e-9, "Unclipped reversed remap 0");
                AssertHelper.AssertClose(unclipped[1], 50.0, 1e-9, "Unclipped reversed remap 1");
                AssertHelper.AssertClose(unclipped[2], -50.0, 1e-9, "Unclipped reversed remap 2");

                // Clipped
                using var rBuf = session.Remap(sBuf, bBuf, 100.0, 0.0, true);
                var res = session.DownloadScalars(rBuf);

                AssertHelper.AssertClose(res[0], 100.0, 1e-9, "Reversed clamp 0");
                AssertHelper.AssertClose(res[1], 50.0, 1e-9, "Reversed clamp 1");
                AssertHelper.AssertClose(res[2], 0.0, 1e-9, "Reversed clamp 2");

                record(new TestResultItem
                {
                    Name = "Remap_ReversedTarget_PreservesIntervalAndClamps",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Reversed target intervals [100, 0] properly preserved and clamped against [min, max]."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Remap_ReversedTarget_PreservesIntervalAndClamps",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestMapFunctions(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var input = new[] { -0.5, 0.0, 0.5, 1.0, 1.5 };
                using var buf = session.UploadScalars(input);

                using var lin = session.Map(buf, MapFunction.Linear);
                using var smt = session.Map(buf, MapFunction.SmoothStep);
                using var sin = session.Map(buf, MapFunction.Sine);

                var resLin = session.DownloadScalars(lin);
                var resSmt = session.DownloadScalars(smt);
                var resSin = session.DownloadScalars(sin);

                // Linear
                for (int i = 0; i < input.Length; i++)
                {
                    if (Math.Abs(resLin[i] - input[i]) > 1e-9)
                        throw new Exception($"Linear failed at {i}");
                }

                // SmoothStep: clamp(x,0,1)^2 * (3 - 2*clamp(x,0,1))
                // -0.5 -> 0.0
                // 0.0 -> 0.0
                // 0.5 -> 0.25 * 2.0 = 0.5
                // 1.0 -> 1.0
                // 1.5 -> 1.0
                double[] expSmt = { 0.0, 0.0, 0.5, 1.0, 1.0 };
                for (int i = 0; i < input.Length; i++)
                {
                    if (Math.Abs(resSmt[i] - expSmt[i]) > 1e-9)
                        throw new Exception($"SmoothStep failed at {i}: expected {expSmt[i]}, got {resSmt[i]}");
                }

                // Sine: sin(0.0) = 0.0
                if (Math.Abs(resSin[1]) > 1e-9)
                    throw new Exception("Sine(0) != 0");

                record(new TestResultItem
                {
                    Name = "Map_Linear_SmoothStep_Sine",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Linear, SmoothStep, and Sine mappings match exact definitions."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Map_Linear_SmoothStep_Sine",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestColourRampInterpolationAndNonfinite(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // low = 0xFF000000 (Black), high = 0xFFFFFFFF (White)
                uint low = 0xFF000000;
                uint high = 0xFFFFFFFF;

                // 1. Finite ramp interpolation
                var finiteScalars = new[] { 0.0, 1.0, 0.5 };
                using var finiteBuf = session.UploadScalars(finiteScalars);
                using var finiteColourBuf = session.ColourRamp(finiteBuf, low, high);
                var finiteRes = session.DownloadColours(finiteColourBuf);

                if (finiteRes[0] != 0xFF000000)
                    throw new Exception($"At 0.0: expected 0xFF000000, got 0x{finiteRes[0]:X8}");
                if (finiteRes[1] != 0xFFFFFFFF)
                    throw new Exception($"At 1.0: expected 0xFFFFFFFF, got 0x{finiteRes[1]:X8}");

                // 0.5 lerp between 0 and 255: 127.5 rounded with floor(x+0.5) = 128 = 0x80
                if (finiteRes[2] != 0xFF808080)
                    throw new Exception($"At 0.5: expected 0xFF808080, got 0x{finiteRes[2]:X8}");

                // 2. Generate nonfinite scalars on device using finite inputs divided by zero
                // 0.0 / 0.0 = NaN, 1.0 / 0.0 = +Infinity, -1.0 / 0.0 = -Infinity
                // UploadScalars strictly rejects NaNs, but device operations may produce them
                var numerators = new[] { 0.0, 1.0, -1.0 };
                var denominators = new[] { 0.0, 0.0, 0.0 };
                using var numBuf = session.UploadScalars(numerators);
                using var denBuf = session.UploadScalars(denominators);
                using var nonfiniteBuf = session.ScalarMath(numBuf, denBuf, ScalarOperation.Divide);

                // Pass device-resident nonfinite buffer to ColourRamp: must use lowArgb per contract
                using var nonfiniteColourBuf = session.ColourRamp(nonfiniteBuf, low, high);
                var nonfiniteRes = session.DownloadColours(nonfiniteColourBuf);

                if (nonfiniteRes[0] != low)
                    throw new Exception($"NaN colour must use lowArgb: expected 0x{low:X8}, got 0x{nonfiniteRes[0]:X8}");
                if (nonfiniteRes[1] != low)
                    throw new Exception($"+Infinity colour must use lowArgb: expected 0x{low:X8}, got 0x{nonfiniteRes[1]:X8}");
                if (nonfiniteRes[2] != low)
                    throw new Exception($"-Infinity colour must use lowArgb: expected 0x{low:X8}, got 0x{nonfiniteRes[2]:X8}");

                record(new TestResultItem
                {
                    Name = "ColourRamp_ByteLerp_Rounding_NonfiniteFallback",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Linear ARGB byte interpolation, floor(v+0.5) rounding, and device nonfinite fallback to lowArgb verified."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ColourRamp_ByteLerp_Rounding_NonfiniteFallback",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestRepeatColoursCyclicPattern(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Native Mesh Colours REA requirement:
                // Accepts an ARGB colour pattern and repeats it cyclically (WrapInteger), NOT repeat-last!
                // Pattern has 2 colours: Red, Green. Requested count = 5.
                // Expected: Red, Green, Red, Green, Red.
                uint red = 0xFFFF0000;
                uint green = 0xFF00FF00;
                var pattern = new[] { red, green };

                using var pBuf = session.UploadColours(pattern);
                using var rBuf = session.RepeatColours(pBuf, 5);

                var res = session.DownloadColours(rBuf);
                if (res.Length != 5)
                    throw new Exception($"Expected 5 colours, got {res.Length}");

                uint[] expected = { red, green, red, green, red };
                for (int i = 0; i < 5; i++)
                {
                    if (res[i] != expected[i])
                        throw new Exception($"Cyclic wrap at {i} failed: expected 0x{expected[i]:X8}, got 0x{res[i]:X8}");
                }

                // Count 0 returns empty
                using var emptyOut = session.RepeatColours(pBuf, 0);
                if (emptyOut.Count != 0)
                    throw new Exception("RepeatColours with count 0 must return empty buffer.");

                record(new TestResultItem
                {
                    Name = "RepeatColours_CyclicWrapPattern_REA",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Confirmed REA Mesh Colours cyclic modulo wrap for pattern repeating."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "RepeatColours_CyclicWrapPattern_REA",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestRepeatColoursEmptyError(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                using var emptyPattern = session.UploadColours(Array.Empty<uint>());

                bool threw = false;
                try
                {
                    using var res = session.RepeatColours(emptyPattern, 5);
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
                    throw new Exception("Empty pattern with count > 0 must throw an error.");

                record(new TestResultItem
                {
                    Name = "RepeatColours_EmptyPattern_NonzeroCount_ThrowsError",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "RepeatColours_EmptyPattern_NonzeroCount_ThrowsError",
                    Category = "BoundsRemapMap",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
