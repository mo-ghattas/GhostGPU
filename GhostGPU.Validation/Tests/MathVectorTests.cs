using System;
using GhostGPU.Core;
using GhostGPU.Validation.Models;

namespace GhostGPU.Validation.Tests
{
    public static class MathVectorTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestScaledDistanceNormalAndLarge(session, recordResult);
            TestZeroVectorUnitize(session, recordResult);
            TestAmplitudeUnitizeMultiplyNegative(session, recordResult);
            TestZeroVectorAmplitude(session, recordResult);
            TestVectorBetween(session, recordResult);
            TestMove(session, recordResult);
            TestScalarMathAllOperations(session, recordResult);
            TestScalarMathDivideByZeroIeee(session, recordResult);
            TestSubnormalAndGiantVectorsNormalization(session, recordResult);
        }

        private static void TestScaledDistanceNormalAndLarge(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Normal 3-4-5 triangle and 1e155 coordinate vector
                var ptsA = new[] { new V3(0, 0, 0), new V3(1e155, 0, 0) };
                var ptsB = new[] { new V3(3, 4, 0), new V3(1e155, 3e150, 4e150) };

                using var bA = session.UploadPoints(ptsA);
                using var bB = session.UploadPoints(ptsB);
                using var distBuf = session.Distance(bA, bB);

                var dists = session.DownloadScalars(distBuf);
                if (dists.Length != 2)
                    throw new Exception("Output length mismatch.");

                AssertHelper.AssertClose(dists[0], 5.0, 1e-9, "Normal distance");
                AssertHelper.AssertFinite(dists[1], "Large distance 1e155");

                double expectedLarge = 5e150;
                double relErr = Math.Abs((dists[1] - expectedLarge) / expectedLarge);
                if (relErr > 1e-9)
                    throw new Exception($"Large distance mismatch: expected {expectedLarge}, got {dists[1]} (relErr {relErr})");

                record(new TestResultItem
                {
                    Name = "Distance_Scaled_NoOverflow_1e155",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Rhino Vector3d.GetLengthHelper scaled distance avoids squaring overflow at 1e155 scale."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Distance_Scaled_NoOverflow_1e155",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestZeroVectorUnitize(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // API contract: Zero vector Unitize returns zero vector (0, 0, 0), not NaN
                var vectors = new[] { new V3(0, 0, 0), new V3(0, 5, 0) };
                using var vBuf = session.UploadVectors(vectors);
                using var uBuf = session.Unitize(vBuf);

                var result = session.DownloadVectors(uBuf);
                if (result.Length != 2)
                    throw new Exception("Length mismatch.");

                AssertHelper.AssertClose(result[0], new V3(0, 0, 0), 1e-9, "Zero vector unitize");
                AssertHelper.AssertClose(result[1], new V3(0, 1, 0), 1e-9, "Unitize (0,5,0)");

                record(new TestResultItem
                {
                    Name = "Unitize_ZeroVector_ReturnsZero",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Unitizing (0,0,0) returns (0,0,0) without NaN or crash."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Unitize_ZeroVector_ReturnsZero",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestAmplitudeUnitizeMultiplyNegative(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // REA finding: Native Amplitude unitizes then multiplies by magnitude (including negative)
                var vectors = new[] { new V3(3, 0, 0), new V3(0, 4, 0) };
                var magnitudes = new[] { -2.0, 10.0 };

                using var vBuf = session.UploadVectors(vectors);
                using var mBuf = session.UploadScalars(magnitudes);
                using var aBuf = session.Amplitude(vBuf, mBuf);

                var result = session.DownloadVectors(aBuf);

                // Vector 0: unitized is (1, 0, 0), multiplied by -2.0 is (-2, 0, 0)
                AssertHelper.AssertClose(result[0], new V3(-2, 0, 0), 1e-9, "Negative amplitude");

                // Vector 1: unitized is (0, 1, 0), multiplied by 10.0 is (0, 10, 0)
                AssertHelper.AssertClose(result[1], new V3(0, 10, 0), 1e-9, "Positive amplitude");

                record(new TestResultItem
                {
                    Name = "Amplitude_UnitizeMultiply_NegativeMagnitudeReverses",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "REA confirmed: Amplitude unitizes then multiplies by magnitude; negative magnitude reverses direction."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Amplitude_UnitizeMultiply_NegativeMagnitudeReverses",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestZeroVectorAmplitude(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var vectors = new[] { new V3(0, 0, 0) };
                var magnitudes = new[] { 100.0 };

                using var vBuf = session.UploadVectors(vectors);
                using var mBuf = session.UploadScalars(magnitudes);
                using var aBuf = session.Amplitude(vBuf, mBuf);

                var result = session.DownloadVectors(aBuf);
                AssertHelper.AssertClose(result[0], new V3(0, 0, 0), 1e-9, "Zero vector amplitude");

                record(new TestResultItem
                {
                    Name = "Amplitude_ZeroVector_ReturnsZero",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Amplitude_ZeroVector_ReturnsZero",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestVectorBetween(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var ptsA = new[] { new V3(1, 2, 3), new V3(10, 20, 30) };
                var ptsB = new[] { new V3(4, 6, 8), new V3(5, 10, 15) };

                using var aBuf = session.UploadPoints(ptsA);
                using var bBuf = session.UploadPoints(ptsB);
                using var vBuf = session.VectorBetween(aBuf, bBuf); // b - a

                var result = session.DownloadVectors(vBuf);

                AssertHelper.AssertClose(result[0], new V3(3, 4, 5), 1e-9, "VectorBetween 0");
                AssertHelper.AssertClose(result[1], new V3(-5, -10, -15), 1e-9, "VectorBetween 1");

                record(new TestResultItem
                {
                    Name = "VectorBetween_SubtractPoints",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "VectorBetween_SubtractPoints",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestMove(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var pts = new[] { new V3(10, 20, 30) };
                var vec = new[] { new V3(1, -2, 3) };

                using var pBuf = session.UploadPoints(pts);
                using var vBuf = session.UploadVectors(vec);
                using var mBuf = session.Move(pBuf, vBuf);

                var res = session.DownloadPoints(mBuf);
                AssertHelper.AssertClose(res[0], new V3(11, 18, 33), 1e-9, "Move point");

                record(new TestResultItem
                {
                    Name = "Move_PointPlusVector",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Move_PointPlusVector",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestScalarMathAllOperations(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var a = new[] { 10.0 };
                var b = new[] { 4.0 };

                using var bufA = session.UploadScalars(a);
                using var bufB = session.UploadScalars(b);

                using var add = session.ScalarMath(bufA, bufB, ScalarOperation.Add);
                using var sub = session.ScalarMath(bufA, bufB, ScalarOperation.Subtract);
                using var mul = session.ScalarMath(bufA, bufB, ScalarOperation.Multiply);
                using var div = session.ScalarMath(bufA, bufB, ScalarOperation.Divide);

                AssertHelper.AssertClose(session.DownloadScalars(add)[0], 14.0, 1e-9, "Add");
                AssertHelper.AssertClose(session.DownloadScalars(sub)[0], 6.0, 1e-9, "Subtract");
                AssertHelper.AssertClose(session.DownloadScalars(mul)[0], 40.0, 1e-9, "Multiply");
                AssertHelper.AssertClose(session.DownloadScalars(div)[0], 2.5, 1e-9, "Divide");

                record(new TestResultItem
                {
                    Name = "ScalarMath_Arithmetic_AllOperations",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ScalarMath_Arithmetic_AllOperations",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestScalarMathDivideByZeroIeee(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var a = new[] { 1.0, -1.0, 0.0 };
                var b = new[] { 0.0, 0.0, 0.0 };

                using var bufA = session.UploadScalars(a);
                using var bufB = session.UploadScalars(b);
                using var div = session.ScalarMath(bufA, bufB, ScalarOperation.Divide);

                var res = session.DownloadScalars(div);

                AssertHelper.AssertClose(res[0], double.PositiveInfinity, 0.0, "Divide positive by zero");
                AssertHelper.AssertClose(res[1], double.NegativeInfinity, 0.0, "Divide negative by zero");
                AssertHelper.AssertClose(res[2], double.NaN, 0.0, "Divide zero by zero");

                record(new TestResultItem
                {
                    Name = "ScalarMath_DivideByZero_IEEE754",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Divide by zero produces IEEE 754 nonfinite outputs (+Inf, -Inf, NaN) without aborting."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "ScalarMath_DivideByZero_IEEE754",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestSubnormalAndGiantVectorsNormalization(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Regression required by Task #93 and verified in artifacts/validation/native-numeric.json:
                // 1. Subnormal Unitize: (1e-320, 1e-320, 0) and (1e-308, 1e-308, 0)
                //    Native Rhino Unitize produces [0.7071067811865475, 0.7071067811865475, 0.0] (sqrt(0.5)).
                // 2. Giant Vector Unitize: (1e308, 1e308, 1e308)
                //    Native Rhino Unitize produces [0.5773502691896258, 0.5773502691896258, 0.5773502691896258] (1/sqrt(3)).
                // 3. Giant Vector 1.7e308: components scale finitely without overflow.
                // 4. Amplitude with subnormal/tiny vector: (1e-320, 1e-320, 0) with magnitude 2.0
                //    Produces unitized * 2.0 = [1.4142135623730951, 1.4142135623730951, 0.0].

                double sqrtHalf = Math.Sqrt(0.5); // 0.7071067811865475
                double invSqrt3 = 1.0 / Math.Sqrt(3.0); // 0.5773502691896258

                var testVectors = new[]
                {
                    new V3(1e-320, 1e-320, 0.0),    // subnormal
                    new V3(1e-308, 1e-308, 0.0),    // tiny normal
                    new V3(1e308, 1e308, 1e308),    // giant
                    new V3(1.7e308, 0.0, 0.0)       // near-max double
                };

                using var vBuf = session.UploadVectors(testVectors);
                using var uBuf = session.Unitize(vBuf);

                var unitized = session.DownloadVectors(uBuf);
                if (unitized.Length != 4)
                    throw new Exception($"Expected 4 unitized vectors, got {unitized.Length}");

                // Subnormal vector 1e-320
                AssertHelper.AssertFinite(unitized[0], "Unitized subnormal 1e-320");
                AssertHelper.AssertClose(unitized[0].X, sqrtHalf, 1e-12, "Subnormal 1e-320 X");
                AssertHelper.AssertClose(unitized[0].Y, sqrtHalf, 1e-12, "Subnormal 1e-320 Y");
                AssertHelper.AssertClose(unitized[0].Z, 0.0, 1e-12, "Subnormal 1e-320 Z");

                // Tiny vector 1e-308
                AssertHelper.AssertFinite(unitized[1], "Unitized tiny 1e-308");
                AssertHelper.AssertClose(unitized[1].X, sqrtHalf, 1e-12, "Tiny 1e-308 X");
                AssertHelper.AssertClose(unitized[1].Y, sqrtHalf, 1e-12, "Tiny 1e-308 Y");
                AssertHelper.AssertClose(unitized[1].Z, 0.0, 1e-12, "Tiny 1e-308 Z");

                // Giant vector 1e308
                AssertHelper.AssertFinite(unitized[2], "Unitized giant 1e308");
                AssertHelper.AssertClose(unitized[2].X, invSqrt3, 1e-12, "Giant 1e308 X");
                AssertHelper.AssertClose(unitized[2].Y, invSqrt3, 1e-12, "Giant 1e308 Y");
                AssertHelper.AssertClose(unitized[2].Z, invSqrt3, 1e-12, "Giant 1e308 Z");

                // Giant vector 1.7e308
                AssertHelper.AssertFinite(unitized[3], "Unitized giant 1.7e308");
                AssertHelper.AssertClose(unitized[3].X, 1.0, 1e-12, "Giant 1.7e308 X");
                AssertHelper.AssertClose(unitized[3].Y, 0.0, 1e-12, "Giant 1.7e308 Y");
                AssertHelper.AssertClose(unitized[3].Z, 0.0, 1e-12, "Giant 1.7e308 Z");

                // Amplitude with subnormal vector and magnitude 2.0
                var tinyVec = new[] { new V3(1e-320, 1e-320, 0.0) };
                var mag2 = new[] { 2.0 };
                using var tvBuf = session.UploadVectors(tinyVec);
                using var tmBuf = session.UploadScalars(mag2);
                using var ampBuf = session.Amplitude(tvBuf, tmBuf);

                var ampRes = session.DownloadVectors(ampBuf);
                if (ampRes.Length != 1)
                    throw new Exception("Amplitude output count mismatch.");

                double expectedAmpCoord = 2.0 * sqrtHalf; // 1.4142135623730951
                AssertHelper.AssertFinite(ampRes[0], "Amplitude subnormal result");
                AssertHelper.AssertClose(ampRes[0].X, expectedAmpCoord, 1e-12, "Amplitude subnormal X");
                AssertHelper.AssertClose(ampRes[0].Y, expectedAmpCoord, 1e-12, "Amplitude subnormal Y");
                AssertHelper.AssertClose(ampRes[0].Z, 0.0, 1e-12, "Amplitude subnormal Z");

                record(new TestResultItem
                {
                    Name = "Unitize_Subnormal_And_Giant_Vectors_Normalization",
                    Category = "MathVector",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Matches Rhino 8.35 recorded native-numeric.json: 1e-320 and 1e-308 normalize to [sqrt(0.5), sqrt(0.5), 0], 1e308 normalizes to [1/sqrt(3)], and amplitude with magnitude 2.0 scales finitely."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Unitize_Subnormal_And_Giant_Vectors_Normalization",
                    Category = "MathVector",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
