using System;
using GhostGPU.Core;

namespace GhostGPU.Validation.Reference
{
    /// <summary>
    /// Explicit naive, unvectorized, single-threaded CPU reference implementation.
    /// Used for independent correctness verification and baseline benchmarking.
    /// Not intended as an optimized Kd-tree or SIMD baseline.
    /// </summary>
    public static class NaiveCpuReference
    {
        public const string Description = "Explicit naive unvectorized O(N) or O(N*M) CPU reference loop; no KD-tree, spatial index, or SIMD optimization.";

        /// <summary>
        /// Scaled Euclidean distance to avoid overflow at large magnitudes (up to 1e155),
        /// matching Rhino Vector3d.GetLengthHelper / Point3d.DistanceTo behavior.
        /// </summary>
        public static double ScaledDistance(V3 a, V3 b)
        {
            double dx = a.X - b.X;
            double dy = a.Y - b.Y;
            double dz = a.Z - b.Z;

            double max = Math.Max(Math.Abs(dx), Math.Max(Math.Abs(dy), Math.Abs(dz)));
            if (max == 0.0 || double.IsNaN(max))
                return 0.0;
            if (double.IsInfinity(max))
                return double.PositiveInfinity;

            double sx = dx / max;
            double sy = dy / max;
            double sz = dz / max;
            return max * Math.Sqrt(sx * sx + sy * sy + sz * sz);
        }

        public static double ScaledLength(V3 v)
        {
            double max = Math.Max(Math.Abs(v.X), Math.Max(Math.Abs(v.Y), Math.Abs(v.Z)));
            if (max == 0.0 || double.IsNaN(max))
                return 0.0;
            if (double.IsInfinity(max))
                return double.PositiveInfinity;

            double sx = v.X / max;
            double sy = v.Y / max;
            double sz = v.Z / max;
            return max * Math.Sqrt(sx * sx + sy * sy + sz * sz);
        }

        public static (V3[] Points, int[] Indices, double[] Distances) ClosestPoints(V3[] queries, V3[] targets)
        {
            if (queries.Length > 0 && targets.Length == 0)
                throw new ArgumentException("Nonempty queries with empty targets is an error per API contract.");

            if (queries.Length == 0)
                return (Array.Empty<V3>(), Array.Empty<int>(), Array.Empty<double>());

            var resultPoints = new V3[queries.Length];
            var resultIndices = new int[queries.Length];
            var resultDistances = new double[queries.Length];

            for (int i = 0; i < queries.Length; i++)
            {
                V3 q = queries[i];
                double bestDist = double.PositiveInfinity;
                int bestIdx = 0;
                V3 bestPt = targets[0];

                for (int j = 0; j < targets.Length; j++)
                {
                    double d = ScaledDistance(q, targets[j]);
                    // Strict < comparison: earlier/lower target index wins exact ties
                    if (d < bestDist)
                    {
                        bestDist = d;
                        bestIdx = j;
                        bestPt = targets[j];
                    }
                }

                resultPoints[i] = bestPt;
                resultIndices[i] = bestIdx;
                resultDistances[i] = bestDist;
            }

            return (resultPoints, resultIndices, resultDistances);
        }

        public static double[] Distance(V3[] a, V3[] b)
        {
            if (a.Length == 0 || b.Length == 0)
                return Array.Empty<double>();

            int count = Math.Max(a.Length, b.Length);
            var result = new double[count];

            for (int i = 0; i < count; i++)
            {
                V3 pa = a[i < a.Length ? i : a.Length - 1];
                V3 pb = b[i < b.Length ? i : b.Length - 1];
                result[i] = ScaledDistance(pa, pb);
            }

            return result;
        }

        public static V3[] VectorBetween(V3[] a, V3[] b)
        {
            if (a.Length == 0 || b.Length == 0)
                return Array.Empty<V3>();

            int count = Math.Max(a.Length, b.Length);
            var result = new V3[count];

            for (int i = 0; i < count; i++)
            {
                V3 pa = a[i < a.Length ? i : a.Length - 1];
                V3 pb = b[i < b.Length ? i : b.Length - 1];
                result[i] = pb - pa;
            }

            return result;
        }

        public static V3 ScaledUnitVector(V3 v)
        {
            double max = Math.Max(Math.Abs(v.X), Math.Max(Math.Abs(v.Y), Math.Abs(v.Z)));
            if (max == 0.0 || double.IsNaN(max) || double.IsInfinity(max))
                return new V3(0, 0, 0);

            double sx = v.X / max;
            double sy = v.Y / max;
            double sz = v.Z / max;
            double sLen = Math.Sqrt(sx * sx + sy * sy + sz * sz);
            if (sLen == 0.0 || double.IsNaN(sLen) || double.IsInfinity(sLen))
                return new V3(0, 0, 0);

            return new V3(sx / sLen, sy / sLen, sz / sLen);
        }

        public static V3[] Unitize(V3[] vectors)
        {
            var result = new V3[vectors.Length];
            for (int i = 0; i < vectors.Length; i++)
            {
                result[i] = ScaledUnitVector(vectors[i]);
            }
            return result;
        }

        public static V3[] Amplitude(V3[] vectors, double[] magnitudes)
        {
            if (vectors.Length == 0 || magnitudes.Length == 0)
                return Array.Empty<V3>();

            int count = Math.Max(vectors.Length, magnitudes.Length);
            var result = new V3[count];

            for (int i = 0; i < count; i++)
            {
                V3 v = vectors[i < vectors.Length ? i : vectors.Length - 1];
                double mag = magnitudes[i < magnitudes.Length ? i : magnitudes.Length - 1];

                V3 unit = ScaledUnitVector(v);
                result[i] = unit * mag;
            }

            return result;
        }

        public static V3[] Move(V3[] points, V3[] vectors)
        {
            if (points.Length == 0 || vectors.Length == 0)
                return Array.Empty<V3>();

            int count = Math.Max(points.Length, vectors.Length);
            var result = new V3[count];

            for (int i = 0; i < count; i++)
            {
                V3 p = points[i < points.Length ? i : points.Length - 1];
                V3 v = vectors[i < vectors.Length ? i : vectors.Length - 1];
                result[i] = p + v;
            }

            return result;
        }

        public static double[] ScalarMath(double[] a, double[] b, ScalarOperation op)
        {
            if (a.Length == 0 || b.Length == 0)
                return Array.Empty<double>();

            int count = Math.Max(a.Length, b.Length);
            var result = new double[count];

            for (int i = 0; i < count; i++)
            {
                double va = a[i < a.Length ? i : a.Length - 1];
                double vb = b[i < b.Length ? i : b.Length - 1];

                result[i] = op switch
                {
                    ScalarOperation.Add => va + vb,
                    ScalarOperation.Subtract => va - vb,
                    ScalarOperation.Multiply => va * vb,
                    ScalarOperation.Divide => va / vb, // IEEE standard divide by zero (+/-Inf, NaN)
                    _ => throw new ArgumentOutOfRangeException(nameof(op))
                };
            }

            return result;
        }

        public static double[] Bounds(double[] scalars)
        {
            if (scalars.Length == 0)
                return new[] { double.NaN, double.NaN };

            double min = double.PositiveInfinity;
            double max = double.NegativeInfinity;

            for (int i = 0; i < scalars.Length; i++)
            {
                double v = scalars[i];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            return new[] { min, max };
        }

        public static double[] Remap(double[] scalars, double[] sourceBounds, double targetMin, double targetMax, bool clamp)
        {
            if (sourceBounds.Length != 2)
                throw new ArgumentException("Source bounds must have exactly 2 elements [min, max].");

            var result = new double[scalars.Length];
            double sMin = sourceBounds[0];
            double sMax = sourceBounds[1];

            // REA / ILSpy correction: collapsed source interval maps to target MIDPOINT
            bool isCollapsed = (sMin == sMax) || double.IsNaN(sMin) || double.IsNaN(sMax);

            double targetMid = (targetMin + targetMax) * 0.5;
            double cMin = Math.Min(targetMin, targetMax);
            double cMax = Math.Max(targetMin, targetMax);

            for (int i = 0; i < scalars.Length; i++)
            {
                if (isCollapsed)
                {
                    result[i] = targetMid;
                }
                else
                {
                    double t = (scalars[i] - sMin) / (sMax - sMin);
                    double val = targetMin + t * (targetMax - targetMin);

                    if (clamp)
                    {
                        val = Math.Clamp(val, cMin, cMax);
                    }

                    result[i] = val;
                }
            }

            return result;
        }

        public static double[] Map(double[] scalars, MapFunction function)
        {
            var result = new double[scalars.Length];
            for (int i = 0; i < scalars.Length; i++)
            {
                double x = scalars[i];
                result[i] = function switch
                {
                    MapFunction.Linear => x,
                    MapFunction.SmoothStep => SmoothStep(x),
                    MapFunction.Sine => Math.Sin(x),
                    _ => throw new ArgumentOutOfRangeException(nameof(function))
                };
            }
            return result;
        }

        private static double SmoothStep(double x)
        {
            double c = Math.Clamp(x, 0.0, 1.0);
            return c * c * (3.0 - 2.0 * c);
        }

        public static uint[] ColourRamp(double[] scalars, uint lowArgb, uint highArgb)
        {
            var result = new uint[scalars.Length];

            byte a0 = (byte)((lowArgb >> 24) & 0xFF);
            byte r0 = (byte)((lowArgb >> 16) & 0xFF);
            byte g0 = (byte)((lowArgb >> 8) & 0xFF);
            byte b0 = (byte)(lowArgb & 0xFF);

            byte a1 = (byte)((highArgb >> 24) & 0xFF);
            byte r1 = (byte)((highArgb >> 16) & 0xFF);
            byte g1 = (byte)((highArgb >> 8) & 0xFF);
            byte b1 = (byte)(highArgb & 0xFF);

            for (int i = 0; i < scalars.Length; i++)
            {
                double x = scalars[i];
                if (double.IsNaN(x) || double.IsInfinity(x))
                {
                    result[i] = lowArgb;
                    continue;
                }

                double t = Math.Clamp(x, 0.0, 1.0);

                // Linearly interpolate each byte, round using floor(val + 0.5)
                byte a = ClampToByte(Math.Floor(a0 + t * (a1 - a0) + 0.5));
                byte r = ClampToByte(Math.Floor(r0 + t * (r1 - r0) + 0.5));
                byte g = ClampToByte(Math.Floor(g0 + t * (g1 - g0) + 0.5));
                byte b = ClampToByte(Math.Floor(b0 + t * (b1 - b0) + 0.5));

                result[i] = ((uint)a << 24) | ((uint)r << 16) | ((uint)g << 8) | (uint)b;
            }

            return result;
        }

        private static byte ClampToByte(double val)
        {
            if (val <= 0.0) return 0;
            if (val >= 255.0) return 255;
            return (byte)val;
        }

        public static uint[] RepeatColours(uint[] colours, int count)
        {
            if (colours.Length == 0 && count > 0)
                throw new ArgumentException("Empty colours pattern with requested nonzero count is an error.");

            if (count <= 0)
                return Array.Empty<uint>();

            var result = new uint[count];
            for (int i = 0; i < count; i++)
            {
                // Native Mesh Colours REA cyclic pattern wrap
                result[i] = colours[i % colours.Length];
            }

            return result;
        }
    }
}
