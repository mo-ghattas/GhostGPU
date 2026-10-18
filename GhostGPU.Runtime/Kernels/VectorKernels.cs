using System;
using GhostGPU.Core;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime.Cuda;

namespace GhostGPU.Runtime.Kernels
{
    /// <summary>
    /// Host/device bit-identity helper. The CUDA JIT contracts a plain PTX mul.f64 followed by add.f64 into
    /// fma.rn.f64 (one rounding instead of two), which made device results differ from the host by 1-2 ulp.
    /// mul.rn.f64 is never contracted, so every product that feeds an addition in a shared element function
    /// goes through Mul. On the host and the ILGPU CPU accelerator it is a plain multiply (.NET never contracts).
    /// </summary>
    internal static class Fp
    {
        public static double Mul(double a, double b)
        {
            if (CudaAsm.IsSupported)
            {
                CudaAsm.Emit("mul.rn.f64 %0, %1, %2;", out double r, a, b);
                return r;
            }
            return a * b;
        }

        /// <summary>
        /// Subtraction the ILGPU IR optimizer cannot see through. It folds c - (c - x) to x, which destroys
        /// error-compensation terms like ((1 - w) - hz) where w = 1 - hz. Use for the operand of such terms.
        /// </summary>
        public static double Sub(double a, double b)
        {
            if (CudaAsm.IsSupported)
            {
                CudaAsm.Emit("sub.rn.f64 %0, %1, %2;", out double r, a, b);
                return r;
            }
            return a - b;
        }
    }

    internal static class VectorKernels
    {
        private const double MinNormal = 2.2250738585072014e-308;

        public static bool IsValidDouble(double d) => !double.IsNaN(d) && !double.IsInfinity(d);

        public static double GetLengthHelper(double dx, double dy, double dz)
        {
            if (!IsValidDouble(dx) || !IsValidDouble(dy) || !IsValidDouble(dz))
                return 0.0;

            double num = dx < 0.0 ? -dx : dx;
            double num2 = dy < 0.0 ? -dy : dy;
            double num3 = dz < 0.0 ? -dz : dz;
            if (num2 >= num && num2 >= num3)
            {
                double num4 = num;
                num = num2;
                num2 = num4;
            }
            else if (num3 >= num && num3 >= num2)
            {
                double num4 = num;
                num = num3;
                num3 = num4;
            }
            if (num > 2.2250738585072014e-308)
            {
                double num4 = 1.0 / num;
                num2 *= num4;
                num3 *= num4;
                return num * XMath.Sqrt(1.0 + Fp.Mul(num2, num2) + Fp.Mul(num3, num3));
            }
            if (num > 0.0 && IsValidDouble(num))
            {
                return num;
            }
            return 0.0;
        }

        public static double RobustDistance(V3 a, V3 b)
        {
            return GetLengthHelper(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        public static V3 NormalizeVector(V3 v)
        {
            if (!IsValidDouble(v.X) || !IsValidDouble(v.Y) || !IsValidDouble(v.Z))
                return new V3(0.0, 0.0, 0.0);

            double ax = v.X < 0.0 ? -v.X : v.X;
            double ay = v.Y < 0.0 ? -v.Y : v.Y;
            double az = v.Z < 0.0 ? -v.Z : v.Z;

            double m = ax;
            if (ay > m) m = ay;
            if (az > m) m = az;

            if (m == 0.0)
                return new V3(0.0, 0.0, 0.0);

            double sx = v.X / m;
            double sy = v.Y / m;
            double sz = v.Z / m;

            double scaledLen = XMath.Sqrt(Fp.Mul(sx, sx) + Fp.Mul(sy, sy) + Fp.Mul(sz, sz));
            if (scaledLen > 0.0)
            {
                return new V3(sx / scaledLen, sy / scaledLen, sz / scaledLen);
            }

            return new V3(0.0, 0.0, 0.0);
        }

        // Shared pure element functions for host path and device kernels
        public static double DistanceElement(V3 a, V3 b) => RobustDistance(a, b);
        public static V3 VectorBetweenElement(V3 a, V3 b) => new V3(b.X - a.X, b.Y - a.Y, b.Z - a.Z);
        public static V3 UnitizeElement(V3 v) => NormalizeVector(v);
        public static V3 AmplitudeElement(V3 v, double mag)
        {
            V3 u = NormalizeVector(v);
            return new V3(u.X * mag, u.Y * mag, u.Z * mag);
        }
        public static V3 MoveElement(V3 p, V3 v) => new V3(p.X + v.X, p.Y + v.Y, p.Z + v.Z);

        // Device kernels operate on V3D views; compact V3F inputs are widened into an arena temp
        // first (WidenV3FKernel) and V3F outputs narrowed after (NarrowV3Kernel), so host and device\r
        // paths run the exact same element functions and conversions.
        public static void DistanceKernel(
            Index1D index,
            ArrayView<V3> a,
            ArrayView<V3> b,
            int aCount,
            int bCount,
            ArrayView<double> output)
        {
            int i = index;
            int idxA = i < aCount ? i : aCount - 1;
            int idxB = i < bCount ? i : bCount - 1;
            output[i] = DistanceElement(a[idxA], b[idxB]);
        }

        public static void VectorBetweenKernel(
            Index1D index,
            ArrayView<V3> a,
            ArrayView<V3> b,
            int aCount,
            int bCount,
            ArrayView<V3> output)
        {
            int i = index;
            int idxA = i < aCount ? i : aCount - 1;
            int idxB = i < bCount ? i : bCount - 1;
            output[i] = VectorBetweenElement(a[idxA], b[idxB]);
        }

        public static void UnitizeKernel(
            Index1D index,
            ArrayView<V3> vectors,
            ArrayView<V3> output)
        {
            int i = index;
            output[i] = UnitizeElement(vectors[i]);
        }

        public static void AmplitudeKernel(
            Index1D index,
            ArrayView<V3> vectors,
            ArrayView<double> magnitudes,
            int vCount,
            int mCount,
            ArrayView<V3> output)
        {
            int i = index;
            int idxV = i < vCount ? i : vCount - 1;
            int idxM = i < mCount ? i : mCount - 1;
            output[i] = AmplitudeElement(vectors[idxV], magnitudes[idxM]);
        }

        public static void MoveKernel(
            Index1D index,
            ArrayView<V3> points,
            ArrayView<V3> vectors,
            int pCount,
            int vCount,
            ArrayView<V3> output)
        {
            int i = index;
            int idxP = i < pCount ? i : pCount - 1;
            int idxV = i < vCount ? i : vCount - 1;
            output[i] = MoveElement(points[idxP], vectors[idxV]);
        }

        public static void WidenV3FKernel(
            Index1D index,
            ArrayView<V3F> input,
            V3 origin,
            ArrayView<V3> output)
        {
            int i = index;
            output[i] = input[i].ToV3(origin);
        }

        public static void NarrowV3Kernel(
            Index1D index,
            ArrayView<V3> input,
            ArrayView<V3F> output,
            V3 origin)
        {
            int i = index;
            output[i] = V3F.FromV3(input[i], origin);
        }
    }
}
