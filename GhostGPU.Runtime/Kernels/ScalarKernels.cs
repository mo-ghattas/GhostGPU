using System;
using GhostGPU.Core;
using ILGPU;
using ILGPU.Algorithms;

namespace GhostGPU.Runtime.Kernels
{
    internal static class ScalarKernels
    {
        // Shared pure element functions
        public static double ScalarMathElement(double va, double vb, int op)
        {
            if (op == 0) return va + vb;
            if (op == 1) return va - vb;
            if (op == 2) return va * vb;
            if (op == 3) return va / vb;
            return 0.0;
        }

        public static double RemapElement(double x, double sMin, double sMax, double targetMin, double targetMax, bool clamp)
        {
            double tMid = (targetMin + targetMax) * 0.5;
            double tMin = targetMin < targetMax ? targetMin : targetMax;
            double tMax = targetMin > targetMax ? targetMin : targetMax;

            if (sMin == sMax || targetMin == targetMax)
            {
                return tMid;
            }

            double t = (x - sMin) / (sMax - sMin);
            double res = targetMin + Fp.Mul(t, targetMax - targetMin);
            if (clamp)
            {
                if (res < tMin) res = tMin;
                else if (res > tMax) res = tMax;
            }
            return res;
        }

        public static double MapElement(double x, int function)
        {
            if (function == 0) // Linear
            {
                return x;
            }
            if (function == 1) // SmoothStep
            {
                double t = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
                return t * t * (3.0 - Fp.Mul(2.0, t));
            }
            if (function == 2) // Sine
            {
                return Sin(x);
            }
            return 0.0;
        }

        /// <summary>
        /// Shared sine (fdlibm algorithm: 3-step Cody-Waite reduction by pi/2 + __kernel_sin/__kernel_cos
        /// polynomials, &lt; 1 ulp). XMath.Sin is Math.Sin on the host but a different ~10-ulp routine on
        /// CUDA, so Map(Sine) could not be bit-identical across paths. Every product feeding an add goes
        /// through Fp.Mul (no FMA contraction on CUDA).
        /// Limitation: medium-argument reduction only; beyond |x| ~ 1.6e6 rad accuracy degrades gradually
        /// (still deterministic and host/device identical). Add Payne-Hanek if huge angles ever matter.
        /// </summary>
        public static double Sin(double x)
        {
            const double InvPio2 = 6.36619772367581382433e-01;
            const double P1 = 1.57079632673412561417e+00, P2 = 6.07710050630396597660e-11, P2t = 2.02226624879595063154e-21;
            const double P3 = 2.02226624871116645580e-21, P3t = 8.47842766036889956997e-32;

            double fn = XMath.Floor(Fp.Mul(x, InvPio2) + 0.5);
            // Fp.Sub keeps the compensation terms ((t - r) - w) etc. from being folded to 0 on the device.
            double r = x - Fp.Mul(fn, P1);
            double t = r;
            double w = Fp.Mul(fn, P2);
            r = Fp.Sub(t, w);
            w = Fp.Mul(fn, P2t) - ((t - r) - w);
            t = r;
            w = Fp.Mul(fn, P3);
            r = Fp.Sub(t, w);
            w = Fp.Mul(fn, P3t) - ((t - r) - w);
            double y0 = Fp.Sub(r, w);
            double y1 = (r - y0) - w;

            double quadrant = fn - Fp.Mul(4.0, XMath.Floor(Fp.Mul(fn, 0.25)));
            if (quadrant == 0.0) return KernelSin(y0, y1);
            if (quadrant == 1.0) return KernelCos(y0, y1);
            if (quadrant == 2.0) return -KernelSin(y0, y1);
            return -KernelCos(y0, y1); // quadrant 3, or NaN propagating from NaN/Inf input
        }

        private static double KernelSin(double x, double y)
        {
            const double S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03, S3 = -1.98412698298579493134e-04;
            const double S4 = 2.75573137070700676789e-06, S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
            double z = Fp.Mul(x, x);
            double v = Fp.Mul(z, x);
            double r = S2 + Fp.Mul(z, S3 + Fp.Mul(z, S4 + Fp.Mul(z, S5 + Fp.Mul(z, S6))));
            return x - ((Fp.Mul(z, Fp.Mul(0.5, y) - Fp.Mul(v, r)) - y) - Fp.Mul(v, S1));
        }

        private static double KernelCos(double x, double y)
        {
            const double C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03, C3 = 2.48015872894767294178e-05;
            const double C4 = -2.75573143513906633035e-07, C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
            double z = Fp.Mul(x, x);
            double r = Fp.Mul(z, C1 + Fp.Mul(z, C2 + Fp.Mul(z, C3 + Fp.Mul(z, C4 + Fp.Mul(z, C5 + Fp.Mul(z, C6))))));
            double hz = Fp.Mul(0.5, z);
            double w = Fp.Sub(1.0, hz);
            return w + (((1.0 - w) - hz) + (Fp.Mul(z, r) - Fp.Mul(x, y)));
        }

        // F64 Kernels
        public static void ScalarMathKernel(
            Index1D index,
            ArrayView<double> a,
            ArrayView<double> b,
            int aCount,
            int bCount,
            int op,
            ArrayView<double> output)
        {
            int i = index;
            int idxA = i < aCount ? i : aCount - 1;
            int idxB = i < bCount ? i : bCount - 1;
            output[i] = ScalarMathElement(a[idxA], b[idxB], op);
        }

        public static void RemapKernel(
            Index1D index,
            ArrayView<double> scalars,
            ArrayView<double> sourceBounds,
            double targetMin,
            double targetMax,
            int clampFlag,
            ArrayView<double> output)
        {
            int i = index;
            double sMin = sourceBounds[0];
            double sMax = sourceBounds[1];
            output[i] = RemapElement(scalars[i], sMin, sMax, targetMin, targetMax, clampFlag != 0);
        }

        public static void MapKernel(
            Index1D index,
            ArrayView<double> scalars,
            int function,
            ArrayView<double> output)
        {
            int i = index;
            output[i] = MapElement(scalars[i], function);
        }

        // F32 storage: widen on load / narrow on store around the F64 kernels above.
        public static void WidenF32Kernel(Index1D index, ArrayView<float> input, ArrayView<double> output)
        {
            output[index] = input[index];
        }

        public static void NarrowF64Kernel(Index1D index, ArrayView<double> input, ArrayView<float> output)
        {
            output[index] = (float)input[index];
        }
    }
}
