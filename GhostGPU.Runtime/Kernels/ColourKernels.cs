using System;
using ILGPU;
using ILGPU.Algorithms;

namespace GhostGPU.Runtime.Kernels
{
    internal static class ColourKernels
    {
        public static uint ColourRampElement(double x, uint lowArgb, uint highArgb)
        {
            if (double.IsNaN(x) || double.IsInfinity(x))
            {
                return lowArgb;
            }

            double t = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
            int lowA = (int)((lowArgb >> 24) & 0xFF);
            int lowR = (int)((lowArgb >> 16) & 0xFF);
            int lowG = (int)((lowArgb >> 8) & 0xFF);
            int lowB = (int)(lowArgb & 0xFF);
            int highA = (int)((highArgb >> 24) & 0xFF);
            int highR = (int)((highArgb >> 16) & 0xFF);
            int highG = (int)((highArgb >> 8) & 0xFF);
            int highB = (int)(highArgb & 0xFF);

            double aVal = lowA + Fp.Mul(t, highA - lowA);
            double rVal = lowR + Fp.Mul(t, highR - lowR);
            double gVal = lowG + Fp.Mul(t, highG - lowG);
            double bVal = lowB + Fp.Mul(t, highB - lowB);

            int a = (int)XMath.Floor(aVal + 0.5);
            int r = (int)XMath.Floor(rVal + 0.5);
            int g = (int)XMath.Floor(gVal + 0.5);
            int b = (int)XMath.Floor(bVal + 0.5);

            if (a < 0) a = 0; else if (a > 255) a = 255;
            if (r < 0) r = 0; else if (r > 255) r = 255;
            if (g < 0) g = 0; else if (g > 255) g = 255;
            if (b < 0) b = 0; else if (b > 255) b = 255;

            return ((uint)a << 24) | ((uint)r << 16) | ((uint)g << 8) | (uint)b;
        }

        public static void ColourRampKernel(
            Index1D index,
            ArrayView<double> scalars,
            uint lowArgb,
            uint highArgb,
            ArrayView<uint> output)
        {
            int i = index;
            output[i] = ColourRampElement(scalars[i], lowArgb, highArgb);
        }

        public static void RepeatColoursKernel(
            Index1D index,
            ArrayView<uint> pattern,
            int patternLength,
            ArrayView<uint> output)
        {
            int i = index;
            int idx = i % patternLength;
            output[i] = pattern[idx];
        }
    }
}
