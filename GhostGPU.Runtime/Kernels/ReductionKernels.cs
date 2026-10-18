using System;
using GhostGPU.Core;
using ILGPU;
using ILGPU.Algorithms;

namespace GhostGPU.Runtime.Kernels
{
    internal static class ReductionKernels
    {
        /// <summary>
        /// Validation oracle: serial single-thread bounds kernel from v1. Not used by Bounds;
        /// the parallel path below must match it (NaN skipped, empty/all-NaN -> NaN, +Inf min / -Inf max -> NaN).
        /// </summary>
        public static void SerialBoundsKernel(
            Index1D index,
            ArrayView<double> input,
            int inputLength,
            ArrayView<double> output)
        {
            double min = double.PositiveInfinity;
            double max = double.NegativeInfinity;
            for (int i = 0; i < inputLength; i++)
            {
                double val = input[i];
                if (!double.IsNaN(val))
                {
                    if (val < min) min = val;
                    if (val > max) max = val;
                }
            }
            if (double.IsPositiveInfinity(min)) min = double.NaN;
            if (double.IsNegativeInfinity(max)) max = double.NaN;
            output[0] = min;
            output[1] = max;
        }

        /// <summary>
        /// Pass 1: grid-stride min/max over input[i * elemStride + elemOffset] for i in [0, count),
        /// skipping NaN. elemStride/offset let the same kernel reduce one component of a V3 view
        /// cast to doubles (stride 3).
        /// </summary>
        public static void BoundsPass1Kernel(
            Index1D index,
            ArrayView<double> input,
            int count,
            int elemStride,
            int elemOffset,
            int threads,
            ArrayView<double> partialMins,
            ArrayView<double> partialMaxs)
        {
            int t = index;
            double localMin = double.PositiveInfinity;
            double localMax = double.NegativeInfinity;

            for (int i = t; i < count; i += threads)
            {
                double val = input[(long)i * elemStride + elemOffset];
                if (!double.IsNaN(val))
                {
                    if (val < localMin) localMin = val;
                    if (val > localMax) localMax = val;
                }
            }

            partialMins[t] = localMin;
            partialMaxs[t] = localMax;
        }

        /// <summary>
        /// Pass 2..n: grid-stride reduction of partial (min, max) pairs. With finalize != 0 (single
        /// thread, last level) applies the v1 rule: unchanged +Inf min / -Inf max become NaN.
        /// </summary>
        public static void BoundsReducePartialsKernel(
            Index1D index,
            ArrayView<double> inMins,
            ArrayView<double> inMaxs,
            int count,
            int threads,
            ArrayView<double> outMins,
            ArrayView<double> outMaxs,
            int finalize)
        {
            int t = index;
            double min = double.PositiveInfinity;
            double max = double.NegativeInfinity;

            for (int i = t; i < count; i += threads)
            {
                double pMin = inMins[i];
                double pMax = inMaxs[i];
                if (pMin < min) min = pMin;
                if (pMax > max) max = pMax;
            }

            if (finalize != 0)
            {
                if (double.IsPositiveInfinity(min)) min = double.NaN;
                if (double.IsNegativeInfinity(max)) max = double.NaN;
            }

            outMins[t] = min;
            outMaxs[t] = max;
        }
    }
}
