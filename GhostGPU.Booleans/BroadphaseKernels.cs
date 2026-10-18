using System;
using GhostGPU.Core;
using ILGPU;

namespace GhostGPU.Booleans
{
    public static class BroadphaseKernels
    {
        public static void CountOverlapsKernel(
            Index1D index,
            ArrayView<V3> minA,
            ArrayView<V3> maxA,
            ArrayView<V3> minB,
            ArrayView<V3> maxB,
            int countB,
            double tol,
            ArrayView<int> countsOut)
        {
            V3 miA = minA[index];
            V3 maA = maxA[index];
            int c = 0;

            for (int j = 0; j < countB; j++)
            {
                V3 miB = minB[j];
                V3 maB = maxB[j];

                if (miA.X <= maB.X + tol && maA.X >= miB.X - tol &&
                    miA.Y <= maB.Y + tol && maA.Y >= miB.Y - tol &&
                    miA.Z <= maB.Z + tol && maA.Z >= miB.Z - tol)
                {
                    c++;
                }
            }
            countsOut[index] = c;
        }

        public static void WriteOverlapsKernelI32(
            Index1D index,
            ArrayView<V3> minA,
            ArrayView<V3> maxA,
            ArrayView<V3> minB,
            ArrayView<V3> maxB,
            int countB,
            double tol,
            ArrayView<int> offsets,
            ArrayView<int> pairsOut)
        {
            V3 miA = minA[index];
            V3 maA = maxA[index];
            int outPtr = offsets[index] * 2;

            for (int j = 0; j < countB; j++)
            {
                V3 miB = minB[j];
                V3 maB = maxB[j];

                if (miA.X <= maB.X + tol && maA.X >= miB.X - tol &&
                    miA.Y <= maB.Y + tol && maA.Y >= miB.Y - tol &&
                    miA.Z <= maB.Z + tol && maA.Z >= miB.Z - tol)
                {
                    pairsOut[outPtr] = index;
                    pairsOut[outPtr + 1] = j;
                    outPtr += 2;
                }
            }
        }

        public static void WriteOverlapsKernelU16(
            Index1D index,
            ArrayView<V3> minA,
            ArrayView<V3> maxA,
            ArrayView<V3> minB,
            ArrayView<V3> maxB,
            int countB,
            double tol,
            ArrayView<ushort> offsets,
            ArrayView<int> pairsOut)
        {
            V3 miA = minA[index];
            V3 maA = maxA[index];
            int outPtr = (int)offsets[index] * 2;

            for (int j = 0; j < countB; j++)
            {
                V3 miB = minB[j];
                V3 maB = maxB[j];

                if (miA.X <= maB.X + tol && maA.X >= miB.X - tol &&
                    miA.Y <= maB.Y + tol && maA.Y >= miB.Y - tol &&
                    miA.Z <= maB.Z + tol && maA.Z >= miB.Z - tol)
                {
                    pairsOut[outPtr] = index;
                    pairsOut[outPtr + 1] = j;
                    outPtr += 2;
                }
            }
        }
    }
}
