using System;
using System.Runtime.CompilerServices;
using GhostGPU.Core;
using ILGPU;

namespace GhostGPU.Kernel
{
    public static class GpuEvalKernels
    {
        public const int MaxDegreePlus1 = 32;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int FindSpan(ArrayView<double> knots, int degree, int cvCount, double t)
        {
            if (t >= knots[cvCount]) return cvCount - 1;
            if (t <= knots[degree]) return degree;

            int low = degree;
            int high = cvCount;
            while (high - low > 1)
            {
                int mid = (low + high) / 2;
                if (t < knots[mid])
                    high = mid;
                else
                    low = mid;
            }
            return low;
        }

        public static void EvaluateSurfacePointsKernel(
            Index1D index,
            ArrayView<V3> uvPoints,
            ArrayView<double> knotsU,
            ArrayView<double> knotsV,
            ArrayView<double> cvs,
            int degreeU,
            int degreeV,
            int cvCountU,
            int cvCountV,
            int isRational,
            ArrayView<V3> outPoints)
        {
            V3 uv = uvPoints[index];
            double u = uv.X;
            double v = uv.Y;

            int spanU = FindSpan(knotsU, degreeU, cvCountU, u);
            int spanV = FindSpan(knotsV, degreeV, cvCountV, v);

            var leftU = LocalMemory.Allocate<double>(MaxDegreePlus1);
            var rightU = LocalMemory.Allocate<double>(MaxDegreePlus1);
            var nu = LocalMemory.Allocate<double>(MaxDegreePlus1);

            var leftV = LocalMemory.Allocate<double>(MaxDegreePlus1);
            var rightV = LocalMemory.Allocate<double>(MaxDegreePlus1);
            var nv = LocalMemory.Allocate<double>(MaxDegreePlus1);

            nu[0] = 1.0;
            for (int j = 1; j <= degreeU; j++)
            {
                leftU[j] = u - knotsU[spanU + 1 - j];
                rightU[j] = knotsU[spanU + j] - u;
                double saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    double temp = nu[r] / (rightU[r + 1] + leftU[j - r]);
                    nu[r] = saved + rightU[r + 1] * temp;
                    saved = leftU[j - r] * temp;
                }
                nu[j] = saved;
            }

            nv[0] = 1.0;
            for (int j = 1; j <= degreeV; j++)
            {
                leftV[j] = v - knotsV[spanV + 1 - j];
                rightV[j] = knotsV[spanV + j] - v;
                double saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    double temp = nv[r] / (rightV[r + 1] + leftV[j - r]);
                    nv[r] = saved + rightV[r + 1] * temp;
                    saved = leftV[j - r] * temp;
                }
                nv[j] = saved;
            }

            int startU = spanU - degreeU;
            int startV = spanV - degreeV;
            double sumX = 0.0, sumY = 0.0, sumZ = 0.0, sumW = 0.0;

            for (int i = 0; i <= degreeU; i++)
            {
                double nui = nu[i];
                int rowIdx = (startU + i) * cvCountV;
                for (int j = 0; j <= degreeV; j++)
                {
                    double weight = nui * nv[j];
                    int cvIdx = (rowIdx + (startV + j)) * 4;
                    sumX += weight * cvs[cvIdx];
                    sumY += weight * cvs[cvIdx + 1];
                    sumZ += weight * cvs[cvIdx + 2];
                    sumW += weight * cvs[cvIdx + 3];
                }
            }

            if (isRational != 0 && sumW != 0.0)
            {
                double invW = 1.0 / sumW;
                sumX *= invW;
                sumY *= invW;
                sumZ *= invW;
            }

            outPoints[index] = new V3(sumX, sumY, sumZ);
        }

        public static void PatchBoxesKernel(
            Index1D index,
            ArrayView<double> meta,
            ArrayView<double> cvs,
            ArrayView<V3> outBoxes)
        {
            int offset = (int)meta[index * 2];
            int count = (int)meta[index * 2 + 1];

            double minX = double.PositiveInfinity, minY = double.PositiveInfinity, minZ = double.PositiveInfinity;
            double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity, maxZ = double.NegativeInfinity;

            int baseIdx = offset * 4;
            for (int i = 0; i < count; i++)
            {
                int idx = baseIdx + i * 4;
                double w = cvs[idx + 3];
                double invW = (w != 0.0) ? (1.0 / w) : 1.0;
                double x = cvs[idx] * invW;
                double y = cvs[idx + 1] * invW;
                double z = cvs[idx + 2] * invW;

                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (y < minY) minY = y; if (y > maxY) maxY = y;
                if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
            }

            outBoxes[index * 2] = new V3(minX, minY, minZ);
            outBoxes[index * 2 + 1] = new V3(maxX, maxY, maxZ);
        }
    }
}
