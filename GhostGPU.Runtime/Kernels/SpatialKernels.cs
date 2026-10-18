using System;
using GhostGPU.Core;
using ILGPU;
using ILGPU.Algorithms;

namespace GhostGPU.Runtime.Kernels
{
    internal static class SpatialKernels
    {
        /// <summary>Shared tie rule for every nearest path (host, brute, grid): (distance, index) lexicographic.</summary>
        public static bool IsBetter(double dist, int index, double bestDist, int bestIndex)
        {
            return dist < bestDist || (dist == bestDist && index < bestIndex);
        }

        public static void BruteClosestPointsKernel(
            Index1D index,
            ArrayView<V3> queries,
            ArrayView<V3> targets,
            int targetCount,
            ArrayView<V3> outPoints,
            ArrayView<int> outIndices,
            ArrayView<double> outDistances)
        {
            int i = index;
            V3 q = queries[i];
            double bestDist = double.PositiveInfinity;
            int bestIdx = 0;
            V3 bestPt = targets[0];

            for (int j = 0; j < targetCount; j++)
            {
                V3 t = targets[j];
                double dist = VectorKernels.RobustDistance(q, t);
                if (IsBetter(dist, j, bestDist, bestIdx))
                {
                    bestDist = dist;
                    bestIdx = j;
                    bestPt = t;
                }
            }

            outPoints[i] = bestPt;
            outIndices[i] = bestIdx;
            outDistances[i] = bestDist;
        }

        /// <summary>
        /// Uniform grid spatial hashing: assign each target point to a 3D cell index.
        /// </summary>
        public static void AssignGridCellsKernel(
            Index1D index,
            ArrayView<V3> targets,
            int targetCount,
            V3 gridMin,
            double cellSizeInv,
            int gridDimX,
            int gridDimY,
            int gridDimZ,
            ArrayView<int> cellKeys,
            ArrayView<int> targetIndices)
        {
            int i = index;
            if (i >= targetCount) return;

            V3 p = targets[i];
            int cx = (int)((p.X - gridMin.X) * cellSizeInv);
            int cy = (int)((p.Y - gridMin.Y) * cellSizeInv);
            int cz = (int)((p.Z - gridMin.Z) * cellSizeInv);

            if (cx < 0) cx = 0; else if (cx >= gridDimX) cx = gridDimX - 1;
            if (cy < 0) cy = 0; else if (cy >= gridDimY) cy = gridDimY - 1;
            if (cz < 0) cz = 0; else if (cz >= gridDimZ) cz = gridDimZ - 1;

            int key = (cz * gridDimY + cy) * gridDimX + cx;
            cellKeys[i] = key;
            targetIndices[i] = i;
        }

        /// <summary>
        /// Identify cell boundaries (start and end indices) in sorted cellKeys.
        /// Initialized to -1.
        /// </summary>
        public static void FindCellBoundsKernel(
            Index1D index,
            ArrayView<int> sortedCellKeys,
            int targetCount,
            ArrayView<int> cellStart,
            ArrayView<int> cellEnd)
        {
            int i = index;
            if (i >= targetCount) return;

            int key = sortedCellKeys[i];
            if (i == 0 || key != sortedCellKeys[i - 1])
            {
                cellStart[key] = i;
            }
            if (i == targetCount - 1 || key != sortedCellKeys[i + 1])
            {
                cellEnd[key] = i + 1;
            }
        }

        /// <summary>
        /// Uniform spatial grid search with ring expansion.
        /// Exact stop criterion: terminates when distance to outer bounding box of next ring exceeds current bestDist.
        /// Ties broken by (distance, index) lexicographically (strict distance or dist == bestDist and j &lt; bestIdx).
        /// </summary>
        public static void GridClosestPointsKernel(
            Index1D index,
            ArrayView<V3> queries,
            ArrayView<V3> targets,
            ArrayView<int> sortedIndices,
            ArrayView<int> cellStart,
            ArrayView<int> cellEnd,
            V3 gridMin,
            double cellSize,
            double cellSizeInv,
            int gridDimX,
            int gridDimY,
            int gridDimZ,
            ArrayView<V3> outPoints,
            ArrayView<int> outIndices,
            ArrayView<double> outDistances)
        {
            int i = index;
            V3 q = queries[i];

            int cx = (int)((q.X - gridMin.X) * cellSizeInv);
            int cy = (int)((q.Y - gridMin.Y) * cellSizeInv);
            int cz = (int)((q.Z - gridMin.Z) * cellSizeInv);

            if (cx < 0) cx = 0; else if (cx >= gridDimX) cx = gridDimX - 1;
            if (cy < 0) cy = 0; else if (cy >= gridDimY) cy = gridDimY - 1;
            if (cz < 0) cz = 0; else if (cz >= gridDimZ) cz = gridDimZ - 1;

            double bestDist = double.PositiveInfinity;
            int bestIdx = int.MaxValue;
            V3 bestPt = default;

            int maxDim = gridDimX > gridDimY ? gridDimX : gridDimY;
            if (gridDimZ > maxDim) maxDim = gridDimZ;

            // Search concentric rings from radius 0 outwards
            for (int r = 0; r <= maxDim; r++)
            {
                // Stop criterion: every cell in ring r is at least (r - 1) * cellSize from q. The bound is
                // shrunk by 1e-9 relative so cell-assignment rounding and the 1-ulp error of RobustDistance can
                // never prune an exact tie (equal distance, lower index) sitting on a cell boundary.
                if (r > 0)
                {
                    double minRingDist = (r - 1) * cellSize * (1.0 - 1e-9);
                    if (minRingDist > bestDist)
                    {
                        break;
                    }
                }

                int x0 = cx - r; if (x0 < 0) x0 = 0;
                int x1 = cx + r; if (x1 >= gridDimX) x1 = gridDimX - 1;
                int y0 = cy - r; if (y0 < 0) y0 = 0;
                int y1 = cy + r; if (y1 >= gridDimY) y1 = gridDimY - 1;
                int z0 = cz - r; if (z0 < 0) z0 = 0;
                int z1 = cz + r; if (z1 >= gridDimZ) z1 = gridDimZ - 1;

                for (int z = z0; z <= z1; z++)
                {
                    bool zBoundary = (z == cz - r || z == cz + r);
                    for (int y = y0; y <= y1; y++)
                    {
                        bool yBoundary = (y == cy - r || y == cy + r);
                        for (int x = x0; x <= x1; x++)
                        {
                            bool xBoundary = (x == cx - r || x == cx + r);
                            // Only search the surface of the ring (or all of r=0)
                            if (r > 0 && !xBoundary && !yBoundary && !zBoundary)
                                continue;

                            int cellKey = (z * gridDimY + y) * gridDimX + x;
                            int start = cellStart[cellKey];
                            if (start < 0) continue;
                            int end = cellEnd[cellKey];

                            for (int k = start; k < end; k++)
                            {
                                int origIdx = sortedIndices[k];
                                V3 t = targets[origIdx];
                                double dist = VectorKernels.RobustDistance(q, t);

                                if (IsBetter(dist, origIdx, bestDist, bestIdx))
                                {
                                    bestDist = dist;
                                    bestIdx = origIdx;
                                    bestPt = t;
                                }
                            }
                        }
                    }
                }
            }

            outPoints[i] = bestPt;
            outIndices[i] = bestIdx;
            outDistances[i] = bestDist;
        }
    }
}
