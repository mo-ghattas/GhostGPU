using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    public sealed class BezierPatch
    {
        public int SourceSurface;
        public Interval U, V;
        public int DegreeU, DegreeV;
        public double[] Cvs; /* homogeneous, (DegreeU+1)*(DegreeV+1)*4 */
        public V3 BoxMin, BoxMax;
    }

    public static class BezierDecomposition
    {
        private const double KnotEps = 1e-12;

        public static void ComputeBox(double[] cvs, int count, out V3 boxMin, out V3 boxMax)
        {
            if (cvs == null || count <= 0)
            {
                boxMin = default;
                boxMax = default;
                return;
            }

            double minX = double.PositiveInfinity, minY = double.PositiveInfinity, minZ = double.PositiveInfinity;
            double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity, maxZ = double.NegativeInfinity;

            for (int i = 0; i < count; i++)
            {
                int idx = i * 4;
                double w = cvs[idx + 3];
                double invW = (w != 0.0) ? (1.0 / w) : 1.0;
                double x = cvs[idx] * invW;
                double y = cvs[idx + 1] * invW;
                double z = cvs[idx + 2] * invW;

                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (y < minY) minY = y; if (y > maxY) maxY = y;
                if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
            }

            boxMin = new V3(minX, minY, minZ);
            boxMax = new V3(maxX, maxY, maxZ);
        }

        public static void Subdivide(BezierPatch p, out BezierPatch a, out BezierPatch b, bool splitU)
        {
            if (p == null) throw new ArgumentNullException(nameof(p));

            int pu = p.DegreeU;
            int pv = p.DegreeV;
            int strideV = pv + 1;
            int totalCvs = (pu + 1) * strideV;

            a = new BezierPatch
            {
                SourceSurface = p.SourceSurface,
                DegreeU = pu,
                DegreeV = pv,
                Cvs = new double[totalCvs * 4]
            };

            b = new BezierPatch
            {
                SourceSurface = p.SourceSurface,
                DegreeU = pu,
                DegreeV = pv,
                Cvs = new double[totalCvs * 4]
            };

            if (splitU)
            {
                double uMid = 0.5 * (p.U.T0 + p.U.T1);
                a.U = new Interval(p.U.T0, uMid);
                a.V = p.V;
                b.U = new Interval(uMid, p.U.T1);
                b.V = p.V;

                unsafe
                {
                    int pyrStride = (pu + 1) * 4;
                    double* pyr = stackalloc double[(pu + 1) * pyrStride];

                    for (int j = 0; j <= pv; j++)
                    {
                        // Copy column j into row 0 of pyramid
                        for (int i = 0; i <= pu; i++)
                        {
                            int src = (i * strideV + j) * 4;
                            int dst = i * 4;
                            pyr[dst] = p.Cvs[src];
                            pyr[dst + 1] = p.Cvs[src + 1];
                            pyr[dst + 2] = p.Cvs[src + 2];
                            pyr[dst + 3] = p.Cvs[src + 3];
                        }

                        // De Casteljau subdivision at t = 0.5
                        for (int step = 1; step <= pu; step++)
                        {
                            int prevRow = (step - 1) * pyrStride;
                            int currRow = step * pyrStride;
                            for (int i = 0; i <= pu - step; i++)
                            {
                                int c0 = prevRow + i * 4;
                                int c1 = prevRow + (i + 1) * 4;
                                int cd = currRow + i * 4;
                                pyr[cd] = 0.5 * (pyr[c0] + pyr[c1]);
                                pyr[cd + 1] = 0.5 * (pyr[c0 + 1] + pyr[c1 + 1]);
                                pyr[cd + 2] = 0.5 * (pyr[c0 + 2] + pyr[c1 + 2]);
                                pyr[cd + 3] = 0.5 * (pyr[c0 + 3] + pyr[c1 + 3]);
                            }
                        }

                        // Left sub-patch a: pyr[step, 0]
                        for (int i = 0; i <= pu; i++)
                        {
                            int src = i * pyrStride;
                            int dst = (i * strideV + j) * 4;
                            a.Cvs[dst] = pyr[src];
                            a.Cvs[dst + 1] = pyr[src + 1];
                            a.Cvs[dst + 2] = pyr[src + 2];
                            a.Cvs[dst + 3] = pyr[src + 3];
                        }

                        // Right sub-patch b: pyr[pu - i, i]
                        for (int i = 0; i <= pu; i++)
                        {
                            int src = (pu - i) * pyrStride + i * 4;
                            int dst = (i * strideV + j) * 4;
                            b.Cvs[dst] = pyr[src];
                            b.Cvs[dst + 1] = pyr[src + 1];
                            b.Cvs[dst + 2] = pyr[src + 2];
                            b.Cvs[dst + 3] = pyr[src + 3];
                        }
                    }
                }
            }
            else
            {
                double vMid = 0.5 * (p.V.T0 + p.V.T1);
                a.U = p.U;
                a.V = new Interval(p.V.T0, vMid);
                b.U = p.U;
                b.V = new Interval(vMid, p.V.T1);

                unsafe
                {
                    int pyrStride = (pv + 1) * 4;
                    double* pyr = stackalloc double[(pv + 1) * pyrStride];

                    for (int i = 0; i <= pu; i++)
                    {
                        // Copy row i into row 0 of pyramid
                        for (int j = 0; j <= pv; j++)
                        {
                            int src = (i * strideV + j) * 4;
                            int dst = j * 4;
                            pyr[dst] = p.Cvs[src];
                            pyr[dst + 1] = p.Cvs[src + 1];
                            pyr[dst + 2] = p.Cvs[src + 2];
                            pyr[dst + 3] = p.Cvs[src + 3];
                        }

                        // De Casteljau subdivision at t = 0.5
                        for (int step = 1; step <= pv; step++)
                        {
                            int prevRow = (step - 1) * pyrStride;
                            int currRow = step * pyrStride;
                            for (int j = 0; j <= pv - step; j++)
                            {
                                int c0 = prevRow + j * 4;
                                int c1 = prevRow + (j + 1) * 4;
                                int cd = currRow + j * 4;
                                pyr[cd] = 0.5 * (pyr[c0] + pyr[c1]);
                                pyr[cd + 1] = 0.5 * (pyr[c0 + 1] + pyr[c1 + 1]);
                                pyr[cd + 2] = 0.5 * (pyr[c0 + 2] + pyr[c1 + 2]);
                                pyr[cd + 3] = 0.5 * (pyr[c0 + 3] + pyr[c1 + 3]);
                            }
                        }

                        // Left sub-patch a: pyr[step, 0]
                        for (int j = 0; j <= pv; j++)
                        {
                            int src = j * pyrStride;
                            int dst = (i * strideV + j) * 4;
                            a.Cvs[dst] = pyr[src];
                            a.Cvs[dst + 1] = pyr[src + 1];
                            a.Cvs[dst + 2] = pyr[src + 2];
                            a.Cvs[dst + 3] = pyr[src + 3];
                        }

                        // Right sub-patch b: pyr[pv - j, j]
                        for (int j = 0; j <= pv; j++)
                        {
                            int src = (pv - j) * pyrStride + j * 4;
                            int dst = (i * strideV + j) * 4;
                            b.Cvs[dst] = pyr[src];
                            b.Cvs[dst + 1] = pyr[src + 1];
                            b.Cvs[dst + 2] = pyr[src + 2];
                            b.Cvs[dst + 3] = pyr[src + 3];
                        }
                    }
                }
            }

            ComputeBox(a.Cvs, totalCvs, out a.BoxMin, out a.BoxMax);
            ComputeBox(b.Cvs, totalCvs, out b.BoxMin, out b.BoxMax);
        }

        public static List<BezierPatch> Decompose(NurbsSurface s, int sourceIndex)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));

            int pu = s.DegreeU;
            int pv = s.DegreeV;

            // Refine in U
            double[] currentKnotsU = (double[])s.KnotsU.Clone();
            int currentCvCountU = s.CvCountU;
            int currentCvCountV = s.CvCountV;
            double[] currentCvs = (double[])s.Cvs.Clone();

            double uStart = currentKnotsU[pu];
            double uEnd = currentKnotsU[currentCvCountU];

            // Identify distinct interior knots in U and how many insertions are needed
            List<double> distinctU = new List<double>();
            for (int i = pu + 1; i < currentCvCountU; i++)
            {
                double u = currentKnotsU[i];
                if (u > uStart + KnotEps && u < uEnd - KnotEps)
                {
                    if (distinctU.Count == 0 || Math.Abs(u - distinctU[distinctU.Count - 1]) > KnotEps)
                    {
                        distinctU.Add(u);
                    }
                }
            }

            foreach (double knotVal in distinctU)
            {
                int currentMult = 0;
                for (int i = pu; i <= currentCvCountU; i++)
                {
                    if (Math.Abs(currentKnotsU[i] - knotVal) <= KnotEps)
                        currentMult++;
                }

                int insertions = pu - currentMult;
                for (int ins = 0; ins < insertions; ins++)
                {
                    InsertKnotU(knotVal, pu, ref currentKnotsU, ref currentCvCountU, currentCvCountV, ref currentCvs);
                }
            }

            // Refine in V
            double[] currentKnotsV = (double[])s.KnotsV.Clone();
            double vStart = currentKnotsV[pv];
            double vEnd = currentKnotsV[currentCvCountV];

            List<double> distinctV = new List<double>();
            for (int j = pv + 1; j < currentCvCountV; j++)
            {
                double v = currentKnotsV[j];
                if (v > vStart + KnotEps && v < vEnd - KnotEps)
                {
                    if (distinctV.Count == 0 || Math.Abs(v - distinctV[distinctV.Count - 1]) > KnotEps)
                    {
                        distinctV.Add(v);
                    }
                }
            }

            foreach (double knotVal in distinctV)
            {
                int currentMult = 0;
                for (int j = pv; j <= currentCvCountV; j++)
                {
                    if (Math.Abs(currentKnotsV[j] - knotVal) <= KnotEps)
                        currentMult++;
                }

                int insertions = pv - currentMult;
                for (int ins = 0; ins < insertions; ins++)
                {
                    InsertKnotV(knotVal, pv, currentCvCountU, ref currentKnotsV, ref currentCvCountV, ref currentCvs);
                }
            }

            // Collect unique domain knot boundaries
            List<double> spansU = new List<double> { currentKnotsU[pu] };
            for (int i = pu + 1; i <= currentCvCountU; i++)
            {
                double u = currentKnotsU[i];
                if (u > spansU[spansU.Count - 1] + KnotEps)
                    spansU.Add(u);
            }

            List<double> spansV = new List<double> { currentKnotsV[pv] };
            for (int j = pv + 1; j <= currentCvCountV; j++)
            {
                double v = currentKnotsV[j];
                if (v > spansV[spansV.Count - 1] + KnotEps)
                    spansV.Add(v);
            }

            int numSpansU = spansU.Count - 1;
            int numSpansV = spansV.Count - 1;
            int patchCvCount = (pu + 1) * (pv + 1);

            List<BezierPatch> patches = new List<BezierPatch>(numSpansU * numSpansV);

            for (int ku = 0; ku < numSpansU; ku++)
            {
                Interval intU = new Interval(spansU[ku], spansU[ku + 1]);
                for (int kv = 0; kv < numSpansV; kv++)
                {
                    Interval intV = new Interval(spansV[kv], spansV[kv + 1]);

                    var patch = new BezierPatch
                    {
                        SourceSurface = sourceIndex,
                        U = intU,
                        V = intV,
                        DegreeU = pu,
                        DegreeV = pv,
                        Cvs = new double[patchCvCount * 4]
                    };

                    for (int i = 0; i <= pu; i++)
                    {
                        int row = ku * pu + i;
                        int srcRow = row * currentCvCountV;
                        int dstRow = i * (pv + 1);
                        for (int j = 0; j <= pv; j++)
                        {
                            int col = kv * pv + j;
                            int srcIdx = (srcRow + col) * 4;
                            int dstIdx = (dstRow + j) * 4;
                            patch.Cvs[dstIdx] = currentCvs[srcIdx];
                            patch.Cvs[dstIdx + 1] = currentCvs[srcIdx + 1];
                            patch.Cvs[dstIdx + 2] = currentCvs[srcIdx + 2];
                            patch.Cvs[dstIdx + 3] = currentCvs[srcIdx + 3];
                        }
                    }

                    ComputeBox(patch.Cvs, patchCvCount, out patch.BoxMin, out patch.BoxMax);
                    patches.Add(patch);
                }
            }

            return patches;
        }

        private static void InsertKnotU(double u, int p, ref double[] knots, ref int cvCountU, int cvCountV, ref double[] cvs)
        {
            // Find span k such that knots[k] <= u < knots[k+1]
            int k = -1;
            for (int i = p; i < cvCountU; i++)
            {
                if (u >= knots[i] - KnotEps && u < knots[i + 1] - KnotEps)
                {
                    k = i;
                    break;
                }
            }
            if (k == -1)
                k = cvCountU - 1;

            double[] newKnots = new double[knots.Length + 1];
            for (int i = 0; i <= k; i++) newKnots[i] = knots[i];
            newKnots[k + 1] = u;
            for (int i = k + 1; i < knots.Length; i++) newKnots[i + 1] = knots[i];

            int newCvCountU = cvCountU + 1;
            double[] newCvs = new double[newCvCountU * cvCountV * 4];

            // For each column in V
            for (int j = 0; j < cvCountV; j++)
            {
                for (int i = 0; i <= k - p; i++)
                {
                    int src = (i * cvCountV + j) * 4;
                    int dst = (i * cvCountV + j) * 4;
                    newCvs[dst] = cvs[src];
                    newCvs[dst + 1] = cvs[src + 1];
                    newCvs[dst + 2] = cvs[src + 2];
                    newCvs[dst + 3] = cvs[src + 3];
                }

                for (int i = k - p + 1; i <= k; i++)
                {
                    double denom = knots[i + p] - knots[i];
                    double alpha = Math.Abs(denom) > 1e-15 ? (u - knots[i]) / denom : 0.0;
                    int p0 = ((i - 1) * cvCountV + j) * 4;
                    int p1 = (i * cvCountV + j) * 4;
                    int dst = (i * cvCountV + j) * 4;
                    newCvs[dst] = (1.0 - alpha) * cvs[p0] + alpha * cvs[p1];
                    newCvs[dst + 1] = (1.0 - alpha) * cvs[p0 + 1] + alpha * cvs[p1 + 1];
                    newCvs[dst + 2] = (1.0 - alpha) * cvs[p0 + 2] + alpha * cvs[p1 + 2];
                    newCvs[dst + 3] = (1.0 - alpha) * cvs[p0 + 3] + alpha * cvs[p1 + 3];
                }

                for (int i = k + 1; i < newCvCountU; i++)
                {
                    int src = ((i - 1) * cvCountV + j) * 4;
                    int dst = (i * cvCountV + j) * 4;
                    newCvs[dst] = cvs[src];
                    newCvs[dst + 1] = cvs[src + 1];
                    newCvs[dst + 2] = cvs[src + 2];
                    newCvs[dst + 3] = cvs[src + 3];
                }
            }

            knots = newKnots;
            cvCountU = newCvCountU;
            cvs = newCvs;
        }

        private static void InsertKnotV(double v, int p, int cvCountU, ref double[] knots, ref int cvCountV, ref double[] cvs)
        {
            // Find span k such that knots[k] <= v < knots[k+1]
            int k = -1;
            for (int j = p; j < cvCountV; j++)
            {
                if (v >= knots[j] - KnotEps && v < knots[j + 1] - KnotEps)
                {
                    k = j;
                    break;
                }
            }
            if (k == -1)
                k = cvCountV - 1;

            double[] newKnots = new double[knots.Length + 1];
            for (int j = 0; j <= k; j++) newKnots[j] = knots[j];
            newKnots[k + 1] = v;
            for (int j = k + 1; j < knots.Length; j++) newKnots[j + 1] = knots[j];

            int newCvCountV = cvCountV + 1;
            double[] newCvs = new double[cvCountU * newCvCountV * 4];

            // For each row in U
            for (int i = 0; i < cvCountU; i++)
            {
                int srcRow = i * cvCountV;
                int dstRow = i * newCvCountV;

                for (int j = 0; j <= k - p; j++)
                {
                    int src = (srcRow + j) * 4;
                    int dst = (dstRow + j) * 4;
                    newCvs[dst] = cvs[src];
                    newCvs[dst + 1] = cvs[src + 1];
                    newCvs[dst + 2] = cvs[src + 2];
                    newCvs[dst + 3] = cvs[src + 3];
                }

                for (int j = k - p + 1; j <= k; j++)
                {
                    double denom = knots[j + p] - knots[j];
                    double alpha = Math.Abs(denom) > 1e-15 ? (v - knots[j]) / denom : 0.0;
                    int p0 = (srcRow + j - 1) * 4;
                    int p1 = (srcRow + j) * 4;
                    int dst = (dstRow + j) * 4;
                    newCvs[dst] = (1.0 - alpha) * cvs[p0] + alpha * cvs[p1];
                    newCvs[dst + 1] = (1.0 - alpha) * cvs[p0 + 1] + alpha * cvs[p1 + 1];
                    newCvs[dst + 2] = (1.0 - alpha) * cvs[p0 + 2] + alpha * cvs[p1 + 2];
                    newCvs[dst + 3] = (1.0 - alpha) * cvs[p0 + 3] + alpha * cvs[p1 + 3];
                }

                for (int j = k + 1; j < newCvCountV; j++)
                {
                    int src = (srcRow + j - 1) * 4;
                    int dst = (dstRow + j) * 4;
                    newCvs[dst] = cvs[src];
                    newCvs[dst + 1] = cvs[src + 1];
                    newCvs[dst + 2] = cvs[src + 2];
                    newCvs[dst + 3] = cvs[src + 3];
                }
            }

            knots = newKnots;
            cvCountV = newCvCountV;
            cvs = newCvs;
        }
    }
}
