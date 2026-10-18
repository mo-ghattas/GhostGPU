using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    /// <summary>
    /// Cubic non-rational least-squares fit of the 3D curve and both UV pullbacks on one shared chord-length
    /// parameterization. Ends interpolated exactly; spans whose 3D error (curve, S_A(uvA), S_B(uvB) vs the exact
    /// intersection samples) exceeds tol/2 are halved until all pass.
    /// </summary>
    internal static class SsiFit
    {
        const int Dim = 7;   // x y z uA vA uB vB

        public static SsiBranch Fit(SsiTracer tr, SsiPiece pc, double tol, List<string> warnings)
        {
            // densify with exact midpoints so the fit is checked between marching nodes too
            var X = new List<double[]>(); var P = new List<V3>();
            for (int i = 0; i < pc.X.Count; i++)
            {
                X.Add(pc.X[i]); P.Add(pc.P[i]);
                if (i + 1 < pc.X.Count && tr.Midpoint(pc.X[i], pc.X[i + 1], out double[] xm, out V3 pm)) { X.Add(xm); P.Add(pm); }
            }
            int m = X.Count;
            var t = new double[m];
            for (int k = 1; k < m; k++) t[k] = t[k - 1] + Math.Max((P[k] - P[k - 1]).Length, 1e-15);
            var D = new double[m][];
            for (int k = 0; k < m; k++) D[k] = new[] { P[k].X, P[k].Y, P[k].Z, X[k][0], X[k][1], X[k][2], X[k][3] };

            int spans = Math.Max(1, (m - 1) / 8);
            var inner = new List<double>();
            for (int j = 1; j < spans; j++)
            {
                double v = t[(int)Math.Round((double)j * (m - 1) / spans)];
                if (v > (inner.Count > 0 ? inner[inner.Count - 1] : t[0]) && v < t[m - 1]) inner.Add(v);
            }

            double[] knots = null; double[][] cv = null; double err = double.MaxValue;
            for (int iter = 0; iter < 40; iter++)
            {
                knots = FullKnots(inner, t[0], t[m - 1]);
                int n = knots.Length - 4;
                cv = Solve(knots, n, t, D);
                // per-span max error
                var spanErr = new double[n];
                err = 0;
                for (int k = 0; k < m; k++)
                {
                    var c = Eval(knots, n, cv, t[k], out int sp);
                    double e3 = Dist3(c, D[k]);
                    double ea = (tr.PointA(c[3], c[4]) - P[k]).Length;
                    double eb = (tr.PointB(c[5], c[6]) - P[k]).Length;
                    double e = Math.Max(e3, Math.Max(ea, eb));
                    if (e > spanErr[sp]) spanErr[sp] = e;
                    if (e > err) err = e;
                }
                if (err <= 0.5 * tol) break;
                var newInner = new List<double>(inner);
                bool added = false;
                for (int s = 3; s < n; s++)
                {
                    if (spanErr[s] <= 0.5 * tol) continue;
                    double a = knots[s], b = knots[s + 1];
                    if (b - a < 1e-9 * (t[m - 1] - t[0])) continue;
                    newInner.Add(0.5 * (a + b)); added = true;
                }
                if (!added) break;
                newInner.Sort();
                inner = newInner;
            }
            if (err > 0.5 * tol)
                warnings.Add($"faces A{pc.FaceA}/B{pc.FaceB}: SSI curve fit error {err:G3} exceeds tol/2");

            int ncv = knots.Length - 4;
            var c3 = new double[ncv * 4]; var ca = new double[ncv * 4]; var cb = new double[ncv * 4];
            for (int i = 0; i < ncv; i++)
            {
                c3[i * 4] = cv[i][0]; c3[i * 4 + 1] = cv[i][1]; c3[i * 4 + 2] = cv[i][2]; c3[i * 4 + 3] = 1;
                ca[i * 4] = cv[i][3]; ca[i * 4 + 1] = cv[i][4]; ca[i * 4 + 3] = 1;
                cb[i * 4] = cv[i][5]; cb[i * 4 + 1] = cv[i][6]; cb[i * 4 + 3] = 1;
            }
            var br = new SsiBranch
            {
                FaceA = pc.FaceA, FaceB = pc.FaceB, IsClosed = pc.Closed,
                Curve3 = new NurbsCurve(3, 3, false, knots, c3),
                UvA = new NurbsCurve(2, 3, false, (double[])knots.Clone(), ca),
                UvB = new NurbsCurve(2, 3, false, (double[])knots.Clone(), cb),
            };
            // MaxDeviation at dense samples
            double dev = 0;
            for (int s = 3; s < ncv; s++)
            {
                if (knots[s + 1] <= knots[s]) continue;
                for (int j = 0; j <= 8; j++)
                {
                    double tt = knots[s] + (knots[s + 1] - knots[s]) * j / 8.0;
                    var c = Eval(knots, ncv, cv, tt, out _);
                    V3 pa = tr.PointA(c[3], c[4]), pb = tr.PointB(c[5], c[6]);
                    dev = Math.Max(dev, Math.Max((pa - pb).Length, (new V3(c[0], c[1], c[2]) - pa).Length));
                }
            }
            br.MaxDeviation = dev;
            return br;
        }

        static double Dist3(double[] c, double[] d)
        {
            double x = c[0] - d[0], y = c[1] - d[1], z = c[2] - d[2];
            return Math.Sqrt(x * x + y * y + z * z);
        }

        static double[] FullKnots(List<double> inner, double t0, double t1)
        {
            var k = new double[inner.Count + 8];
            for (int i = 0; i < 4; i++) { k[i] = t0; k[k.Length - 1 - i] = t1; }
            for (int i = 0; i < inner.Count; i++) k[4 + i] = inner[i];
            return k;
        }

        /// <summary>Cubic B-spline basis at t: returns span s (3..n-1) and N[0..3] for CVs s-3..s.</summary>
        static void Basis(double[] U, int n, double t, out int s, double[] N)
        {
            s = n - 1;
            if (t < U[n])
            {
                int lo = 3, hi = n;
                while (hi - lo > 1) { int mid = (lo + hi) / 2; if (t < U[mid]) hi = mid; else lo = mid; }
                s = lo;
            }
            var left = new double[4]; var right = new double[4];
            N[0] = 1;
            for (int j = 1; j <= 3; j++)
            {
                left[j] = t - U[s + 1 - j]; right[j] = U[s + j] - t;
                double saved = 0;
                for (int r = 0; r < j; r++)
                {
                    double tmp = N[r] / (right[r + 1] + left[j - r]);
                    N[r] = saved + right[r + 1] * tmp;
                    saved = left[j - r] * tmp;
                }
                N[j] = saved;
            }
        }

        static double[] Eval(double[] U, int n, double[][] cv, double t, out int s)
        {
            var N = new double[4];
            Basis(U, n, t, out s, N);
            var r = new double[Dim];
            for (int j = 0; j < 4; j++)
                for (int d = 0; d < Dim; d++) r[d] += N[j] * cv[s - 3 + j][d];
            return r;
        }

        /// <summary>Banded normal equations with fixed end CVs and a tiny second-difference regularizer.</summary>
        static double[][] Solve(double[] U, int n, double[] t, double[][] D)
        {
            int m = t.Length;
            var cv = new double[n][];
            for (int i = 0; i < n; i++) cv[i] = new double[Dim];
            Array.Copy(D[0], cv[0], Dim); Array.Copy(D[m - 1], cv[n - 1], Dim);
            int nu = n - 2;
            if (nu <= 0) return cv;
            const int W = 3;
            var A = new double[nu, W + 1];
            var B = new double[nu, Dim];
            var N = new double[4];
            for (int k = 0; k < m; k++)
            {
                Basis(U, n, t[k], out int s, N);
                var rhs = (double[])D[k].Clone();
                for (int j = 0; j < 4; j++)
                {
                    int ci = s - 3 + j;
                    if (ci == 0 || ci == n - 1) for (int d = 0; d < Dim; d++) rhs[d] -= N[j] * cv[ci][d];
                }
                for (int j = 0; j < 4; j++)
                {
                    int a = s - 3 + j;
                    if (a == 0 || a == n - 1 || N[j] == 0) continue;
                    for (int d = 0; d < Dim; d++) B[a - 1, d] += N[j] * rhs[d];
                    for (int l = j; l < 4; l++)
                    {
                        int b = s - 3 + l;
                        if (b == 0 || b == n - 1) continue;
                        A[a - 1, b - a] += N[j] * N[l];
                    }
                }
            }
            double lam = 1e-6 * m / n;
            double[] c = { 1, -2, 1 };
            for (int i = 1; i <= n - 2; i++)
            {
                for (int p = 0; p < 3; p++)
                {
                    int a = i - 1 + p;
                    if (a == 0 || a == n - 1) continue;
                    for (int q = 0; q < 3; q++)
                    {
                        int b = i - 1 + q;
                        if (b == 0 || b == n - 1) { for (int d = 0; d < Dim; d++) B[a - 1, d] -= lam * c[p] * c[q] * cv[b][d]; continue; }
                        if (b >= a) A[a - 1, b - a] += lam * c[p] * c[q];
                    }
                }
            }
            // banded Cholesky A = L L^T, L[i, i-j] stored at Lb[i, i-j]
            var L = new double[nu, W + 1];
            for (int i = 0; i < nu; i++)
            {
                for (int j = Math.Max(0, i - W); j <= i; j++)
                {
                    double sum = A[j, i - j];
                    for (int k = Math.Max(0, i - W); k < j; k++) sum -= L[i, i - k] * L[j, j - k];
                    if (i == j) L[i, 0] = Math.Sqrt(Math.Max(sum, 1e-300));
                    else L[i, i - j] = sum / L[j, 0];
                }
            }
            for (int d = 0; d < Dim; d++)
            {
                var y = new double[nu];
                for (int i = 0; i < nu; i++)
                {
                    double sum = B[i, d];
                    for (int k = Math.Max(0, i - W); k < i; k++) sum -= L[i, i - k] * y[k];
                    y[i] = sum / L[i, 0];
                }
                for (int i = nu - 1; i >= 0; i--)
                {
                    double sum = y[i];
                    for (int k = i + 1; k <= Math.Min(nu - 1, i + W); k++) sum -= L[k, k - i] * cv[k + 1][d];
                    cv[i + 1][d] = sum / L[i, 0];
                }
            }
            return cv;
        }
    }
}
