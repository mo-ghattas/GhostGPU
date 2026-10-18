using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    /// <summary>NURBS utilities used by FaceSplit: exact sub-curves (knot insertion), reversal, translation,
    /// interpolation, closest points and trimmed-face area by Green's theorem + Gauss quadrature.</summary>
    internal static class SplitGeom
    {
        public static V3 Uv(double u, double v) => new V3(u, v, 0);
        public static double Dist(V3 a, V3 b) => (a - b).Length;
        public static V3 Eval(NurbsSurface s, V3 uv) => NurbsEval.SurfacePoint(s, uv.X, uv.Y);

        // ------------------------------------------------------------------ knots / CV surgery

        static void InsertOnce(ref double[] knots, ref double[] cvs, int p, double x)
        {
            int n = cvs.Length / 4;
            int k = NurbsEval.FindSpan(knots, p, n, x);
            var nk = new double[knots.Length + 1];
            for (int i = 0; i <= k; i++) nk[i] = knots[i];
            nk[k + 1] = x;
            for (int i = k + 1; i < knots.Length; i++) nk[i + 1] = knots[i];
            var nc = new double[cvs.Length + 4];
            for (int i = 0; i <= n; i++)
            {
                for (int d = 0; d < 4; d++)
                {
                    double val;
                    if (i <= k - p) val = cvs[i * 4 + d];
                    else if (i >= k + 1) val = cvs[(i - 1) * 4 + d];
                    else
                    {
                        double a = (x - knots[i]) / (knots[i + p] - knots[i]);
                        val = a * cvs[i * 4 + d] + (1 - a) * cvs[(i - 1) * 4 + d];
                    }
                    nc[i * 4 + d] = val;
                }
            }
            knots = nk; cvs = nc;
        }

        static int Mult(double[] knots, double x)
        {
            int m = 0;
            foreach (var k in knots) if (k == x) m++;
            return m;
        }

        static double SnapToKnot(double[] knots, double x, double eps)
        {
            foreach (var k in knots) if (Math.Abs(k - x) <= eps) return k;
            return x;
        }

        /// <summary>Exact restriction of c to [a, b] (a &lt; b, inside the domain) by knot insertion.</summary>
        public static NurbsCurve SubCurve(NurbsCurve c, double a, double b)
        {
            int p = c.Degree;
            var dom = c.Domain;
            double eps = 1e-13 * Math.Max(1.0, Math.Abs(dom.T0) + Math.Abs(dom.T1));
            var knots = (double[])c.Knots.Clone();
            var cvs = (double[])c.Cvs.Clone();
            a = Math.Max(dom.T0, SnapToKnot(knots, a, eps));
            b = Math.Min(dom.T1, SnapToKnot(knots, b, eps));
            if (!(b > a)) throw new ArgumentException($"SubCurve: empty interval [{a}, {b}]");
            if (a > dom.T0) while (Mult(knots, a) < p) InsertOnce(ref knots, ref cvs, p, a);
            if (b < dom.T1) while (Mult(knots, b) < p) InsertOnce(ref knots, ref cvs, p, b);
            int ia = Array.IndexOf(knots, a), ma = Mult(knots, a);
            int ib = Array.IndexOf(knots, b);
            int sa = ia + ma - 1 - p, eb = ib - 1;
            var nk = new List<double>();
            for (int i = 0; i <= p; i++) nk.Add(a);
            foreach (var k in knots) if (k > a && k < b) nk.Add(k);
            for (int i = 0; i <= p; i++) nk.Add(b);
            int ncv = eb - sa + 1;
            if (ncv + p + 1 != nk.Count) throw new InvalidOperationException("SubCurve: knot/CV count mismatch.");
            var ncvs = new double[ncv * 4];
            Array.Copy(cvs, sa * 4, ncvs, 0, ncv * 4);
            return new NurbsCurve(c.Dimension, p, c.IsRational, nk.ToArray(), ncvs);
        }

        public static NurbsCurve Reverse(NurbsCurve c)
        {
            int n = c.CvCount, L = c.Knots.Length;
            double s = c.Knots[0] + c.Knots[L - 1];
            var k = new double[L];
            for (int i = 0; i < L; i++) k[i] = s - c.Knots[L - 1 - i];
            var cv = new double[n * 4];
            for (int i = 0; i < n; i++) Array.Copy(c.Cvs, (n - 1 - i) * 4, cv, i * 4, 4);
            return new NurbsCurve(c.Dimension, c.Degree, c.IsRational, k, cv);
        }

        public static NurbsCurve Translate(NurbsCurve c, double dx, double dy)
        {
            if (dx == 0 && dy == 0) return c;
            var cv = (double[])c.Cvs.Clone();
            for (int i = 0; i < c.CvCount; i++) { cv[i * 4] += dx * cv[i * 4 + 3]; cv[i * 4 + 1] += dy * cv[i * 4 + 3]; }
            return new NurbsCurve(c.Dimension, c.Degree, c.IsRational, c.Knots, cv);
        }

        /// <summary>Moves the end CVs (clamped curve: they are the end points) exactly onto the given points.</summary>
        public static NurbsCurve SnapEnds(NurbsCurve c, V3? start, V3? end)
        {
            var cv = (double[])c.Cvs.Clone();
            void Put(int i, V3 p)
            {
                double w = cv[i * 4 + 3];
                cv[i * 4] = p.X * w; cv[i * 4 + 1] = p.Y * w; cv[i * 4 + 2] = (c.Dimension == 2 ? 0 : p.Z) * w;
            }
            if (start.HasValue) Put(0, start.Value);
            if (end.HasValue) Put(c.CvCount - 1, end.Value);
            return new NurbsCurve(c.Dimension, c.Degree, c.IsRational, c.Knots, cv);
        }

        public static NurbsCurve Line(V3 a, V3 b, int dim, double t0, double t1)
        {
            return new NurbsCurve(dim, 1, false, new[] { t0, t0, t1, t1 },
                new[] { a.X, a.Y, dim == 2 ? 0 : a.Z, 1, b.X, b.Y, dim == 2 ? 0 : b.Z, 1 });
        }

        static void Basis(double[] U, int p, int span, double t, double[] N)
        {
            var left = new double[p + 1]; var right = new double[p + 1];
            N[0] = 1;
            for (int j = 1; j <= p; j++)
            {
                left[j] = t - U[span + 1 - j]; right[j] = U[span + j] - t;
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

        /// <summary>Global interpolation (The NURBS Book A9.1): non-rational curve through pts at params t
        /// (strictly increasing), knots by averaging.</summary>
        public static NurbsCurve Interpolate(IList<V3> pts, IList<double> t, int degree, int dim)
        {
            int n = pts.Count;
            int p = Math.Min(degree, n - 1);
            if (p < 1) throw new ArgumentException("Interpolate: need at least 2 points.");
            var U = new double[n + p + 1];
            for (int i = 0; i <= p; i++) { U[i] = t[0]; U[n + i] = t[n - 1]; }
            for (int j = 1; j < n - p; j++)
            {
                double s = 0; for (int i = j; i < j + p; i++) s += t[i];
                U[j + p] = s / p;
            }
            var A = new double[n, n];
            var N = new double[p + 1];
            for (int i = 0; i < n; i++)
            {
                int span = NurbsEval.FindSpan(U, p, n, t[i]);
                Basis(U, p, span, t[i], N);
                for (int j = 0; j <= p; j++) A[i, span - p + j] = N[j];
            }
            var B = new double[n, 3];
            for (int i = 0; i < n; i++) { B[i, 0] = pts[i].X; B[i, 1] = pts[i].Y; B[i, 2] = pts[i].Z; }
            Solve(A, B, n);
            var cv = new double[n * 4];
            for (int i = 0; i < n; i++) { cv[i * 4] = B[i, 0]; cv[i * 4 + 1] = B[i, 1]; cv[i * 4 + 2] = dim == 2 ? 0 : B[i, 2]; cv[i * 4 + 3] = 1; }
            return new NurbsCurve(dim, p, false, U, cv);
        }

        static void Solve(double[,] A, double[,] B, int n)
        {
            int m = B.GetLength(1);
            for (int c = 0; c < n; c++)
            {
                int piv = c; double best = Math.Abs(A[c, c]);
                for (int r = c + 1; r < n; r++) if (Math.Abs(A[r, c]) > best) { best = Math.Abs(A[r, c]); piv = r; }
                if (best == 0) throw new InvalidOperationException("Interpolate: singular system.");
                if (piv != c)
                {
                    for (int k = 0; k < n; k++) { var x = A[c, k]; A[c, k] = A[piv, k]; A[piv, k] = x; }
                    for (int k = 0; k < m; k++) { var x = B[c, k]; B[c, k] = B[piv, k]; B[piv, k] = x; }
                }
                for (int r = c + 1; r < n; r++)
                {
                    double f = A[r, c] / A[c, c];
                    if (f == 0) continue;
                    for (int k = c; k < n; k++) A[r, k] -= f * A[c, k];
                    for (int k = 0; k < m; k++) B[r, k] -= f * B[c, k];
                }
            }
            for (int c = n - 1; c >= 0; c--)
                for (int k = 0; k < m; k++)
                {
                    double s = B[c, k];
                    for (int j = c + 1; j < n; j++) s -= A[c, j] * B[j, k];
                    B[c, k] = s / A[c, c];
                }
        }

        // ------------------------------------------------------------------ sampling

        /// <summary>Distinct parameter breakpoints of c inside [t0, t1] (ends included).</summary>
        public static List<double> Breaks(NurbsCurve c, double t0, double t1)
        {
            var r = new List<double> { t0 };
            foreach (var k in c.Knots) if (k > t0 && k < t1 && k > r[r.Count - 1]) r.Add(k);
            r.Add(t1);
            return r;
        }

        public static void Sample(NurbsCurve c, double t0, double t1, out double[] ts, out V3[] pts)
        {
            var br = Breaks(c, t0, t1);
            int spans = br.Count - 1;
            int per = c.Degree == 1 && !c.IsRational ? 1 : Math.Max(4, Math.Min(32, 512 / Math.Max(1, spans)));
            var tl = new List<double>();
            for (int s = 0; s < spans; s++)
                for (int i = 0; i < per; i++) tl.Add(br[s] + (br[s + 1] - br[s]) * i / per);
            tl.Add(t1);
            ts = tl.ToArray();
            pts = new V3[ts.Length];
            for (int i = 0; i < ts.Length; i++) pts[i] = NurbsEval.CurvePoint(c, ts[i]);
        }

        // ------------------------------------------------------------------ closest points

        /// <summary>Closest parameter on a 3D curve within [t0, t1].</summary>
        public static double ClosestOnCurve(NurbsCurve c, V3 x, double t0, double t1)
        {
            Sample(c, t0, t1, out var ts, out var ps);
            int bi = 0; double bd = double.MaxValue;
            for (int i = 0; i < ts.Length; i++) { double d = Dist(ps[i], x); if (d < bd) { bd = d; bi = i; } }
            double t = ts[bi];
            // refine on the neighbouring segments: project onto segments first
            for (int i = Math.Max(0, bi - 1); i < Math.Min(ts.Length - 1, bi + 1); i++)
            {
                var a = ps[i]; var ab = ps[i + 1] - a; double L2 = V3.Dot(ab, ab);
                if (L2 <= 0) continue;
                double lam = Math.Max(0, Math.Min(1, V3.Dot(x - a, ab) / L2));
                var q = a + ab * lam;
                if (Dist(q, x) < bd) { bd = Dist(q, x); t = ts[i] + lam * (ts[i + 1] - ts[i]); }
            }
            for (int it = 0; it < 30; it++)
            {
                var d = NurbsEval.CurveDerivatives(c, t, 2);
                var r = d[0] - x;
                double f1 = V3.Dot(r, d[1]), f2 = V3.Dot(d[1], d[1]) + V3.Dot(r, d[2]);
                if (f2 <= 1e-300) break;
                double nt = Math.Max(t0, Math.Min(t1, t - f1 / f2));
                if (Math.Abs(nt - t) <= 1e-15 * Math.Max(1, Math.Abs(t))) { t = nt; break; }
                // keep the best (Newton may wander on non-convex parts)
                if (Dist(NurbsEval.CurvePoint(c, nt), x) > Dist(d[0], x) * (1 + 1e-12) + 1e-15) break;
                t = nt;
            }
            return t;
        }

        /// <summary>Closest parameter of a curve-on-surface S(c2(s)) to x within [t0, t1], seeded by s0. Returns 3D distance.</summary>
        public static double ProjectCos(NurbsSurface srf, NurbsCurve c2, double t0, double t1, V3 x, ref double s)
        {
            double best = Dist(Eval(srf, NurbsEval.CurvePoint(c2, s)), x);
            for (int it = 0; it < 40; it++)
            {
                var cd = NurbsEval.CurveDerivatives(c2, s, 1);
                var sd = NurbsEval.SurfaceDerivatives(srf, cd[0].X, cd[0].Y, 1);
                var T = sd[1, 0] * cd[1].X + sd[0, 1] * cd[1].Y;
                var r = sd[0, 0] - x;
                double f2 = V3.Dot(T, T);
                if (f2 <= 1e-300) break;
                double ns = Math.Max(t0, Math.Min(t1, s - V3.Dot(r, T) / f2));
                double nd = Dist(Eval(srf, NurbsEval.CurvePoint(c2, ns)), x);
                if (nd > best * (1 + 1e-12) + 1e-15) break;
                bool done = Math.Abs(ns - s) <= 1e-15 * Math.Max(1, Math.Abs(s));
                s = ns; best = nd;
                if (done) break;
            }
            return best;
        }

        /// <summary>Newton closest point on a surface. Closed directions wrap, open ones clamp.</summary>
        public static double ClosestOnSurface(NurbsSurface s, V3 x, ref double u, ref double v)
        {
            var du = s.DomainU; var dv = s.DomainV;
            for (int it = 0; it < 50; it++)
            {
                var d = NurbsEval.SurfaceDerivatives(s, u, v, 2);
                var r = d[0, 0] - x;
                double a = V3.Dot(d[1, 0], d[1, 0]) + V3.Dot(r, d[2, 0]);
                double b = V3.Dot(d[1, 0], d[0, 1]) + V3.Dot(r, d[1, 1]);
                double c = V3.Dot(d[0, 1], d[0, 1]) + V3.Dot(r, d[0, 2]);
                double g0 = V3.Dot(r, d[1, 0]), g1 = V3.Dot(r, d[0, 1]);
                double det = a * c - b * b;
                double su, sv;
                if (Math.Abs(det) > 1e-300 && a > 0 && det > 0) { su = -(c * g0 - b * g1) / det; sv = -(a * g1 - b * g0) / det; }
                else
                {   // gradient step fallback
                    double ga = V3.Dot(d[1, 0], d[1, 0]), gc = V3.Dot(d[0, 1], d[0, 1]);
                    su = ga > 0 ? -g0 / ga : 0; sv = gc > 0 ? -g1 / gc : 0;
                }
                double nu = u + su, nv = v + sv;
                nu = s.IsClosedU ? Wrap(nu, du) : Math.Max(du.T0, Math.Min(du.T1, nu));
                nv = s.IsClosedV ? Wrap(nv, dv) : Math.Max(dv.T0, Math.Min(dv.T1, nv));
                bool done = Math.Abs(su) <= 1e-15 * (1 + Math.Abs(u)) && Math.Abs(sv) <= 1e-15 * (1 + Math.Abs(v));
                u = nu; v = nv;
                if (done) break;
            }
            return Dist(NurbsEval.SurfacePoint(s, u, v), x);
        }

        public static double Wrap(double x, Interval d)
        {
            double P = d.Length;
            double r = x - Math.Floor((x - d.T0) / P) * P;
            if (r >= d.T1) r -= P;
            if (r < d.T0) r = d.T0;
            return r;
        }

        /// <summary>Grid seeded global closest point.</summary>
        public static double ClosestOnSurfaceGlobal(NurbsSurface s, V3 x, out double u, out double v)
        {
            var du = s.DomainU; var dv = s.DomainV;
            int n = 24; double bd = double.MaxValue; u = du.T0; v = dv.T0;
            for (int i = 0; i <= n; i++)
                for (int j = 0; j <= n; j++)
                {
                    double uu = du.ParameterAt((double)i / n), vv = dv.ParameterAt((double)j / n);
                    double dd = Dist(NurbsEval.SurfacePoint(s, uu, vv), x);
                    if (dd < bd) { bd = dd; u = uu; v = vv; }
                }
            return ClosestOnSurface(s, x, ref u, ref v);
        }

        // ------------------------------------------------------------------ area

        static readonly double[] GLx, GLw;   // 20-point Gauss-Legendre on [-1, 1]
        static readonly double[] KX = {
            0.991455371120812639206854697526329, 0.949107912342758524526189684047851, 0.864864423359769072789712788640926,
            0.741531185599394439863864773280788, 0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
            0.207784955007898467600689403773245, 0.0 };
        static readonly double[] KW = {
            0.022935322010529224963732008058970, 0.063092092629978553290700663189204, 0.104790010322250183839876322541518,
            0.140653259715525918745189590510238, 0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
            0.204432940075298892414161999234649, 0.209482141084727828012999174891714 };
        static readonly double[] GW7 = { 0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
            0.381830050505118944950369775488975, 0.417959183673469387755102040816327 };

        static SplitGeom()
        {
            int n = 20; GLx = new double[n]; GLw = new double[n];
            for (int i = 0; i < n; i++)
            {
                double x = Math.Cos(Math.PI * (i + 0.75) / (n + 0.5));
                for (int it = 0; it < 100; it++)
                {
                    double p0 = 1, p1 = x;
                    for (int k = 2; k <= n; k++) { double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
                    double dp = n * (x * p1 - p0) / (x * x - 1);
                    double dx = p1 / dp; x -= dx;
                    if (Math.Abs(dx) < 1e-16) { GLw[i] = 2 / ((1 - x * x) * dp * dp); break; }
                    GLw[i] = 2 / ((1 - x * x) * dp * dp);
                }
                GLx[i] = x;
            }
        }

        static double Jac(NurbsSurface s, double u, double v)
        {
            var d = NurbsEval.SurfaceDerivatives(s, u, v, 1);
            return NurbsEval.Cross(d[1, 0], d[0, 1]).Length;
        }

        /// <summary>G(u, v) = integral from DomainV.T0 to v of |Su x Sv|(u, s) ds, Gauss-Legendre per V knot span.</summary>
        static double G(NurbsSurface s, double[] vBreaks, double u, double v)
        {
            double sum = 0;
            for (int k = 0; k + 1 < vBreaks.Length && vBreaks[k] < v; k++)
            {
                double a = vBreaks[k], b = Math.Min(v, vBreaks[k + 1]);
                double h = 0.5 * (b - a), m = 0.5 * (a + b);
                for (int i = 0; i < GLx.Length; i++) sum += GLw[i] * h * Jac(s, u, m + h * GLx[i]);
            }
            return sum;
        }

        static double[] DistinctKnots(double[] k, int p, int n)
        {
            var r = new List<double> { k[p] };
            for (int i = p + 1; i <= n; i++) if (k[i] > r[r.Count - 1]) r.Add(k[i]);
            return r.ToArray();
        }

        /// <summary>Signed area contribution of one trim: integral of -G(c(t)) u'(t) dt (Green's theorem).</summary>
        public static double TrimAreaIntegral(NurbsSurface s, NurbsCurve c2, double t0, double t1)
        {
            var vb = DistinctKnots(s.KnotsV, s.DegreeV, s.CvCountV);
            double F(double t)
            {
                var d = NurbsEval.CurveDerivatives(c2, t, 1);
                if (d[1].X == 0) return 0;
                double u = d[0].X, v = d[0].Y;
                if (s.IsClosedU) u = Wrap(u, s.DomainU); else u = Math.Max(s.DomainU.T0, Math.Min(s.DomainU.T1, u));
                v = Math.Max(s.DomainV.T0, Math.Min(s.DomainV.T1, v));
                return -G(s, vb, u, v) * d[1].X;
            }
            double total = 0;
            var br = Breaks(c2, t0, t1);
            // Absolute floor scaled to the face (|Su x Sv| at the domain centre times domain area) so a
            // discontinuous integrand (seam wrap, pole) cannot drive the recursion to 2^depth leaves.
            double scale = Jac(s, s.DomainU.ParameterAt(0.5), s.DomainV.ParameterAt(0.5)) * s.DomainU.Length * s.DomainV.Length;
            double absTol = 1e-12 * Math.Max(scale, 1e-300);
            for (int i = 0; i + 1 < br.Count; i++) total += Adapt(F, br[i], br[i + 1], 0, absTol);
            return total;
        }

        static double Adapt(Func<double, double> f, double a, double b, int depth, double absTol)
        {
            double h = 0.5 * (b - a), m = 0.5 * (a + b);
            double fc = f(m);
            double k = KW[7] * fc, g = GW7[3] * fc, ab = KW[7] * Math.Abs(fc);
            for (int i = 0; i < 7; i++)
            {
                double f1 = f(m - h * KX[i]), f2 = f(m + h * KX[i]);
                k += KW[i] * (f1 + f2);
                ab += KW[i] * (Math.Abs(f1) + Math.Abs(f2));
                if (i % 2 == 1) g += GW7[(i - 1) / 2] * (f1 + f2);
            }
            k *= h; g *= h; ab *= Math.Abs(h);
            // Limitation: depth cap 14 (<= 16k leaves); a smooth trim converges far earlier. Upgrade: split at seam/pole parameters up front.
            if (depth >= 14 || Math.Abs(k - g) <= Math.Max(1e-13 * ab, absTol)) return k;
            return Adapt(f, a, m, depth + 1, 0.5 * absTol) + Adapt(f, m, b, depth + 1, 0.5 * absTol);
        }

        public static double FaceArea(KBrep b, int fi)
        {
            var f = b.Faces[fi]; var s = b.Surfaces[f.Surface];
            double a = 0;
            foreach (var li in f.Loops)
                foreach (var ti in b.Loops[li].Trims)
                {
                    var t = b.Trims[ti];
                    a += TrimAreaIntegral(s, b.Curves2[t.Curve2], t.Domain.T0, t.Domain.T1);
                }
            return a;
        }

        public static double BrepArea(KBrep b)
        {
            double a = 0;
            for (int i = 0; i < b.Faces.Count; i++) a += FaceArea(b, i);
            return a;
        }

        // ------------------------------------------------------------------ 2D polygon helpers

        public static double SignedArea(List<V3> poly)
        {
            double a = 0;
            for (int i = 0; i < poly.Count; i++)
            {
                var p = poly[i]; var q = poly[(i + 1) % poly.Count];
                a += p.X * q.Y - q.X * p.Y;
            }
            return 0.5 * a;
        }

        /// <summary>Winding number of a closed polygon around p (non-zero = inside).</summary>
        public static int Winding(List<V3> poly, V3 p)
        {
            int w = 0;
            for (int i = 0; i < poly.Count; i++)
            {
                var a = poly[i]; var b = poly[(i + 1) % poly.Count];
                double cr = (b.X - a.X) * (p.Y - a.Y) - (p.X - a.X) * (b.Y - a.Y);
                if (a.Y <= p.Y) { if (b.Y > p.Y && cr > 0) w++; }
                else if (b.Y <= p.Y && cr < 0) w--;
            }
            return w;
        }
    }
}
