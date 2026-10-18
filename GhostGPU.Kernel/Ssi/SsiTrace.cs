using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    /// <summary>A traced polyline on the intersection: unwrapped params (uA,vA,uB,vB) and 3D points.</summary>
    internal sealed class SsiTrace
    {
        public List<double[]> X = new List<double[]>();
        public List<V3> P = new List<V3>();
        public bool Closed;
    }

    /// <summary>A trimmed, seam-split branch piece ready to fit (params wrapped into the surface domains).</summary>
    internal sealed class SsiPiece
    {
        public int FaceA, FaceB;
        public List<double[]> X = new List<double[]>();
        public List<V3> P = new List<V3>();
        public bool Closed;
        public double Length;
    }

    /// <summary>
    /// Marching for one face pair: tangent T = N_A x N_B, curvature-limited predictor h &lt;= sqrt(8 tolChord / kappa),
    /// pinned-coordinate corrector (3 equations S_A - S_B = 0 plus the param with the largest |dx/ds| fixed), step
    /// halving, exact domain-boundary exits (boundary param pinned), loop closure, seam/trim cutting.
    /// Runs on the host; face pairs are traced in parallel (see Ssi.Intersect).
    /// </summary>
    internal sealed class SsiTracer
    {
        readonly SsiFace _a, _b;
        readonly double _tol, _hmax, _hmin, _tolChord;
        readonly double[] _lo = new double[4], _hi = new double[4];
        readonly bool[] _per = new bool[4];
        public readonly List<SsiTrace> Traces = new List<SsiTrace>();
        public readonly List<string> Warnings = new List<string>();
        readonly Dictionary<long, List<int>> _grid = new Dictionary<long, List<int>>();
        readonly List<V3> _segA = new List<V3>(), _segB = new List<V3>();
        readonly double _cell;
        const int MaxSteps = 200000;

        public SsiTracer(SsiFace a, SsiFace b, double tol, double hmax)
        {
            _a = a; _b = b; _tol = tol;
            _hmax = hmax; _hmin = tol * 1e-3; _tolChord = 0.5 * tol;
            _cell = hmax * 2;
            _lo[0] = a.U0; _hi[0] = a.U1; _per[0] = a.PerU;
            _lo[1] = a.V0; _hi[1] = a.V1; _per[1] = a.PerV;
            _lo[2] = b.U0; _hi[2] = b.U1; _per[2] = b.PerU;
            _lo[3] = b.V0; _hi[3] = b.V1; _per[3] = b.PerV;
        }

        // ---------------- evaluation ----------------

        V3[,] EvA(double[] x, int order) => NurbsEval.SurfaceDerivatives(_a.Srf, _a.WrapU(x[0]), _a.WrapV(x[1]), order);
        V3[,] EvB(double[] x, int order) => NurbsEval.SurfaceDerivatives(_b.Srf, _b.WrapU(x[2]), _b.WrapV(x[3]), order);

        /// <summary>Pinned-coordinate Newton: solve S_A - S_B = 0 for the three free params, x[pin] fixed.</summary>
        public bool Correct(double[] x, int pin, out V3 p)
        {
            p = default;
            double best = double.MaxValue;
            for (int it = 0; it < 16; it++)
            {
                var da = EvA(x, 1); var db = EvB(x, 1);
                V3 r = da[0, 0] - db[0, 0];
                double rn = r.Length;
                p = (da[0, 0] + db[0, 0]) * 0.5;
                if (rn <= 1e-7 * _tol) return true;
                if (it > 6 && rn > 0.5 * best) break;   // not converging quadratically
                best = Math.Min(best, rn);
                V3[] col = { da[1, 0], da[0, 1], db[1, 0] * -1.0, db[0, 1] * -1.0 };
                int[] free = new int[3]; int nf = 0;
                for (int i = 0; i < 4; i++) if (i != pin) free[nf++] = i;
                if (!Solve3(col[free[0]], col[free[1]], col[free[2]], r * -1.0, out double d0, out double d1, out double d2)) return false;
                double[] d = { d0, d1, d2 };
                double scale = 1.0;
                for (int k = 0; k < 3; k++)
                {
                    double lim = 0.1 * (_hi[free[k]] - _lo[free[k]]);
                    if (Math.Abs(d[k]) * scale > lim) scale = lim / Math.Abs(d[k]);
                }
                for (int k = 0; k < 3; k++) x[free[k]] += d[k] * scale;
            }
            var ea = EvA(x, 0); var eb = EvB(x, 0);
            p = (ea[0, 0] + eb[0, 0]) * 0.5;
            return (ea[0, 0] - eb[0, 0]).Length <= 1e-4 * _tol;
        }

        static bool Solve3(V3 c0, V3 c1, V3 c2, V3 rhs, out double x0, out double x1, out double x2)
        {
            // Cramer with scaled singularity test
            double det = V3.Dot(c0, NurbsEval.Cross(c1, c2));
            double sc = c0.Length * c1.Length * c2.Length;
            x0 = x1 = x2 = 0;
            if (!(Math.Abs(det) > 1e-14 * sc) || sc == 0) return false;
            x0 = V3.Dot(rhs, NurbsEval.Cross(c1, c2)) / det;
            x1 = V3.Dot(c0, NurbsEval.Cross(rhs, c2)) / det;
            x2 = V3.Dot(c0, NurbsEval.Cross(c1, rhs)) / det;
            return true;
        }

        /// <summary>Min-norm Gauss-Newton to the nearest intersection point (seed polishing).</summary>
        public double Polish(double[] x)
        {
            double rn = double.MaxValue;
            for (int it = 0; it < 30; it++)
            {
                var da = EvA(x, 1); var db = EvB(x, 1);
                V3 r = da[0, 0] - db[0, 0];
                rn = r.Length;
                if (rn <= 1e-8 * _tol) break;
                V3[] c = { da[1, 0], da[0, 1], db[1, 0] * -1.0, db[0, 1] * -1.0 };
                // M = sum c c^T (3x3), y = M^-1 r, dx = -c.y
                double m00 = 0, m01 = 0, m02 = 0, m11 = 0, m12 = 0, m22 = 0;
                foreach (var v in c) { m00 += v.X * v.X; m01 += v.X * v.Y; m02 += v.X * v.Z; m11 += v.Y * v.Y; m12 += v.Y * v.Z; m22 += v.Z * v.Z; }
                if (!Solve3(new V3(m00, m01, m02), new V3(m01, m11, m12), new V3(m02, m12, m22), r, out double y0, out double y1, out double y2))
                    break;
                var y = new V3(y0, y1, y2);
                double scale = 1.0;
                double[] d = new double[4];
                for (int i = 0; i < 4; i++)
                {
                    d[i] = -V3.Dot(c[i], y);
                    double lim = 0.1 * (_hi[i] - _lo[i]);
                    if (Math.Abs(d[i]) * scale > lim) scale = lim / Math.Abs(d[i]);
                }
                for (int i = 0; i < 4; i++)
                {
                    x[i] += d[i] * scale;
                    if (!_per[i]) x[i] = Math.Max(_lo[i], Math.Min(_hi[i], x[i]));
                }
            }
            return rn;
        }

        /// <summary>Unit tangent, param derivatives d(x)/ds and curvature of the intersection curve at x.</summary>
        bool Tangent(double[] x, out V3 t, out double[] dx, out double kappa, out V3 p)
        {
            t = default; dx = null; kappa = 0;
            var da = EvA(x, 2); var db = EvB(x, 2);
            p = (da[0, 0] + db[0, 0]) * 0.5;
            V3 na = NurbsEval.Cross(da[1, 0], da[0, 1]), nb = NurbsEval.Cross(db[1, 0], db[0, 1]);
            double la = na.Length, lb = nb.Length;
            if (la == 0 || lb == 0) return false;
            na = na * (1.0 / la); nb = nb * (1.0 / lb);
            V3 c = NurbsEval.Cross(na, nb);
            double s = c.Length;
            if (s < 1e-9) return false;
            t = c * (1.0 / s);
            if (!ParamDir(da, t, out double dua, out double dva) || !ParamDir(db, t, out double dub, out double dvb)) return false;
            dx = new[] { dua, dva, dub, dvb };
            double ka = V3.Dot(da[2, 0] * (dua * dua) + da[1, 1] * (2 * dua * dva) + da[0, 2] * (dva * dva), na);
            double kb = V3.Dot(db[2, 0] * (dub * dub) + db[1, 1] * (2 * dub * dvb) + db[0, 2] * (dvb * dvb), nb);
            double cs = V3.Dot(na, nb), s2 = s * s;
            double ca = (ka - cs * kb) / s2, cb = (kb - cs * ka) / s2;
            kappa = (na * ca + nb * cb).Length;
            return true;
        }

        static bool ParamDir(V3[,] d, V3 t, out double du, out double dv)
        {
            V3 su = d[1, 0], sv = d[0, 1];
            double e = V3.Dot(su, su), f = V3.Dot(su, sv), g = V3.Dot(sv, sv);
            double det = e * g - f * f;
            du = dv = 0;
            if (!(det > 1e-20 * (e * g)) || det == 0) return false;
            double a = V3.Dot(su, t), b = V3.Dot(sv, t);
            du = (g * a - f * b) / det; dv = (e * b - f * a) / det;
            return true;
        }

        // ---------------- marching ----------------

        static double[] Axpy(double[] x, double h, double[] d) => new[] { x[0] + h * d[0], x[1] + h * d[1], x[2] + h * d[2], x[3] + h * d[3] };
        static double[] Lerp(double[] a, double[] b, double t) => new[] { a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]), a[3] + t * (b[3] - a[3]) };
        static double[] Neg(double[] d) => new[] { -d[0], -d[1], -d[2], -d[3] };

        static int ArgMaxAbs(double[] d)
        {
            int k = 0;
            for (int i = 1; i < 4; i++) if (Math.Abs(d[i]) > Math.Abs(d[k])) k = i;
            return k;
        }

        double DomEps(int i) => 1e-9 * (_hi[i] - _lo[i]);

        /// <summary>First non-periodic param leaving its domain along a->b: returns fraction and index (or -1).</summary>
        int Exit(double[] a, double[] b, out double lam, out double bound)
        {
            int idx = -1; lam = 2; bound = 0;
            for (int i = 0; i < 4; i++)
            {
                if (_per[i]) continue;
                double l = 2, bd = 0;
                if (b[i] > _hi[i] + DomEps(i) && b[i] != a[i]) { bd = _hi[i]; l = (bd - a[i]) / (b[i] - a[i]); }
                else if (b[i] < _lo[i] - DomEps(i) && b[i] != a[i]) { bd = _lo[i]; l = (bd - a[i]) / (b[i] - a[i]); }
                if (l < lam) { lam = Math.Max(0, l); idx = i; bound = bd; }
            }
            return idx;
        }

        bool InDomain(double[] x)
        {
            for (int i = 0; i < 4; i++)
                if (!_per[i] && (x[i] < _lo[i] - 1e3 * DomEps(i) || x[i] > _hi[i] + 1e3 * DomEps(i))) return false;
            return true;
        }

        void ClampDomain(double[] x)
        {
            for (int i = 0; i < 4; i++) if (!_per[i]) x[i] = Math.Max(_lo[i], Math.Min(_hi[i], x[i]));
        }

        enum End { None, Closed, Boundary, Singular, Merged, StepLimit }

        public bool NearTrace(V3 p, double dist, V3 dir, double minAbsDot, out int segIndex)
        {
            segIndex = -1;
            long cx = (long)Math.Floor(p.X / _cell), cy = (long)Math.Floor(p.Y / _cell), cz = (long)Math.Floor(p.Z / _cell);
            for (long i = cx - 1; i <= cx + 1; i++)
                for (long j = cy - 1; j <= cy + 1; j++)
                    for (long k = cz - 1; k <= cz + 1; k++)
                    {
                        if (!_grid.TryGetValue(Key(i, j, k), out var list)) continue;
                        foreach (int s in list)
                        {
                            V3 a = _segA[s], b = _segB[s], e = b - a;
                            double l2 = V3.Dot(e, e);
                            double t = l2 > 0 ? Math.Max(0, Math.Min(1, V3.Dot(p - a, e) / l2)) : 0;
                            if ((p - (a + e * t)).Length > dist) continue;
                            if (minAbsDot > 0 && l2 > 0 && Math.Abs(V3.Dot(e, dir)) < minAbsDot * Math.Sqrt(l2)) continue;
                            segIndex = s; return true;
                        }
                    }
            return false;
        }

        static long Key(long i, long j, long k) => (i * 73856093L) ^ (j * 19349663L) ^ (k * 83492791L);

        void AddToGrid(SsiTrace tr)
        {
            for (int s = 0; s + 1 < tr.P.Count; s++)
            {
                int id = _segA.Count;
                _segA.Add(tr.P[s]); _segB.Add(tr.P[s + 1]);
                var cells = new HashSet<long>();
                foreach (var q in new[] { tr.P[s], tr.P[s + 1], (tr.P[s] + tr.P[s + 1]) * 0.5 })
                    cells.Add(Key((long)Math.Floor(q.X / _cell), (long)Math.Floor(q.Y / _cell), (long)Math.Floor(q.Z / _cell)));
                foreach (long c in cells)
                {
                    if (!_grid.TryGetValue(c, out var l)) _grid[c] = l = new List<int>();
                    l.Add(id);
                }
            }
        }

        /// <summary>Traces the branch through seed x (already on the curve). Returns false if skipped.</summary>
        public bool TraceSeed(double[] x0)
        {
            double res = Polish(x0);
            if (res > 1e-4 * _tol) return false;
            if (!Tangent(x0, out V3 t0, out _, out _, out V3 p0)) return false;
            if (NearTrace(p0, 2 * _tol, default, 0, out _)) return false;

            var fx = new List<double[]> { x0 }; var fp = new List<V3> { p0 };
            End ef = March(fx, fp, t0, true);
            var tr = new SsiTrace();
            if (ef == End.Closed) { tr.X = fx; tr.P = fp; tr.Closed = true; }
            else
            {
                var bx = new List<double[]> { x0 }; var bp = new List<V3> { p0 };
                End eb = March(bx, bp, t0 * -1.0, false);
                for (int i = bx.Count - 1; i >= 1; i--) { tr.X.Add(bx[i]); tr.P.Add(bp[i]); }
                tr.X.AddRange(fx); tr.P.AddRange(fp);
                if (ef == End.Singular || eb == End.Singular)
                    Warnings.Add($"faces A{_a.Index}/B{_b.Index}: branch ended at a tangential/singular point near {Fmt(ef == End.Singular ? fp[fp.Count - 1] : bp[bp.Count - 1])}");
                if (ef == End.StepLimit || eb == End.StepLimit)
                    Warnings.Add($"faces A{_a.Index}/B{_b.Index}: marching step limit reached");
            }
            if (tr.P.Count < 2) return false;
            Traces.Add(tr);
            AddToGrid(tr);
            return true;
        }

        static string Fmt(V3 p) => $"({p.X:G6}, {p.Y:G6}, {p.Z:G6})";

        End March(List<double[]> xs, List<V3> ps, V3 dir, bool checkClose)
        {
            double[] x = xs[0]; V3 p = ps[0];
            double[] xStart = x; V3 pStart = p;
            if (!Tangent(x, out V3 t, out double[] dx, out double kap, out _)) return End.Singular;
            if (V3.Dot(t, dir) < 0) { t = t * -1.0; dx = Neg(dx); }
            double hLast = _hmax;
            for (int step = 0; step < MaxSteps; step++)
            {
                double hk = kap > 1e-12 ? Math.Sqrt(8 * _tolChord / kap) : _hmax;
                double h = Math.Min(Math.Min(hk, _hmax), 2 * hLast);
                double[] xn = null; V3 pn = default, tn = default; double[] dxn = null; double kn = 0;
                bool boundary = false;
                while (true)
                {
                    double[] xp = Axpy(x, h, dx);
                    int bi = Exit(x, xp, out double lam, out double bound);
                    if (bi >= 0)
                    {
                        var xb = Lerp(x, xp, lam); xb[bi] = bound;
                        if (Correct(xb, bi, out V3 pb) && InDomain(xb) && (pb - p).Length < 1.5 * h)
                        {
                            ClampDomain(xb);
                            xs.Add(xb); ps.Add(pb);
                            return End.Boundary;
                        }
                    }
                    else
                    {
                        int pin = ArgMaxAbs(dx);
                        if (Correct(xp, pin, out V3 pc))
                        {
                            double d = (pc - p).Length;
                            if (d < 1.5 * h && d > 0.3 * h && Tangent(xp, out V3 tc, out double[] dxc, out double kc, out _))
                            {
                                if (V3.Dot(tc, t) < 0) { tc = tc * -1.0; dxc = Neg(dxc); }
                                if (V3.Dot(tc, t) > 0.85 && ChordOk(x, p, xp, pc))
                                {
                                    int bo = Exit(x, xp, out double lam2, out double bound2);
                                    if (bo >= 0)
                                    {
                                        var xb = Lerp(x, xp, lam2); xb[bo] = bound2;
                                        if (Correct(xb, bo, out V3 pb) && InDomain(xb))
                                        {
                                            ClampDomain(xb);
                                            xs.Add(xb); ps.Add(pb);
                                            return End.Boundary;
                                        }
                                    }
                                    else { xn = xp; pn = pc; tn = tc; dxn = dxc; kn = kc; break; }
                                }
                            }
                        }
                    }
                    h *= 0.5;
                    if (h < _hmin)
                    {
                        if (TryJump(x, p, t, dx, hLast, out xn, out pn, out tn, out dxn, out kn))
                        {
                            Warnings.Add($"faces A{_a.Index}/B{_b.Index}: stepped across a near-singular point at {Fmt(p)}");
                            h = (pn - p).Length;
                            break;
                        }
                        return IsOnBoundary(x) ? End.Boundary : End.Singular;
                    }
                }
                if (boundary) return End.Boundary;
                hLast = h;
                // loop closure
                if (checkClose && ps.Count > 3)
                {
                    double d0 = (pStart - pn).Length;
                    double hNext = Math.Min(_hmax, kn > 1e-12 ? Math.Sqrt(8 * _tolChord / kn) : _hmax);
                    if ((d0 < _tol || (d0 <= 1.05 * Math.Max(h, hNext) && V3.Dot(tn, pStart - pn) > 0.5 * d0))
                        && SameParams(xn, xStart, out double[] shift))
                    {
                        if (d0 >= _tol) { xs.Add(xn); ps.Add(pn); }
                        xs.Add(new[] { xStart[0] + shift[0], xStart[1] + shift[1], xStart[2] + shift[2], xStart[3] + shift[3] });
                        ps.Add(pStart);
                        return End.Closed;
                    }
                }
                // running into an earlier branch of this face pair (duplicate tracing)
                if (NearTrace(pn, 2 * _tol, tn, 0.9, out int seg))
                {
                    xs.Add(xn); ps.Add(pn);
                    return End.Merged;
                }
                xs.Add(xn); ps.Add(pn);
                x = xn; p = pn; t = tn; dx = dxn; kap = kn;
            }
            return End.StepLimit;
        }

        /// <summary>Chord a-b stays within tolChord of the exact curve at its midpoint (keeps the polyline a tight proxy).</summary>
        bool ChordOk(double[] xa, V3 pa, double[] xb, V3 pb)
        {
            if (!Midpoint(xa, xb, out _, out V3 pm)) return false;
            return (pm - (pa + pb) * 0.5).Length <= _tolChord;
        }

        bool IsOnBoundary(double[] x)
        {
            for (int i = 0; i < 4; i++)
                if (!_per[i] && (Math.Abs(x[i] - _lo[i]) < 1e3 * DomEps(i) || Math.Abs(x[i] - _hi[i]) < 1e3 * DomEps(i))) return true;
            return false;
        }

        bool TryJump(double[] x, V3 p, V3 t, double[] dx, double hLast, out double[] xn, out V3 pn, out V3 tn, out double[] dxn, out double kn)
        {
            xn = null; pn = default; tn = default; dxn = null; kn = 0;
            foreach (double f in new[] { 4.0, 16.0, 64.0 })
            {
                double hj = Math.Min(_hmax, Math.Max(f * hLast, f * 10 * _tol));
                double[] xp = Axpy(x, hj, dx);
                if (Exit(x, xp, out _, out _) >= 0) continue;
                if (!Correct(xp, ArgMaxAbs(dx), out V3 pc)) continue;
                double d = (pc - p).Length;
                if (d > 1.5 * hj || d < 0.3 * hj) continue;
                if (!Tangent(xp, out V3 tc, out double[] dxc, out double kc, out _)) continue;
                if (V3.Dot(tc, t) < 0) { tc = tc * -1.0; dxc = Neg(dxc); }
                if (V3.Dot(tc, t) < 0.5) continue;
                xn = xp; pn = pc; tn = tc; dxn = dxc; kn = kc;
                return true;
            }
            return false;
        }

        bool SameParams(double[] x, double[] x0, out double[] shift)
        {
            shift = new double[4];
            for (int i = 0; i < 4; i++)
            {
                double d = x[i] - x0[i];
                if (_per[i]) { double per = _hi[i] - _lo[i]; double k = Math.Round(d / per); shift[i] = k * per; d -= k * per; }
                if (Math.Abs(d) > 0.05 * (_hi[i] - _lo[i])) return false;
            }
            return true;
        }

        // ---------------- cutting at seams and trims ----------------

        bool InsideBoth(double[] x)
        {
            var da = EvA(x, 1); var db = EvB(x, 1);
            return _a.Inside(x[0], x[1], da[1, 0], da[0, 1], _tol) && _b.Inside(x[2], x[3], db[1, 0], db[0, 1], _tol);
        }

        int Cell(int i, double v) => (int)Math.Floor((v - _lo[i]) / (_hi[i] - _lo[i]));

        /// <summary>Splits a trace at seam crossings and trim boundaries; returns the pieces inside both faces.</summary>
        public List<SsiPiece> Cut(SsiTrace tr)
        {
            // 1. seam cut nodes
            var xs = new List<double[]>(); var ps = new List<V3>(); var cut = new List<bool>();
            xs.Add(tr.X[0]); ps.Add(tr.P[0]); cut.Add(false);
            for (int s = 0; s + 1 < tr.X.Count; s++)
            {
                double[] a = tr.X[s], b = tr.X[s + 1];
                while (true)
                {
                    int best = -1; double bl = 2, bv = 0;
                    for (int i = 0; i < 4; i++)
                    {
                        if (!_per[i]) continue;
                        int ca = Cell(i, a[i]), cb = Cell(i, b[i]);
                        if (ca == cb) continue;
                        double seam = _lo[i] + Math.Max(ca, cb) * (_hi[i] - _lo[i]);
                        if (Math.Abs(a[i] - seam) < 1e-12 * (_hi[i] - _lo[i])) continue; // a already on it
                        double l = (seam - a[i]) / (b[i] - a[i]);
                        if (l > 0 && l < bl) { bl = l; best = i; bv = seam; }
                    }
                    if (best < 0) break;
                    var xc = Lerp(a, b, bl); xc[best] = bv;
                    if (!Correct(xc, best, out V3 pc)) { var e = EvA(xc, 0); pc = e[0, 0]; }
                    xs.Add(xc); ps.Add(pc); cut.Add(true);
                    a = xc;
                }
                xs.Add(b); ps.Add(tr.P[s + 1]); cut.Add(false);
            }

            // 2. trim transitions
            var ins = new List<bool>();
            foreach (var x in xs) ins.Add(InsideBoth(x));
            var X2 = new List<double[]>(); var P2 = new List<V3>(); var C2 = new List<bool>(); var I2 = new List<bool>();
            for (int i = 0; i < xs.Count; i++)
            {
                if (i > 0 && ins[i] != ins[i - 1])
                {
                    double[] lo = xs[i - 1], hi = xs[i]; bool inLo = ins[i - 1];
                    int pin = ArgMaxAbs(new[] { hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], hi[3] - lo[3] });
                    double[] good = inLo ? lo : hi; V3 pg = inLo ? ps[i - 1] : ps[i];
                    for (int it = 0; it < 60; it++)
                    {
                        var xm = Lerp(lo, hi, 0.5);
                        if (!Correct(xm, pin, out V3 pm)) xm = Lerp(lo, hi, 0.5);
                        bool im = InsideBoth(xm);
                        if (im == inLo) lo = xm; else hi = xm;
                        if (im) { good = xm; pg = pm; }
                        if (Math.Abs(hi[pin] - lo[pin]) < 1e-13 * (1 + Math.Abs(hi[pin]))) break;
                    }
                    X2.Add(good); P2.Add(pg); C2.Add(true); I2.Add(true);
                }
                X2.Add(xs[i]); P2.Add(ps[i]); C2.Add(cut[i]); I2.Add(ins[i]);
            }

            // 3. rotate closed traces to start at a cut
            bool closed = tr.Closed;
            if (closed)
            {
                int first = C2.IndexOf(true);
                if (first > 0)
                {
                    int n = X2.Count - 1; // last == first point + shift
                    var shift = new double[4];
                    for (int i = 0; i < 4; i++) shift[i] = X2[n][i] - X2[0][i];
                    var RX = new List<double[]>(); var RP = new List<V3>(); var RC = new List<bool>(); var RI = new List<bool>();
                    for (int i = first; i <= n; i++) { RX.Add(X2[i]); RP.Add(P2[i]); RC.Add(C2[i]); RI.Add(I2[i]); }
                    for (int i = 1; i <= first; i++)
                    {
                        RX.Add(new[] { X2[i][0] + shift[0], X2[i][1] + shift[1], X2[i][2] + shift[2], X2[i][3] + shift[3] });
                        RP.Add(P2[i]); RC.Add(C2[i]); RI.Add(I2[i]);
                    }
                    RC[RC.Count - 1] = true;
                    X2 = RX; P2 = RP; C2 = RC; I2 = RI;
                }
            }
            if (!closed) { C2[0] = true; C2[C2.Count - 1] = true; }
            else if (C2.IndexOf(true) < 0) { C2[0] = true; C2[C2.Count - 1] = true; }
            int cuts = 0; for (int i = 0; i < C2.Count - 1; i++) if (C2[i]) cuts++;

            // 4. pieces between cut nodes
            var pieces = new List<SsiPiece>();
            int startIdx = 0;
            for (int i = 1; i < X2.Count; i++)
            {
                if (!C2[i]) continue;
                var pc = new SsiPiece { FaceA = _a.Index, FaceB = _b.Index };
                for (int k = startIdx; k <= i; k++) { pc.X.Add(X2[k]); pc.P.Add(P2[k]); }
                startIdx = i;
                // inside? decide on an interior node, else on a corrected midpoint
                bool inside;
                double[] mid;
                if (pc.X.Count > 2) { int m = pc.X.Count / 2; inside = I2[i - (pc.X.Count - 1) + m]; mid = pc.X[m]; }
                else
                {
                    mid = Lerp(pc.X[0], pc.X[1], 0.5);
                    int pin = ArgMaxAbs(new[] { pc.X[1][0] - pc.X[0][0], pc.X[1][1] - pc.X[0][1], pc.X[1][2] - pc.X[0][2], pc.X[1][3] - pc.X[0][3] });
                    Correct(mid, pin, out _);
                    inside = InsideBoth(mid);
                }
                if (!inside) continue;
                // wrap into one period cell
                double[] mid2 = pc.X.Count > 2 ? pc.X[pc.X.Count / 2] : Lerp(pc.X[0], pc.X[1], 0.5);
                for (int j = 0; j < 4; j++)
                {
                    if (!_per[j]) continue;
                    double sh = Cell(j, mid2[j]) * (_hi[j] - _lo[j]);
                    if (sh == 0) continue;
                    for (int k = 0; k < pc.X.Count; k++)
                    {
                        var xx = (double[])pc.X[k].Clone(); xx[j] -= sh; pc.X[k] = xx;
                    }
                }
                for (int k = 1; k < pc.P.Count; k++) pc.Length += (pc.P[k] - pc.P[k - 1]).Length;
                if (pc.Length < _tol) continue;
                pc.Closed = closed && cuts <= 1 && (pc.P[0] - pc.P[pc.P.Count - 1]).Length < _tol;
                pieces.Add(pc);
            }
            return pieces;
        }

        /// <summary>Exact intersection point between two curve nodes (for densifying before the fit).</summary>
        public bool Midpoint(double[] a, double[] b, out double[] xm, out V3 pm)
        {
            xm = Lerp(a, b, 0.5);
            int pin = ArgMaxAbs(new[] { b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3] });
            return Correct(xm, pin, out pm);
        }

        public V3 PointA(double u, double v) => NurbsEval.SurfacePoint(_a.Srf, _a.WrapU(u), _a.WrapV(v));
        public V3 PointB(double u, double v) => NurbsEval.SurfacePoint(_b.Srf, _b.WrapU(u), _b.WrapV(v));
    }
}
