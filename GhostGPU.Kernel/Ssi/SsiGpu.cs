using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using GhostGPU.Runtime;
using ILGPU;
using ILGPU.Runtime;

namespace GhostGPU.Kernel
{
    /// <summary>A pair of Bezier sub-patches: patch index plus local [0,1] parameter rectangle on each side.</summary>
    public struct SsiPair
    {
        public int A, B;
        public double Au0, Au1, Av0, Av1, Bu0, Bu1, Bv0, Bv1;
    }

    /// <summary>Newton seed result in patch-local parameters.</summary>
    public struct SsiSeed
    {
        public double Sa, Ta, Sb, Tb, Residual;
    }

    // Data accessors so the same element math runs on host arrays and device views (ILGPU devirtualizes struct generics).
    public interface ISsiDoubles { double Get(int i); }
    public interface ISsiInts { int Get(int i); }
    public struct HostDoubles : ISsiDoubles { public double[] A; public double Get(int i) => A[i]; }
    public struct HostInts : ISsiInts { public int[] A; public int Get(int i) => A[i]; }
    public struct DevDoubles : ISsiDoubles { public ArrayView<double> V; public double Get(int i) => V[i]; }
    public struct DevInts : ISsiInts { public ArrayView<int> V; public int Get(int i) => V[i]; }

    /// <summary>
    /// Element math shared by host and device: Bezier sub-patch hulls via blossoming (no local arrays, so the
    /// identical code runs in ILGPU kernels and Parallel.For), and min-norm Newton seeding on Bezier patches.
    /// Patch meta: 3 ints per patch (first CV index, degreeU, degreeV); CVs homogeneous, row-major over U.
    /// </summary>
    public static class SsiElement
    {
        public const int CodeOverlap = 1, CodeSplitA = 2, CodeSplitAU = 4, CodeSplitB = 8, CodeSplitBU = 16;

        static double IPow(double x, int n) { double r = 1.0; for (int i = 0; i < n; i++) r *= x; return r; }

        static double Binom(int n, int k)
        {
            if (k < 0 || k > n) return 0.0;
            double r = 1.0;
            for (int j = 1; j <= k; j++) r = r * (n - k + j) / j;
            return r;
        }

        static double Bern(int p, int i, double s)
        {
            if (i < 0 || i > p) return 0.0;
            return Binom(p, i) * IPow(s, i) * IPow(1.0 - s, p - i);
        }

        /// <summary>Coefficient of P_k in control point Q_i of the degree-p Bezier restricted to [s0,s1] (blossom).</summary>
        static double Restrict(int p, int i, int k, double s0, double s1)
        {
            double sum = 0.0;
            int jlo = i + k - p; if (jlo < 0) jlo = 0;
            int jhi = i < k ? i : k;
            for (int j = jlo; j <= jhi; j++)
                sum += Binom(i, j) * Binom(p - i, k - j) * IPow(s1, j) * IPow(1.0 - s1, i - j)
                     * IPow(s0, k - j) * IPow(1.0 - s0, p - i - k + j);
            return sum;
        }

        /// <summary>Convex-hull box of the sub-patch [u0,u1]x[v0,v1], plus control-net lengths along u and v.</summary>
        public static void SubBox<TD, TI>(TD cvs, TI meta, int patch, double u0, double u1, double v0, double v1,
            out double x0, out double y0, out double z0, out double x1, out double y1, out double z1,
            out double lenU, out double lenV)
            where TD : struct, ISsiDoubles where TI : struct, ISsiInts
        {
            int off = meta.Get(3 * patch), p = meta.Get(3 * patch + 1), q = meta.Get(3 * patch + 2);
            x0 = y0 = z0 = double.MaxValue; x1 = y1 = z1 = -double.MaxValue;
            double c00x = 0, c00y = 0, c00z = 0, c0qx = 0, c0qy = 0, c0qz = 0;
            double cp0x = 0, cp0y = 0, cp0z = 0, cpqx = 0, cpqy = 0, cpqz = 0;
            for (int i = 0; i <= p; i++)
            {
                for (int j = 0; j <= q; j++)
                {
                    double x = 0, y = 0, z = 0, w = 0;
                    for (int k = 0; k <= p; k++)
                    {
                        double a = Restrict(p, i, k, u0, u1);
                        if (a == 0.0) continue;
                        for (int l = 0; l <= q; l++)
                        {
                            double b = a * Restrict(q, j, l, v0, v1);
                            if (b == 0.0) continue;
                            int idx = (off + k * (q + 1) + l) * 4;
                            x += b * cvs.Get(idx); y += b * cvs.Get(idx + 1); z += b * cvs.Get(idx + 2); w += b * cvs.Get(idx + 3);
                        }
                    }
                    double iw = w != 0.0 ? 1.0 / w : 1.0;
                    x *= iw; y *= iw; z *= iw;
                    if (x < x0) x0 = x; if (x > x1) x1 = x;
                    if (y < y0) y0 = y; if (y > y1) y1 = y;
                    if (z < z0) z0 = z; if (z > z1) z1 = z;
                    if (i == 0 && j == 0) { c00x = x; c00y = y; c00z = z; }
                    if (i == 0 && j == q) { c0qx = x; c0qy = y; c0qz = z; }
                    if (i == p && j == 0) { cp0x = x; cp0y = y; cp0z = z; }
                    if (i == p && j == q) { cpqx = x; cpqy = y; cpqz = z; }
                }
            }
            lenU = Len(cp0x - c00x, cp0y - c00y, cp0z - c00z) + Len(cpqx - c0qx, cpqy - c0qy, cpqz - c0qz);
            lenV = Len(c0qx - c00x, c0qy - c00y, c0qz - c00z) + Len(cpqx - cp0x, cpqy - cp0y, cpqz - cp0z);
        }

        static double Len(double x, double y, double z) => Math.Sqrt(x * x + y * y + z * z);

        /// <summary>Box-overlap test of the pair and per-side split decision (see Code* bits). 0 = disjoint.</summary>
        public static int Classify<TD, TI>(TD cvs, TI meta, SsiPair pr, double inflate, double leafSize)
            where TD : struct, ISsiDoubles where TI : struct, ISsiInts
        {
            SubBox(cvs, meta, pr.A, pr.Au0, pr.Au1, pr.Av0, pr.Av1, out double ax0, out double ay0, out double az0,
                   out double ax1, out double ay1, out double az1, out double aLu, out double aLv);
            SubBox(cvs, meta, pr.B, pr.Bu0, pr.Bu1, pr.Bv0, pr.Bv1, out double bx0, out double by0, out double bz0,
                   out double bx1, out double by1, out double bz1, out double bLu, out double bLv);
            if (ax0 > bx1 + inflate || bx0 > ax1 + inflate || ay0 > by1 + inflate || by0 > ay1 + inflate
                || az0 > bz1 + inflate || bz0 > az1 + inflate) return 0;
            int code = CodeOverlap;
            double ea = Math.Max(ax1 - ax0, Math.Max(ay1 - ay0, az1 - az0));
            double eb = Math.Max(bx1 - bx0, Math.Max(by1 - by0, bz1 - bz0));
            const double minWidth = 1e-9;
            if (ea > leafSize)
            {
                bool su = aLu >= aLv;
                if (su && pr.Au1 - pr.Au0 < minWidth) su = false;
                if (!su && pr.Av1 - pr.Av0 < minWidth) su = pr.Au1 - pr.Au0 >= minWidth;
                if (pr.Au1 - pr.Au0 >= minWidth || pr.Av1 - pr.Av0 >= minWidth)
                    code |= CodeSplitA | (su ? CodeSplitAU : 0);
            }
            if (eb > leafSize)
            {
                bool su = bLu >= bLv;
                if (su && pr.Bu1 - pr.Bu0 < minWidth) su = false;
                if (!su && pr.Bv1 - pr.Bv0 < minWidth) su = pr.Bu1 - pr.Bu0 >= minWidth;
                if (pr.Bu1 - pr.Bu0 >= minWidth || pr.Bv1 - pr.Bv0 >= minWidth)
                    code |= CodeSplitB | (su ? CodeSplitBU : 0);
            }
            return code;
        }

        /// <summary>Rational Bezier patch point and first partials at local (s,t).</summary>
        public static void EvalPatch<TD, TI>(TD cvs, TI meta, int patch, double s, double t,
            out double px, out double py, out double pz, out double sx, out double sy, out double sz,
            out double tx, out double ty, out double tz)
            where TD : struct, ISsiDoubles where TI : struct, ISsiInts
        {
            int off = meta.Get(3 * patch), p = meta.Get(3 * patch + 1), q = meta.Get(3 * patch + 2);
            double ax = 0, ay = 0, az = 0, aw = 0, bx = 0, by = 0, bz = 0, bw = 0, cx = 0, cy = 0, cz = 0, cw = 0;
            for (int i = 0; i <= p; i++)
            {
                double bi = Bern(p, i, s);
                double dbi = p * (Bern(p - 1, i - 1, s) - Bern(p - 1, i, s));
                for (int j = 0; j <= q; j++)
                {
                    double bj = Bern(q, j, t);
                    double dbj = q * (Bern(q - 1, j - 1, t) - Bern(q - 1, j, t));
                    int idx = (off + i * (q + 1) + j) * 4;
                    double X = cvs.Get(idx), Y = cvs.Get(idx + 1), Z = cvs.Get(idx + 2), W = cvs.Get(idx + 3);
                    double f = bi * bj, fs = dbi * bj, ft = bi * dbj;
                    ax += f * X; ay += f * Y; az += f * Z; aw += f * W;
                    bx += fs * X; by += fs * Y; bz += fs * Z; bw += fs * W;
                    cx += ft * X; cy += ft * Y; cz += ft * Z; cw += ft * W;
                }
            }
            double iw = 1.0 / aw;
            px = ax * iw; py = ay * iw; pz = az * iw;
            sx = (bx - px * bw) * iw; sy = (by - py * bw) * iw; sz = (bz - pz * bw) * iw;
            tx = (cx - px * cw) * iw; ty = (cy - py * cw) * iw; tz = (cz - pz * cw) * iw;
        }

        /// <summary>Min-norm Gauss-Newton (J^T (J J^T)^-1) on S_A(s,t) = S_B(s,t) from the pair's rectangle centres.</summary>
        public static SsiSeed Newton<TD, TI>(TD cvs, TI meta, SsiPair pr)
            where TD : struct, ISsiDoubles where TI : struct, ISsiInts
        {
            double sa = 0.5 * (pr.Au0 + pr.Au1), ta = 0.5 * (pr.Av0 + pr.Av1);
            double sb = 0.5 * (pr.Bu0 + pr.Bu1), tb = 0.5 * (pr.Bv0 + pr.Bv1);
            double res = double.MaxValue;
            for (int it = 0; it < 40; it++)
            {
                EvalPatch(cvs, meta, pr.A, sa, ta, out double pax, out double pay, out double paz,
                          out double asx, out double asy, out double asz, out double atx, out double aty, out double atz);
                EvalPatch(cvs, meta, pr.B, sb, tb, out double pbx, out double pby, out double pbz,
                          out double bsx, out double bsy, out double bsz, out double btx, out double bty, out double btz);
                double rx = pax - pbx, ry = pay - pby, rz = paz - pbz;
                res = Math.Sqrt(rx * rx + ry * ry + rz * rz);
                double scale = 1.0 + Math.Abs(pax) + Math.Abs(pay) + Math.Abs(paz);
                if (res < 1e-13 * scale) break;
                // columns c1=As c2=At c3=-Bs c4=-Bt ; M = sum c c^T
                double m00 = asx * asx + atx * atx + bsx * bsx + btx * btx;
                double m01 = asx * asy + atx * aty + bsx * bsy + btx * bty;
                double m02 = asx * asz + atx * atz + bsx * bsz + btx * btz;
                double m11 = asy * asy + aty * aty + bsy * bsy + bty * bty;
                double m12 = asy * asz + aty * atz + bsy * bsz + bty * btz;
                double m22 = asz * asz + atz * atz + bsz * bsz + btz * btz;
                double c00 = m11 * m22 - m12 * m12, c01 = m02 * m12 - m01 * m22, c02 = m01 * m12 - m02 * m11;
                double det = m00 * c00 + m01 * c01 + m02 * c02;
                double tr = m00 + m11 + m22;
                if (!(Math.Abs(det) > 1e-30 * tr * tr * tr)) break;
                double c11 = m00 * m22 - m02 * m02, c12 = m01 * m02 - m00 * m12, c22 = m00 * m11 - m01 * m01;
                double id = 1.0 / det;
                double yx = (c00 * rx + c01 * ry + c02 * rz) * id;
                double yy = (c01 * rx + c11 * ry + c12 * rz) * id;
                double yz = (c02 * rx + c12 * ry + c22 * rz) * id;
                double d0 = -(asx * yx + asy * yy + asz * yz);
                double d1 = -(atx * yx + aty * yy + atz * yz);
                double d2 = (bsx * yx + bsy * yy + bsz * yz);
                double d3 = (btx * yx + bty * yy + btz * yz);
                double mx = Math.Max(Math.Max(Math.Abs(d0), Math.Abs(d1)), Math.Max(Math.Abs(d2), Math.Abs(d3)));
                if (mx > 0.25) { double f = 0.25 / mx; d0 *= f; d1 *= f; d2 *= f; d3 *= f; }
                sa = Clamp01(sa + d0); ta = Clamp01(ta + d1); sb = Clamp01(sb + d2); tb = Clamp01(tb + d3);
            }
            SsiSeed r;
            r.Sa = sa; r.Ta = ta; r.Sb = sb; r.Tb = tb; r.Residual = res;
            return r;
        }

        static double Clamp01(double x) => x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
    }

    public static class SsiKernels
    {
        public static void ClassifyKernel(Index1D i, ArrayView<SsiPair> pairs, ArrayView<double> cvs, ArrayView<int> meta,
                                          double inflate, double leafSize, ArrayView<int> codes)
        {
            codes[i] = SsiElement.Classify(new DevDoubles { V = cvs }, new DevInts { V = meta }, pairs[i], inflate, leafSize);
        }

        public static void NewtonKernel(Index1D i, ArrayView<SsiPair> pairs, ArrayView<double> cvs, ArrayView<int> meta,
                                        ArrayView<SsiSeed> seeds)
        {
            seeds[i] = SsiElement.Newton(new DevDoubles { V = cvs }, new DevInts { V = meta }, pairs[i]);
        }
    }

    /// <summary>Batched drivers: device kernels when a session is given and the batch is big enough, else host Parallel.For.</summary>
    internal sealed class SsiBatch : IDisposable
    {
        // Limitation: fixed cutoff measured on RTX 3070 Ti (FP64 1/64 rate); per-element work is ~1e4-1e5 flops,
        // so the break-even is far below the runtime's generic HostWorkThreshold.
        // measured (torus patches, CUDA vs host Parallel.For): classify 512 pairs 1.5 vs 2.5 ms, 131072: 8 vs 28 ms;
        // Newton 512: 10 vs 7 ms, 2048: 8 vs 22 ms, 131072: 51 vs 163 ms.
        public const int MinDeviceBatch = 512, MinDeviceNewton = 2048;

        readonly GpuSession _s;
        readonly double[] _cvs;
        readonly int[] _meta;
        DeviceLease<double> _dCvs;
        DeviceLease<int> _dMeta;
        public int DeviceDispatches, HostDispatches;

        public SsiBatch(GpuSession s, double[] cvs, int[] meta) { _s = s; _cvs = cvs; _meta = meta; }

        bool UseDevice(int n, int min = MinDeviceBatch) => _s != null && n >= min;

        void EnsureDevice()
        {
            if (_dCvs != null) return;
            _dCvs = _s.Rent<double>(_cvs.Length);
            _dCvs.View.CopyFromCPU(_s.Accelerator.DefaultStream, _cvs);
            _dMeta = _s.Rent<int>(_meta.Length);
            _dMeta.View.CopyFromCPU(_s.Accelerator.DefaultStream, _meta);
        }

        public int[] Classify(SsiPair[] pairs, double inflate, double leafSize)
        {
            int n = pairs.Length;
            var codes = new int[n];
            if (n == 0) return codes;
            if (!UseDevice(n))
            {
                var hc = new HostDoubles { A = _cvs }; var hm = new HostInts { A = _meta };
                Parallel.For(0, n, i => codes[i] = SsiElement.Classify(hc, hm, pairs[i], inflate, leafSize));
                _s?.CountHostDispatch(); HostDispatches++;
                return codes;
            }
            EnsureDevice();
            using (var dp = _s.Rent<SsiPair>(n))
            using (var dc = _s.Rent<int>(n))
            {
                dp.View.CopyFromCPU(_s.Accelerator.DefaultStream, pairs);
                var k = _s.GetKernel<Action<Index1D, ArrayView<SsiPair>, ArrayView<double>, ArrayView<int>, double, double, ArrayView<int>>>(
                    (Action<Index1D, ArrayView<SsiPair>, ArrayView<double>, ArrayView<int>, double, double, ArrayView<int>>)SsiKernels.ClassifyKernel);
                k(new Index1D(n), dp.View, _dCvs.View, _dMeta.View, inflate, leafSize, dc.View);
                _s.CountDispatch(); DeviceDispatches++;
                dc.View.CopyToCPU(_s.Accelerator.DefaultStream, codes);
                _s.Synchronize();
            }
            return codes;
        }

        public SsiSeed[] Newton(SsiPair[] pairs)
        {
            int n = pairs.Length;
            var seeds = new SsiSeed[n];
            if (n == 0) return seeds;
            if (!UseDevice(n, MinDeviceNewton))
            {
                var hc = new HostDoubles { A = _cvs }; var hm = new HostInts { A = _meta };
                Parallel.For(0, n, i => seeds[i] = SsiElement.Newton(hc, hm, pairs[i]));
                _s?.CountHostDispatch(); HostDispatches++;
                return seeds;
            }
            EnsureDevice();
            using (var dp = _s.Rent<SsiPair>(n))
            using (var ds = _s.Rent<SsiSeed>(n))
            {
                dp.View.CopyFromCPU(_s.Accelerator.DefaultStream, pairs);
                var k = _s.GetKernel<Action<Index1D, ArrayView<SsiPair>, ArrayView<double>, ArrayView<int>, ArrayView<SsiSeed>>>(
                    (Action<Index1D, ArrayView<SsiPair>, ArrayView<double>, ArrayView<int>, ArrayView<SsiSeed>>)SsiKernels.NewtonKernel);
                k(new Index1D(n), dp.View, _dCvs.View, _dMeta.View, ds.View);
                _s.CountDispatch(); DeviceDispatches++;
                ds.View.CopyToCPU(_s.Accelerator.DefaultStream, seeds);
                _s.Synchronize();
            }
            return seeds;
        }

        /// <summary>
        /// Breadth-first subdivision of overlapping sub-patch pairs until both sides are below leafSize (or
        /// maxLevels). Returns the overlapping leaves. Each level is one batched classify (device or host).
        /// </summary>
        public List<SsiPair> Subdivide(List<SsiPair> init, double inflate, double leafSize, int maxLevels, int maxPairs,
                                       List<string> warnings)
        {
            var leaves = new List<SsiPair>();
            var cur = init.ToArray();
            for (int level = 0; cur.Length > 0; level++)
            {
                int[] codes = Classify(cur, inflate, leafSize);
                var next = new List<SsiPair>();
                for (int i = 0; i < cur.Length; i++)
                {
                    int c = codes[i];
                    if (c == 0) continue;
                    bool sa = (c & SsiElement.CodeSplitA) != 0 && level < maxLevels;
                    bool sb = (c & SsiElement.CodeSplitB) != 0 && level < maxLevels;
                    if (!sa && !sb) { leaves.Add(cur[i]); continue; }
                    var p = cur[i];
                    int na = sa ? 2 : 1, nb = sb ? 2 : 1;
                    for (int ia = 0; ia < na; ia++)
                        for (int ib = 0; ib < nb; ib++)
                        {
                            var ch = p;
                            if (sa)
                            {
                                if ((c & SsiElement.CodeSplitAU) != 0) { double m = 0.5 * (p.Au0 + p.Au1); if (ia == 0) ch.Au1 = m; else ch.Au0 = m; }
                                else { double m = 0.5 * (p.Av0 + p.Av1); if (ia == 0) ch.Av1 = m; else ch.Av0 = m; }
                            }
                            if (sb)
                            {
                                if ((c & SsiElement.CodeSplitBU) != 0) { double m = 0.5 * (p.Bu0 + p.Bu1); if (ib == 0) ch.Bu1 = m; else ch.Bu0 = m; }
                                else { double m = 0.5 * (p.Bv0 + p.Bv1); if (ib == 0) ch.Bv1 = m; else ch.Bv0 = m; }
                            }
                            next.Add(ch);
                        }
                }
                if (next.Count > maxPairs)
                {
                    warnings.Add($"SSI subdivision exceeded {maxPairs} candidate pairs at level {level}; "
                               + "surfaces are coincident or tangent over a region. Seeding from the current level.");
                    for (int i = 0; i < cur.Length; i++) if (codes[i] != 0) leaves.Add(cur[i]);
                    break;
                }
                cur = next.ToArray();
            }
            return leaves;
        }

        public void Dispose()
        {
            _dCvs?.Dispose(); _dMeta?.Dispose();
            _dCvs = null; _dMeta = null;
        }
    }
}
