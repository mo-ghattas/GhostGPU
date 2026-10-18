using System;
using System.Runtime.CompilerServices;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    public static class NurbsEval
    {
        public static int FindSpan(double[] knots, int degree, int cvCount, double t)
        {
            if (knots == null) throw new ArgumentNullException(nameof(knots));
            if (degree < 1) throw new ArgumentOutOfRangeException(nameof(degree));
            if (cvCount <= degree) throw new ArgumentOutOfRangeException(nameof(cvCount));

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double Binomial(int n, int k)
        {
            if (k < 0 || k > n) return 0.0;
            if (k == 0 || k == n) return 1.0;
            if (n == 2 && k == 1) return 2.0;
            if (n == 3 && (k == 1 || k == 2)) return 3.0;
            double res = 1.0;
            for (int i = 1; i <= k; i++)
                res = res * (n - (k - i)) / i;
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static V3 Cross(V3 a, V3 b)
        {
            return new V3(
                a.Y * b.Z - a.Z * b.Y,
                a.Z * b.X - a.X * b.Z,
                a.X * b.Y - a.Y * b.X);
        }

        private static unsafe void BasisFuns(int span, double u, int p, double[] knots, double* N)
        {
            const int MaxP = 31;
            if (p > MaxP) throw new NotSupportedException($"Degrees above {MaxP} are not supported.");
            double* left = stackalloc double[p + 1];
            double* right = stackalloc double[p + 1];

            N[0] = 1.0;
            for (int j = 1; j <= p; j++)
            {
                left[j] = u - knots[span + 1 - j];
                right[j] = knots[span + j] - u;
                double saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    double temp = N[r] / (right[r + 1] + left[j - r]);
                    N[r] = saved + right[r + 1] * temp;
                    saved = left[j - r] * temp;
                }
                N[j] = saved;
            }
        }

        private static unsafe void DersBasisFuns(int span, double u, int p, int n, double[] knots, double* ders)
        {
            const int MaxP = 31;
            if (p > MaxP) throw new NotSupportedException($"Degrees above {MaxP} are not supported.");
            int stride = p + 1;
            double* ndu = stackalloc double[stride * stride];
            double* left = stackalloc double[stride];
            double* right = stackalloc double[stride];
            double* a = stackalloc double[2 * stride];

            ndu[0] = 1.0;
            for (int j = 1; j <= p; j++)
            {
                left[j] = u - knots[span + 1 - j];
                right[j] = knots[span + j] - u;
                double saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    ndu[j * stride + r] = right[r + 1] + left[j - r];
                    double temp = ndu[r * stride + (j - 1)] / ndu[j * stride + r];
                    ndu[r * stride + j] = saved + right[r + 1] * temp;
                    saved = left[j - r] * temp;
                }
                ndu[j * stride + j] = saved;
            }

            for (int j = 0; j <= p; j++)
            {
                ders[0 * stride + j] = ndu[j * stride + p];
            }

            int maxK = Math.Min(n, p);
            for (int r = 0; r <= p; r++)
            {
                int s1 = 0;
                int s2 = 1;
                a[0] = 1.0;
                for (int k = 1; k <= maxK; k++)
                {
                    double d = 0.0;
                    int rk = r - k;
                    int pk = p - k;
                    if (r >= k)
                    {
                        a[s2 * stride + 0] = a[s1 * stride + 0] / ndu[(pk + 1) * stride + rk];
                        d = a[s2 * stride + 0] * ndu[rk * stride + pk];
                    }
                    int j1 = (rk >= -1) ? 1 : -rk;
                    int j2 = (r - 1 <= pk) ? (k - 1) : (p - r);
                    for (int j = j1; j <= j2; j++)
                    {
                        a[s2 * stride + j] = (a[s1 * stride + j] - a[s1 * stride + (j - 1)]) / ndu[(pk + 1) * stride + (rk + j)];
                        d += a[s2 * stride + j] * ndu[(rk + j) * stride + pk];
                    }
                    if (r <= pk)
                    {
                        a[s2 * stride + k] = -a[s1 * stride + (k - 1)] / ndu[(pk + 1) * stride + r];
                        d += a[s2 * stride + k] * ndu[r * stride + pk];
                    }
                    ders[k * stride + r] = d;
                    int tmp = s1; s1 = s2; s2 = tmp;
                }
            }

            double fac = p;
            for (int k = 1; k <= maxK; k++)
            {
                for (int j = 0; j <= p; j++)
                {
                    ders[k * stride + j] *= fac;
                }
                fac *= (p - k);
            }

            for (int k = p + 1; k <= n; k++)
            {
                for (int j = 0; j <= p; j++)
                {
                    ders[k * stride + j] = 0.0;
                }
            }
        }

        public static unsafe V3 CurvePoint(NurbsCurve c, double t)
        {
            if (c == null) throw new ArgumentNullException(nameof(c));
            int p = c.Degree;
            int span = FindSpan(c.Knots, p, c.CvCount, t);
            double* N = stackalloc double[p + 1];
            BasisFuns(span, t, p, c.Knots, N);

            double x = 0, y = 0, z = 0, w = 0;
            int start = span - p;
            for (int j = 0; j <= p; j++)
            {
                double nj = N[j];
                int cvIdx = (start + j) * 4;
                x += nj * c.Cvs[cvIdx];
                y += nj * c.Cvs[cvIdx + 1];
                z += nj * c.Cvs[cvIdx + 2];
                w += nj * c.Cvs[cvIdx + 3];
            }

            if (c.IsRational && w != 0.0)
            {
                double invW = 1.0 / w;
                x *= invW;
                y *= invW;
                z *= invW;
            }

            return new V3(x, y, c.Dimension == 2 ? 0.0 : z);
        }

        public static unsafe V3[] CurveDerivatives(NurbsCurve c, double t, int order)
        {
            if (c == null) throw new ArgumentNullException(nameof(c));
            if (order < 0) throw new ArgumentOutOfRangeException(nameof(order));
            int p = c.Degree;
            int span = FindSpan(c.Knots, p, c.CvCount, t);

            int stride = p + 1;
            double* ders = stackalloc double[(order + 1) * stride];
            DersBasisFuns(span, t, p, order, c.Knots, ders);

            V3[] result = new V3[order + 1];
            int start = span - p;

            if (!c.IsRational)
            {
                for (int k = 0; k <= order; k++)
                {
                    double x = 0, y = 0, z = 0;
                    int kOffset = k * stride;
                    for (int j = 0; j <= p; j++)
                    {
                        double basis = ders[kOffset + j];
                        int cvIdx = (start + j) * 4;
                        x += basis * c.Cvs[cvIdx];
                        y += basis * c.Cvs[cvIdx + 1];
                        z += basis * c.Cvs[cvIdx + 2];
                    }
                    result[k] = new V3(x, y, c.Dimension == 2 ? 0.0 : z);
                }
                return result;
            }

            // Rational curve: Piegl & Tiller A4.2
            double* cw = stackalloc double[(order + 1) * 4];
            for (int k = 0; k <= order; k++)
            {
                double x = 0, y = 0, z = 0, w = 0;
                int kOffset = k * stride;
                for (int j = 0; j <= p; j++)
                {
                    double basis = ders[kOffset + j];
                    int cvIdx = (start + j) * 4;
                    x += basis * c.Cvs[cvIdx];
                    y += basis * c.Cvs[cvIdx + 1];
                    z += basis * c.Cvs[cvIdx + 2];
                    w += basis * c.Cvs[cvIdx + 3];
                }
                int idx = k * 4;
                cw[idx] = x;
                cw[idx + 1] = y;
                cw[idx + 2] = c.Dimension == 2 ? 0.0 : z;
                cw[idx + 3] = w;
            }

            double w0 = cw[3];
            double invW0 = w0 != 0.0 ? 1.0 / w0 : 1.0;

            for (int k = 0; k <= order; k++)
            {
                int kIdx = k * 4;
                V3 v = new V3(cw[kIdx], cw[kIdx + 1], cw[kIdx + 2]);
                for (int j = 1; j <= k; j++)
                {
                    double bin = Binomial(k, j);
                    double wj = cw[j * 4 + 3];
                    v = v - result[k - j] * (bin * wj);
                }
                result[k] = v * invW0;
            }

            return result;
        }

        public static unsafe V3 SurfacePoint(NurbsSurface s, double u, double v)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));
            int pu = s.DegreeU;
            int pv = s.DegreeV;
            int spanU = FindSpan(s.KnotsU, pu, s.CvCountU, u);
            int spanV = FindSpan(s.KnotsV, pv, s.CvCountV, v);

            double* nu = stackalloc double[pu + 1];
            double* nv = stackalloc double[pv + 1];
            BasisFuns(spanU, u, pu, s.KnotsU, nu);
            BasisFuns(spanV, v, pv, s.KnotsV, nv);

            double x = 0, y = 0, z = 0, w = 0;
            int startU = spanU - pu;
            int startV = spanV - pv;
            int cvCountV = s.CvCountV;

            for (int i = 0; i <= pu; i++)
            {
                double nui = nu[i];
                int rowIdx = (startU + i) * cvCountV;
                for (int j = 0; j <= pv; j++)
                {
                    double weight = nui * nv[j];
                    int cvIdx = (rowIdx + (startV + j)) * 4;
                    x += weight * s.Cvs[cvIdx];
                    y += weight * s.Cvs[cvIdx + 1];
                    z += weight * s.Cvs[cvIdx + 2];
                    w += weight * s.Cvs[cvIdx + 3];
                }
            }

            if (s.IsRational && w != 0.0)
            {
                double invW = 1.0 / w;
                x *= invW;
                y *= invW;
                z *= invW;
            }

            return new V3(x, y, z);
        }

        public static unsafe V3[,] SurfaceDerivatives(NurbsSurface s, double u, double v, int order)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));
            if (order < 0) throw new ArgumentOutOfRangeException(nameof(order));

            int pu = s.DegreeU;
            int pv = s.DegreeV;
            int spanU = FindSpan(s.KnotsU, pu, s.CvCountU, u);
            int spanV = FindSpan(s.KnotsV, pv, s.CvCountV, v);

            int strideU = pu + 1;
            int strideV = pv + 1;
            double* dersU = stackalloc double[(order + 1) * strideU];
            double* dersV = stackalloc double[(order + 1) * strideV];

            DersBasisFuns(spanU, u, pu, order, s.KnotsU, dersU);
            DersBasisFuns(spanV, v, pv, order, s.KnotsV, dersV);

            int startU = spanU - pu;
            int startV = spanV - pv;
            int cvCountV = s.CvCountV;

            V3[,] result = new V3[order + 1, order + 1];

            if (!s.IsRational)
            {
                for (int k = 0; k <= order; k++)
                {
                    int kOffset = k * strideU;
                    for (int l = 0; l <= order; l++)
                    {
                        int lOffset = l * strideV;
                        double x = 0, y = 0, z = 0;
                        for (int i = 0; i <= pu; i++)
                        {
                            double nui = dersU[kOffset + i];
                            int rowIdx = (startU + i) * cvCountV;
                            for (int j = 0; j <= pv; j++)
                            {
                                double weight = nui * dersV[lOffset + j];
                                int cvIdx = (rowIdx + (startV + j)) * 4;
                                x += weight * s.Cvs[cvIdx];
                                y += weight * s.Cvs[cvIdx + 1];
                                z += weight * s.Cvs[cvIdx + 2];
                            }
                        }
                        result[k, l] = new V3(x, y, z);
                    }
                }
                return result;
            }

            // Rational surface: Piegl & Tiller A4.4
            int derivGridSize = (order + 1) * (order + 1);
            double* aw = stackalloc double[derivGridSize * 4];

            for (int k = 0; k <= order; k++)
            {
                int kOffset = k * strideU;
                for (int l = 0; l <= order; l++)
                {
                    int lOffset = l * strideV;
                    double x = 0, y = 0, z = 0, w = 0;
                    for (int i = 0; i <= pu; i++)
                    {
                        double nui = dersU[kOffset + i];
                        int rowIdx = (startU + i) * cvCountV;
                        for (int j = 0; j <= pv; j++)
                        {
                            double weight = nui * dersV[lOffset + j];
                            int cvIdx = (rowIdx + (startV + j)) * 4;
                            x += weight * s.Cvs[cvIdx];
                            y += weight * s.Cvs[cvIdx + 1];
                            z += weight * s.Cvs[cvIdx + 2];
                            w += weight * s.Cvs[cvIdx + 3];
                        }
                    }
                    int idx = (k * (order + 1) + l) * 4;
                    aw[idx] = x;
                    aw[idx + 1] = y;
                    aw[idx + 2] = z;
                    aw[idx + 3] = w;
                }
            }

            double w00 = aw[3];
            double invW00 = w00 != 0.0 ? 1.0 / w00 : 1.0;

            for (int k = 0; k <= order; k++)
            {
                for (int l = 0; l <= order; l++)
                {
                    int idx = (k * (order + 1) + l) * 4;
                    V3 v3 = new V3(aw[idx], aw[idx + 1], aw[idx + 2]);

                    for (int j = 1; j <= l; j++)
                    {
                        double w0j = aw[(0 * (order + 1) + j) * 4 + 3];
                        v3 = v3 - result[k, l - j] * (Binomial(l, j) * w0j);
                    }

                    for (int i = 1; i <= k; i++)
                    {
                        double wi0 = aw[(i * (order + 1) + 0) * 4 + 3];
                        v3 = v3 - result[k - i, l] * (Binomial(k, i) * wi0);
                        V3 v2 = default;
                        for (int j = 1; j <= l; j++)
                        {
                            double wij = aw[(i * (order + 1) + j) * 4 + 3];
                            v2 = v2 + result[k - i, l - j] * (Binomial(l, j) * wij);
                        }
                        v3 = v3 - v2 * Binomial(k, i);
                    }

                    result[k, l] = v3 * invW00;
                }
            }

            return result;
        }

        public static V3 SurfaceNormal(NurbsSurface s, double u, double v)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));

            var ders = SurfaceDerivatives(s, u, v, 2);
            V3 su = ders[1, 0];
            V3 sv = ders[0, 1];
            V3 n = Cross(su, sv);
            double len = n.Length;

            if (len > 1e-12)
            {
                return n * (1.0 / len);
            }

            // Degenerate normal: pole handling via second-order limits
            V3 suv = ders[1, 1];
            if (su.Length <= 1e-12)
            {
                // Pole along U (e.g. at Vmin or Vmax)
                if (Math.Abs(v - s.DomainV.T0) <= 1e-8)
                    n = Cross(suv, sv);
                else if (Math.Abs(v - s.DomainV.T1) <= 1e-8)
                    n = Cross(sv, suv);
                else
                    n = Cross(suv, sv);
            }
            else if (sv.Length <= 1e-12)
            {
                // Pole along V (e.g. at Umin or Umax)
                if (Math.Abs(u - s.DomainU.T0) <= 1e-8)
                    n = Cross(su, suv);
                else if (Math.Abs(u - s.DomainU.T1) <= 1e-8)
                    n = Cross(suv, su);
                else
                    n = Cross(su, suv);
            }

            len = n.Length;
            if (len > 1e-12)
            {
                return n * (1.0 / len);
            }

            // Fallback: evaluate normal slightly perturbed into interior
            double du = (s.DomainU.T1 - s.DomainU.T0) * 1e-5;
            double dv = (s.DomainV.T1 - s.DomainV.T0) * 1e-5;
            double up = (u - s.DomainU.T0 < s.DomainU.T1 - u) ? (u + du) : (u - du);
            double vp = (v - s.DomainV.T0 < s.DomainV.T1 - v) ? (v + dv) : (v - dv);

            var dPert = SurfaceDerivatives(s, up, vp, 1);
            var nPert = Cross(dPert[1, 0], dPert[0, 1]);
            double pLen = nPert.Length;
            if (pLen > 1e-12)
            {
                return nPert * (1.0 / pLen);
            }

            return new V3(0, 0, 0);
        }
    }
}
