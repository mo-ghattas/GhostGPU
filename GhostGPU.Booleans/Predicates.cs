using System;
using System.Numerics;
using GhostGPU.Core;

namespace GhostGPU.Booleans
{
    /// <summary>
    /// Exact orientation predicates on double inputs: Shewchuk's static floating-point filter (stage A error bounds),
    /// falling back to exact BigInteger arithmetic when the filter cannot certify the sign. The fallback is slow but
    /// rare (only near-degenerate inputs reach it). Limitation: no adaptive stages B/C; add them if profiling shows the
    /// BigInteger path is hot.
    /// </summary>
    internal static class Predicates
    {
        private const double Eps = 1.1102230246251565e-16; // 2^-53
        private const double Ccw = (3.0 + 16.0 * Eps) * Eps;
        private const double O3d = (7.0 + 56.0 * Eps) * Eps;

        /// <summary>Sign of the 2D orientation of (a, b, c): +1 counter-clockwise, -1 clockwise, 0 collinear.</summary>
        public static int Orient2D(double ax, double ay, double bx, double by, double cx, double cy)
        {
            double l = (ax - cx) * (by - cy), r = (ay - cy) * (bx - cx);
            double det = l - r;
            double bound = Ccw * (Math.Abs(l) + Math.Abs(r));
            if (det > bound) return 1;
            if (-det > bound) return -1;
            return Orient2DExact(ax, ay, bx, by, cx, cy);
        }

        /// <summary>Sign of dot(cross(b - a, c - a), d - a): +1 if d is on the side the CCW triangle (a,b,c) faces.</summary>
        public static int Side3D(V3 a, V3 b, V3 c, V3 d)
        {
            // Shewchuk orient3d(a,b,c,d) = det[a-d; b-d; c-d], positive when d is BELOW (a,b,c); Side3D is its negation.
            double adx = a.X - d.X, ady = a.Y - d.Y, adz = a.Z - d.Z;
            double bdx = b.X - d.X, bdy = b.Y - d.Y, bdz = b.Z - d.Z;
            double cdx = c.X - d.X, cdy = c.Y - d.Y, cdz = c.Z - d.Z;
            double bc = bdy * cdz, cb = bdz * cdy, ca = cdy * adz, ac = cdz * ady, ab = ady * bdz, ba = adz * bdy;
            double det = adx * (bc - cb) + bdx * (ca - ac) + cdx * (ab - ba);
            double perm = (Math.Abs(bc) + Math.Abs(cb)) * Math.Abs(adx) + (Math.Abs(ca) + Math.Abs(ac)) * Math.Abs(bdx) +
                          (Math.Abs(ab) + Math.Abs(ba)) * Math.Abs(cdx);
            double bound = O3d * perm;
            if (det > bound) return -1;
            if (-det > bound) return 1;
            return -Orient3DExact(a, b, c, d);
        }

        private static int Orient2DExact(double ax, double ay, double bx, double by, double cx, double cy)
        {
            var v = Scaled(ax, ay, bx, by, cx, cy);
            BigInteger det = (v[0] - v[4]) * (v[3] - v[5]) - (v[1] - v[5]) * (v[2] - v[4]);
            return det.Sign;
        }

        private static int Orient3DExact(V3 a, V3 b, V3 c, V3 d)
        {
            var v = Scaled(a.X, a.Y, a.Z, b.X, b.Y, b.Z, c.X, c.Y, c.Z, d.X, d.Y, d.Z);
            BigInteger adx = v[0] - v[9], ady = v[1] - v[10], adz = v[2] - v[11];
            BigInteger bdx = v[3] - v[9], bdy = v[4] - v[10], bdz = v[5] - v[11];
            BigInteger cdx = v[6] - v[9], cdy = v[7] - v[10], cdz = v[8] - v[11];
            BigInteger det = adx * (bdy * cdz - bdz * cdy) + bdx * (cdy * adz - cdz * ady) + cdx * (ady * bdz - adz * bdy);
            return det.Sign;
        }

        // Every finite double is m * 2^e with integer m; scaling all inputs by 2^-min(e) makes them exact integers.
        private static BigInteger[] Scaled(params double[] x)
        {
            var m = new long[x.Length];
            var e = new int[x.Length];
            int emin = int.MaxValue;
            for (int i = 0; i < x.Length; i++)
            {
                Decompose(x[i], out m[i], out e[i]);
                if (m[i] != 0) emin = Math.Min(emin, e[i]);
            }
            var r = new BigInteger[x.Length];
            for (int i = 0; i < x.Length; i++)
                r[i] = m[i] == 0 ? BigInteger.Zero : new BigInteger(m[i]) << (e[i] - emin);
            return r;
        }

        private static void Decompose(double d, out long mantissa, out int exponent)
        {
            long bits = BitConverter.DoubleToInt64Bits(d);
            bool neg = bits < 0;
            int exp = (int)((bits >> 52) & 0x7FF);
            long man = bits & 0xFFFFFFFFFFFFFL;
            if (exp == 0) exp++; else man |= 1L << 52;
            exponent = exp - 1075;
            mantissa = neg ? -man : man;
        }
    }
}

namespace GhostGPU.Booleans
{
    /// <summary>
    /// Hash comparers for packed keys. The default long hash is lo ^ hi, so an edge key (u &lt;&lt; 32 | v) hashes to u ^ v:
    /// nearby vertex ids collide massively and dictionaries degrade towards O(n) per lookup.
    /// </summary>
    internal sealed class KeyComparer : System.Collections.Generic.IEqualityComparer<long>,
        System.Collections.Generic.IEqualityComparer<(long, int)>, System.Collections.Generic.IEqualityComparer<(long, long, long)>
    {
        public static readonly KeyComparer Instance = new KeyComparer();

        private static int Mix(ulong x)
        {
            x ^= x >> 33; x *= 0xff51afd7ed558ccdUL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53UL; x ^= x >> 33;
            return (int)x;
        }

        public bool Equals(long a, long b) => a == b;
        public int GetHashCode(long k) => Mix((ulong)k);
        public bool Equals((long, int) a, (long, int) b) => a.Item1 == b.Item1 && a.Item2 == b.Item2;
        public int GetHashCode((long, int) k) => Mix((ulong)k.Item1 * 31UL + (ulong)k.Item2);
        public bool Equals((long, long, long) a, (long, long, long) b) => a.Item1 == b.Item1 && a.Item2 == b.Item2 && a.Item3 == b.Item3;
        public int GetHashCode((long, long, long) k) => Mix(((ulong)k.Item1 * 0x9E3779B97F4A7C15UL) ^ ((ulong)k.Item2 * 0xC2B2AE3D27D4EB4FUL) ^ (ulong)k.Item3);
    }
}
