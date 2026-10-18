using System;
using GhostGPU.Core;

namespace GhostGPU.Booleans
{
    public readonly struct Segment3D
    {
        public readonly V3 P0;
        public readonly V3 P1;

        public Segment3D(V3 p0, V3 p1)
        {
            P0 = p0;
            P1 = p1;
        }

        public double LengthSq => (P1.X - P0.X) * (P1.X - P0.X) + (P1.Y - P0.Y) * (P1.Y - P0.Y) + (P1.Z - P0.Z) * (P1.Z - P0.Z);
        public double Length => Math.Sqrt(LengthSq);
    }

    /// <summary>
    /// Robust double-precision triangle-triangle intersection using filtered orientation predicates.
    /// Returns the 3D intersection line segment if triangles cross.
    /// </summary>
    public static class TriTriIntersection
    {
        public static bool Intersect(V3 a0, V3 a1, V3 a2, V3 b0, V3 b1, V3 b2, double tolerance, out Segment3D segment)
        {
            segment = default;

            // 1. Plane B normal
            V3 abB = b1 - b0;
            V3 acB = b2 - b0;
            V3 nB = Cross(abB, acB);
            double lenBSq = Dot(nB, nB);
            if (lenBSq < 1e-24) return false;
            double lenB = Math.Sqrt(lenBSq);

            // Signed distances of A vertices to plane B
            double da0 = Dot(a0 - b0, nB);
            double da1 = Dot(a1 - b0, nB);
            double da2 = Dot(a2 - b0, nB);

            double tolB = tolerance * lenB;
            if (Math.Abs(da0) < tolB) da0 = 0.0;
            if (Math.Abs(da1) < tolB) da1 = 0.0;
            if (Math.Abs(da2) < tolB) da2 = 0.0;

            if ((da0 > 0.0 && da1 > 0.0 && da2 > 0.0) || (da0 < 0.0 && da1 < 0.0 && da2 < 0.0))
                return false;

            // 2. Plane A normal
            V3 abA = a1 - a0;
            V3 acA = a2 - a0;
            V3 nA = Cross(abA, acA);
            double lenASq = Dot(nA, nA);
            if (lenASq < 1e-24) return false;
            double lenA = Math.Sqrt(lenASq);

            // Signed distances of B vertices to plane A
            double db0 = Dot(b0 - a0, nA);
            double db1 = Dot(b1 - a0, nA);
            double db2 = Dot(b2 - a0, nA);

            double tolA = tolerance * lenA;
            if (Math.Abs(db0) < tolA) db0 = 0.0;
            if (Math.Abs(db1) < tolA) db1 = 0.0;
            if (Math.Abs(db2) < tolA) db2 = 0.0;

            if ((db0 > 0.0 && db1 > 0.0 && db2 > 0.0) || (db0 < 0.0 && db1 < 0.0 && db2 < 0.0))
                return false;

            // 3. Direction of intersection line
            V3 d = Cross(nA, nB);
            double lenDSq = Dot(d, d);
            if (lenDSq < tolerance * tolerance * lenASq * lenBSq)
            {
                // Coplanar or nearly coplanar
                return false;
            }

            // 4. Find intersection segment of triangle A with plane B
            if (!GetPlaneIntersectionSegment(a0, a1, a2, da0, da1, da2, d, out V3 pa0, out V3 pa1, out double ta0, out double ta1))
                return false;

            // 5. Find intersection segment of triangle B with plane A
            if (!GetPlaneIntersectionSegment(b0, b1, b2, db0, db1, db2, d, out V3 pb0, out V3 pb1, out double tb0, out double tb1))
                return false;

            // Order intervals
            double tAmin = ta0, tAmax = ta1;
            V3 pAmin = pa0, pAmax = pa1;
            if (tAmin > tAmax)
            {
                tAmin = ta1; tAmax = ta0;
                pAmin = pa1; pAmax = pa0;
            }

            double tBmin = tb0, tBmax = tb1;
            V3 pBmin = pb0, pBmax = pb1;
            if (tBmin > tBmax)
            {
                tBmin = tb1; tBmax = tb0;
                pBmin = pb1; pBmax = pb0;
            }

            // Overlap of [tAmin, tAmax] and [tBmin, tBmax]
            double tStart = Math.Max(tAmin, tBmin);
            double tEnd = Math.Min(tAmax, tBmax);

            if (tStart >= tEnd - tolerance * Math.Sqrt(lenDSq))
                return false;

            // Overlap endpoints are exactly interval endpoints: no interpolation, so shared points stay bit-identical.
            V3 pStart = tAmin >= tBmin ? pAmin : pBmin;
            V3 pEnd = tAmax <= tBmax ? pAmax : pBmax;

            V3 diff = pEnd - pStart;
            if (Dot(diff, diff) < tolerance * tolerance)
                return false;

            segment = new Segment3D(pStart, pEnd);
            return true;
        }

        private static bool GetPlaneIntersectionSegment(
            V3 v0, V3 v1, V3 v2,
            double d0, double d1, double d2,
            V3 lineDir,
            out V3 p0, out V3 p1,
            out double t0, out double t1)
        {
            p0 = default; p1 = default;
            t0 = 0.0; t1 = 0.0;

            Span<V3> pts = stackalloc V3[3];
            int ptCount = 0;

            // Edge 0: v0 -> v1
            CheckEdge(v0, v1, d0, d1, pts, ref ptCount);
            // Edge 1: v1 -> v2
            CheckEdge(v1, v2, d1, d2, pts, ref ptCount);
            // Edge 2: v2 -> v0
            CheckEdge(v2, v0, d2, d0, pts, ref ptCount);

            if (ptCount < 2)
                return false;

            p0 = pts[0];
            p1 = pts[1];
            t0 = Dot(p0, lineDir);
            t1 = Dot(p1, lineDir);
            return true;
        }

        private static void CheckEdge(V3 va, V3 vb, double da, double db, Span<V3> pts, ref int ptCount)
        {
            if (ptCount >= 2) return;

            if (da == 0.0)
            {
                AddUniquePoint(va, pts, ref ptCount);
            }
            if (db == 0.0)
            {
                AddUniquePoint(vb, pts, ref ptCount);
            }
            else if ((da > 0.0 && db < 0.0) || (da < 0.0 && db > 0.0))
            {
                double s = da / (da - db);
                V3 p = new V3(va.X + s * (vb.X - va.X), va.Y + s * (vb.Y - va.Y), va.Z + s * (vb.Z - va.Z));
                AddUniquePoint(p, pts, ref ptCount);
            }
        }

        private static void AddUniquePoint(V3 p, Span<V3> pts, ref int ptCount)
        {
            for (int i = 0; i < ptCount; i++)
            {
                V3 diff = pts[i] - p;
                if (Dot(diff, diff) < 1e-18)
                    return;
            }
            if (ptCount < pts.Length)
            {
                pts[ptCount++] = p;
            }
        }

        private static V3 Cross(V3 a, V3 b)
        {
            return new V3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
        }

        private static double Dot(V3 a, V3 b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }
    }
}
