using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Booleans
{
    public readonly struct Point2D
    {
        public readonly double X;
        public readonly double Y;

        public Point2D(double x, double y)
        {
            X = x;
            Y = y;
        }

        public static Point2D operator -(Point2D a, Point2D b) => new Point2D(a.X - b.X, a.Y - b.Y);
        public static Point2D operator +(Point2D a, Point2D b) => new Point2D(a.X + b.X, a.Y + b.Y);
    }

    /// <summary>
    /// Stand-alone constrained re-triangulation of one triangle by loose cut segments (MeshBoolean itself uses the shared
    /// global-vertex pipeline). Segment endpoints within tolerance of a corner snap to it, within tolerance of an edge snap
    /// onto it, and endpoints within tolerance of each other are merged; the triangulation is the exact-predicate
    /// <see cref="Arrangement"/>, which splits crossing constraints and never folds. Output keeps the input orientation.
    /// </summary>
    public static class ConstrainedTriangulation
    {
        /// <summary>Returns false if a near-degenerate configuration had to be snapped (the output is still valid, gap-free).</summary>
        public static bool Retriangulate(
            V3 a, V3 b, V3 c,
            List<Segment3D> segments,
            double tolerance,
            List<V3> outVertices,
            List<int> outTriangles)
        {
            V3 n = Cross(b - a, c - a);
            double maxEdge = Math.Max(Math.Max((b - a).Length, (c - b).Length), (a - c).Length);
            if (segments == null || segments.Count == 0 || !(maxEdge > 0) || n.Length <= 1e-14 * maxEdge * maxEdge)
            {
                Emit(new List<V3> { a, b, c }, new List<int> { 0, 1, 2 }, outVertices, outTriangles);
                return true;
            }

            double tol = Math.Min(tolerance, 0.01 * n.Length / maxEdge);
            var pts = new List<V3> { a, b, c };
            var edgePts = new List<int>[3];
            var edgeT = new List<double>[3];
            var cons = new List<(int, int)>();

            int Add(V3 p)
            {
                for (int k = 0; k < pts.Count; k++)
                    if ((pts[k] - p).Length <= tol) return k;
                for (int e = 0; e < 3; e++)
                {
                    V3 e0 = pts[e], d = pts[(e + 1) % 3] - e0;
                    double t = V3.Dot(p - e0, d) / V3.Dot(d, d);
                    if (t <= 0 || t >= 1 || (e0 + d * t - p).Length > tol) continue;
                    int id = pts.Count;
                    pts.Add(e0 + d * t);
                    (edgePts[e] ?? (edgePts[e] = new List<int>())).Add(id);
                    (edgeT[e] ?? (edgeT[e] = new List<double>())).Add(t);
                    return id;
                }
                pts.Add(p);
                return pts.Count - 1;
            }

            foreach (var s in segments)
            {
                int i = Add(s.P0), j = Add(s.P1);
                if (i != j) cons.Add((i, j));
            }
            for (int e = 0; e < 3; e++)
                if (edgePts[e] != null)
                {
                    var ids = edgePts[e].ToArray();
                    Array.Sort(edgeT[e].ToArray(), ids);
                    edgePts[e] = new List<int>(ids);
                }

            var arr = Arrangement.Build(new[] { 0, 1, 2 }, pts, edgePts, cons);
            if (arr == null)
            {
                Emit(new List<V3> { a, b, c }, new List<int> { 0, 1, 2 }, outVertices, outTriangles);
                return false;
            }
            int baseNew = pts.Count;
            pts.AddRange(arr.NewPoints);
            var tris = new List<int>(arr.Triangles.Count);
            foreach (int id in arr.Triangles) tris.Add(id >= 0 ? id : baseNew + (-id - 1));
            Emit(pts, tris, outVertices, outTriangles);
            return arr.Degenerate == 0;
        }

        private static void Emit(List<V3> pts, List<int> tris, List<V3> outVertices, List<int> outTriangles)
        {
            var remap = new Dictionary<int, int>();
            foreach (int v in tris)
            {
                if (!remap.TryGetValue(v, out int r)) { r = outVertices.Count; remap[v] = r; outVertices.Add(pts[v]); }
                outTriangles.Add(r);
            }
        }

        private static V3 Cross(V3 a, V3 b) => new V3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    }
}
