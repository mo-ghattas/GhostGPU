using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Booleans
{
    /// <summary>
    /// Constrained triangulation of one input triangle with its boundary points and cut segments, in the 2D projection
    /// that drops the dominant normal axis (projected coordinates are the exact 3D doubles, so every orientation test is
    /// exact via <see cref="Predicates"/>). Steps:
    /// 1. ear-clip the boundary polygon (corners + points on each edge, ordered along the edge; boundary membership is
    ///    topological, never decided by a tolerance);
    /// 2. insert interior points (1-&gt;3, or 2-&gt;4 / 1-&gt;2 when exactly on an edge);
    /// 3. insert each constraint: split at vertices exactly on it, split at crossings with earlier constraints, then
    ///    remove the triangles it crosses and ear-clip the two cavity polygons (Anglada-style cavity retriangulation).
    /// No step can leave a folded triangle; impossible configurations (only reachable through rounding of constructed
    /// points) are snapped and counted in <see cref="Degenerate"/>.
    /// Vertex ids: global ids &gt;= 0; points created here (constraint crossings) are returned as -(k + 1), k indexing
    /// <see cref="NewPoints"/>.
    /// </summary>
    internal sealed class Arrangement
    {
        private readonly List<double> _x = new List<double>(), _y = new List<double>();
        private readonly List<V3> _p3 = new List<V3>();
        private readonly List<int> _gid = new List<int>();
        private readonly Dictionary<int, int> _local = new Dictionary<int, int>();
        private readonly List<int[]> _tris = new List<int[]>();
        private readonly HashSet<long> _cons = new HashSet<long>(KeyComparer.Instance);
        private readonly int _axis;
        private readonly bool _mirror;
        // Boundary bookkeeping: polygon position and edge index of boundary points (corners belong to two edges).
        private readonly Dictionary<int, int> _boundaryEdge = new Dictionary<int, int>();
        private readonly List<int>[] _edgeChain = new List<int>[3];
        private readonly List<int> _aliasList = new List<int>();

        public int Degenerate;
        public readonly List<V3> NewPoints = new List<V3>();
        public readonly List<int> Triangles = new List<int>();
        public readonly List<(int, int)> Constrained = new List<(int, int)>();

        private Arrangement(V3 normal)
        {
            double ax = Math.Abs(normal.X), ay = Math.Abs(normal.Y), az = Math.Abs(normal.Z);
            _axis = ax >= ay && ax >= az ? 0 : ay >= az ? 1 : 2;
            double c = _axis == 0 ? normal.X : _axis == 1 ? normal.Y : normal.Z;
            _mirror = c < 0;
        }

        /// <summary>
        /// Returns null if the triangle is degenerate in its own projection (caller drops it).
        /// edgePoints[i] = global ids strictly inside edge corner i -&gt; corner i+1, already ordered from corner i.
        /// </summary>
        public static Arrangement Build(int[] corners, IReadOnlyList<V3> pts, List<int>[] edgePoints, List<(int, int)> constraints)
        {
            V3 c0 = pts[corners[0]], c1 = pts[corners[1]], c2 = pts[corners[2]];
            var arr = new Arrangement(Cross(c1 - c0, c2 - c0));
            for (int i = 0; i < 3; i++) arr.AddPoint(corners[i], pts[corners[i]]);
            if (arr.Orient(0, 1, 2) <= 0) return null;

            // Boundary polygon.
            var poly = new List<int>();
            for (int e = 0; e < 3; e++)
            {
                arr._edgeChain[e] = new List<int> { e };
                poly.Add(e);
                foreach (int g in edgePoints[e] ?? new List<int>())
                {
                    if (arr._local.ContainsKey(g)) continue;
                    int l = arr.AddPoint(g, pts[g]);
                    arr._boundaryEdge[l] = e;
                    arr._edgeChain[e].Add(l);
                    poly.Add(l);
                }
                arr._edgeChain[e].Add((e + 1) % 3);
            }
            if (!arr.EarClip(poly)) arr.Degenerate++;

            // Interior points, in the order they are first referenced.
            foreach (var (a, b) in constraints)
            {
                if (!arr._local.ContainsKey(a)) arr.InsertPoint(arr.AddPoint(a, pts[a]));
                if (!arr._local.ContainsKey(b)) arr.InsertPoint(arr.AddPoint(b, pts[b]));
            }
            foreach (var (a, b) in constraints)
                arr.InsertConstraint(arr.Resolve(arr._local[a]), arr.Resolve(arr._local[b]));

            foreach (var t in arr._tris)
                for (int k = 0; k < 3; k++) arr.Triangles.Add(arr.Out(t[k]));
            foreach (long key in arr._cons)
                arr.Constrained.Add((arr.Out((int)(key >> 32)), arr.Out((int)(key & 0xFFFFFFFF))));
            return arr;
        }

        private int Out(int l) => _gid[l];

        private int AddPoint(int gid, V3 p)
        {
            int l = _x.Count;
            double u = _axis == 0 ? p.Y : _axis == 1 ? p.Z : p.X;
            double v = _axis == 0 ? p.Z : _axis == 1 ? p.X : p.Y;
            if (_mirror) { double tmp = u; u = v; v = tmp; }
            _x.Add(u); _y.Add(v); _p3.Add(p); _gid.Add(gid); _aliasList.Add(l);
            if (gid >= 0) _local[gid] = l;
            return l;
        }

        private int AddNewPoint(V3 p)
        {
            NewPoints.Add(p);
            return AddPoint(-NewPoints.Count, p); // stored as -(k+1)
        }

        private int Resolve(int l)
        {
            while (_aliasList[l] != l) l = _aliasList[l];
            return l;
        }

        private int Orient(int a, int b, int c) => Predicates.Orient2D(_x[a], _y[a], _x[b], _y[b], _x[c], _y[c]);

        private static long Key(int a, int b) => a < b ? ((long)a << 32) | (uint)b : ((long)b << 32) | (uint)a;

        // ---------------------------------------------------------------- ear clipping
        /// <summary>Triangulates a CCW simple polygon (collinear runs allowed). Returns false if it had to force an ear.</summary>
        private bool EarClip(List<int> polygon)
        {
            var v = new List<int>(polygon);
            bool clean = true;
            while (v.Count > 3)
            {
                int n = v.Count, ear = -1;
                for (int i = 0; i < n && ear < 0; i++)
                {
                    int p = v[(i + n - 1) % n], c = v[i], q = v[(i + 1) % n];
                    if (Orient(p, c, q) <= 0) continue;
                    bool empty = true;
                    for (int j = 0; j < n && empty; j++)
                    {
                        int w = v[j];
                        if (w == p || w == c || w == q) continue;
                        if (Orient(p, c, w) >= 0 && Orient(c, q, w) >= 0 && Orient(q, p, w) >= 0) empty = false;
                    }
                    if (empty) ear = i;
                }
                if (ear < 0)
                {
                    // Not reachable for a simple polygon; force the first convex corner so the loop terminates.
                    clean = false;
                    for (int i = 0; i < n && ear < 0; i++)
                        if (Orient(v[(i + n - 1) % n], v[i], v[(i + 1) % n]) > 0) ear = i;
                    if (ear < 0) return false;
                }
                _tris.Add(new[] { v[(ear + n - 1) % n], v[ear], v[(ear + 1) % n] });
                v.RemoveAt(ear);
            }
            if (v.Count == 3 && Orient(v[0], v[1], v[2]) > 0) _tris.Add(new[] { v[0], v[1], v[2] });
            else if (v.Count == 3) clean = false;
            return clean;
        }

        // ---------------------------------------------------------------- point insertion
        private void InsertPoint(int p)
        {
            int bestT = -1, bestE = -1;
            double bestScore = double.NegativeInfinity;
            for (int t = 0; t < _tris.Count; t++)
            {
                int[] tr = _tris[t];
                int o0 = Orient(tr[0], tr[1], p), o1 = Orient(tr[1], tr[2], p), o2 = Orient(tr[2], tr[0], p);
                if (o0 < 0 || o1 < 0 || o2 < 0)
                {
                    // Track the nearest triangle (largest minimum signed distance) in case p lies outside them all.
                    double score = double.PositiveInfinity;
                    int e = -1;
                    for (int k = 0; k < 3; k++)
                    {
                        double sd = SignedDist(tr[k], tr[(k + 1) % 3], p);
                        if (sd < score) { score = sd; e = k; }
                    }
                    if (score > bestScore) { bestScore = score; bestT = t; bestE = e; }
                    continue;
                }
                int zeros = (o0 == 0 ? 1 : 0) + (o1 == 0 ? 1 : 0) + (o2 == 0 ? 1 : 0);
                if (zeros >= 2)
                {
                    // Coincides with a vertex in the projection: alias to it.
                    int vtx = o0 == 0 && o2 == 0 ? tr[0] : o0 == 0 && o1 == 0 ? tr[1] : tr[2];
                    _aliasList[p] = vtx;
                    return;
                }
                if (zeros == 1)
                {
                    int e = o0 == 0 ? 0 : o1 == 0 ? 1 : 2;
                    if (!SplitEdge(tr[e], tr[(e + 1) % 3], p))
                    {
                        Degenerate++;
                        _aliasList[p] = Dist2(tr[e], p) <= Dist2(tr[(e + 1) % 3], p) ? tr[e] : tr[(e + 1) % 3];
                    }
                    return;
                }
                _tris[t] = new[] { tr[0], tr[1], p };
                _tris.Add(new[] { tr[1], tr[2], p });
                _tris.Add(new[] { tr[2], tr[0], p });
                return;
            }

            // Outside every triangle (a constructed point rounded across the boundary): put it on the nearest boundary
            // edge if both children stay positive, else snap it to the nearest corner of that triangle.
            Degenerate++;
            int[] b = _tris[bestT];
            int u = b[bestE], w = b[(bestE + 1) % 3], c = b[(bestE + 2) % 3];
            if (FindTri(w, u, out _) < 0 && Orient(u, p, c) > 0 && Orient(p, w, c) > 0)
            {
                _tris[bestT] = new[] { u, p, c };
                _tris.Add(new[] { p, w, c });
                return;
            }
            _aliasList[p] = Dist2(u, p) <= Dist2(w, p) ? u : w;
        }

        /// <summary>Splits every triangle on edge (u,v) at p. Returns false (and changes nothing) if a child would not be positive.</summary>
        private bool SplitEdge(int u, int v, int p)
        {
            int t1 = FindTri(u, v, out int w1), t2 = FindTri(v, u, out int w2);
            if ((t1 >= 0 && (Orient(u, p, w1) <= 0 || Orient(p, v, w1) <= 0)) ||
                (t2 >= 0 && (Orient(v, p, w2) <= 0 || Orient(p, u, w2) <= 0)))
                return false;
            if (t1 >= 0) { _tris[t1] = new[] { u, p, w1 }; _tris.Add(new[] { p, v, w1 }); }
            if (t2 >= 0) { _tris[t2] = new[] { v, p, w2 }; _tris.Add(new[] { p, u, w2 }); }
            if (_cons.Remove(Key(u, v))) { _cons.Add(Key(u, p)); _cons.Add(Key(p, v)); }
            return true;
        }

        // ---------------------------------------------------------------- constraint insertion
        private void InsertConstraint(int a0, int b0)
        {
            var stack = new Stack<(int, int)>();
            stack.Push((a0, b0));
            int guard = 0;
            while (stack.Count > 0)
            {
                var (a, b) = stack.Pop();
                a = Resolve(a); b = Resolve(b);
                if (a == b) continue;
                if (++guard > 100000) { Degenerate++; return; }

                // Both on the same original edge: the constraint is that piece of the boundary chain.
                if (SameBoundaryEdge(a, b, out int edge)) { MarkChain(edge, a, b); continue; }
                if (FindTri(a, b, out _) >= 0 || FindTri(b, a, out _) >= 0) { _cons.Add(Key(a, b)); continue; }

                int k = VertexOnSegment(a, b);
                if (k >= 0) { stack.Push((k, b)); stack.Push((a, k)); continue; }

                var crossed = CrossedEdges(a, b, out int cu, out int cv);
                if (crossed.Count == 0) { Degenerate++; continue; }
                if (cu >= 0)
                {
                    // Crosses an earlier constraint: split both at the crossing point.
                    int x = Crossing(a, b, cu, cv);
                    if (x < 0) { Degenerate++; continue; }
                    stack.Push((x, b)); stack.Push((a, x));
                    continue;
                }
                if (!Cavity(a, b, crossed)) { Degenerate++; continue; }
                _cons.Add(Key(a, b));
            }
        }

        private bool SameBoundaryEdge(int a, int b, out int edge)
        {
            edge = -1;
            for (int e = 0; e < 3; e++)
            {
                bool ha = a == e || a == (e + 1) % 3 || (_boundaryEdge.TryGetValue(a, out int ea) && ea == e);
                bool hb = b == e || b == (e + 1) % 3 || (_boundaryEdge.TryGetValue(b, out int eb) && eb == e);
                if (ha && hb) { edge = e; return true; }
            }
            return false;
        }

        private void MarkChain(int e, int a, int b)
        {
            var chain = _edgeChain[e];
            int ia = chain.IndexOf(a), ib = chain.IndexOf(b);
            if (ia > ib) { int t = ia; ia = ib; ib = t; }
            for (int i = ia; i < ib; i++)
            {
                int u = Resolve(chain[i]), v = Resolve(chain[i + 1]);
                if (u != v) _cons.Add(Key(u, v));
            }
        }

        private int VertexOnSegment(int a, int b)
        {
            int best = -1;
            double bestD = double.MaxValue;
            bool useX = Math.Abs(_x[b] - _x[a]) >= Math.Abs(_y[b] - _y[a]);
            double lo = useX ? Math.Min(_x[a], _x[b]) : Math.Min(_y[a], _y[b]);
            double hi = useX ? Math.Max(_x[a], _x[b]) : Math.Max(_y[a], _y[b]);
            for (int k = 0; k < _x.Count; k++)
            {
                if (_aliasList[k] != k || k == a || k == b) continue;
                double c = useX ? _x[k] : _y[k];
                if (c <= lo || c >= hi) continue;
                if (Orient(a, b, k) != 0) continue;
                if (!InTriangulation(k)) continue;
                double d = Dist2(a, k);
                if (d < bestD) { bestD = d; best = k; }
            }
            return best;
        }

        private bool InTriangulation(int k)
        {
            foreach (var t in _tris) if (t[0] == k || t[1] == k || t[2] == k) return true;
            return false;
        }

        /// <summary>Edges properly crossing the open segment (a,b); (cu,cv) = a crossed constrained edge, if any.</summary>
        private List<(int, int)> CrossedEdges(int a, int b, out int cu, out int cv)
        {
            cu = cv = -1;
            var list = new List<(int, int)>();
            var seen = new HashSet<long>(KeyComparer.Instance);
            foreach (var t in _tris)
            {
                for (int e = 0; e < 3; e++)
                {
                    int u = t[e], v = t[(e + 1) % 3];
                    if (u == a || u == b || v == a || v == b) continue;
                    long key = Key(u, v);
                    if (!seen.Add(key)) continue;
                    int o1 = Orient(a, b, u), o2 = Orient(a, b, v);
                    if (o1 == 0 || o2 == 0 || o1 == o2) continue;
                    int o3 = Orient(u, v, a), o4 = Orient(u, v, b);
                    if (o3 == 0 || o4 == 0 || o3 == o4) continue;
                    list.Add((u, v));
                    if (cu < 0 && _cons.Contains(key)) { cu = u; cv = v; }
                }
            }
            return list;
        }

        private int Crossing(int a, int b, int u, int v)
        {
            double rx = _x[b] - _x[a], ry = _y[b] - _y[a], sx = _x[v] - _x[u], sy = _y[v] - _y[u];
            double den = rx * sy - ry * sx;
            double t = ((_x[u] - _x[a]) * sy - (_y[u] - _y[a]) * sx) / den;
            if (!(t > 0 && t < 1)) t = Math.Max(1e-9, Math.Min(1 - 1e-9, double.IsNaN(t) ? 0.5 : t));
            V3 pa = _p3[a], pb = _p3[b];
            int x = AddNewPoint(pa + (pb - pa) * t);
            if (SplitEdge(u, v, x)) return x;
            // Rounding put the point outside the quad around (u,v): snap the crossing to the endpoint of (u,v)
            // nearest to segment (a,b), which keeps every triangle positive.
            Degenerate++;
            _aliasList[x] = Math.Abs(SignedDist(a, b, u)) <= Math.Abs(SignedDist(a, b, v)) ? u : v;
            return Resolve(x);
        }

        private bool Cavity(int a, int b, List<(int, int)> crossed)
        {
            var crossedKeys = new HashSet<long>(KeyComparer.Instance);
            foreach (var (u, v) in crossed) crossedKeys.Add(Key(u, v));
            var removed = new List<int>();
            for (int t = 0; t < _tris.Count; t++)
            {
                int[] tr = _tris[t];
                if (crossedKeys.Contains(Key(tr[0], tr[1])) || crossedKeys.Contains(Key(tr[1], tr[2])) || crossedKeys.Contains(Key(tr[2], tr[0])))
                    removed.Add(t);
            }

            var directed = new HashSet<long>(KeyComparer.Instance);
            foreach (int t in removed)
                for (int e = 0; e < 3; e++) directed.Add(((long)_tris[t][e] << 32) | (uint)_tris[t][(e + 1) % 3]);
            var next = new Dictionary<int, int>();
            foreach (long d in directed)
            {
                int u = (int)(d >> 32), v = (int)(d & 0xFFFFFFFF);
                if (directed.Contains(((long)v << 32) | (uint)u)) continue;
                if (next.ContainsKey(u)) return false; // non-simple cavity boundary
                next[u] = v;
            }
            if (!next.ContainsKey(a) || !next.ContainsKey(b)) return false;
            var left = Walk(next, a, b);
            var right = Walk(next, b, a);
            if (left == null || right == null) return false;

            var saved = new List<int[]>();
            foreach (int t in removed) saved.Add(_tris[t]);
            var keep = new List<int[]>(_tris.Count);
            var removedSet = new HashSet<int>(removed);
            for (int t = 0; t < _tris.Count; t++) if (!removedSet.Contains(t)) keep.Add(_tris[t]);
            _tris.Clear();
            _tris.AddRange(keep);
            int before = _tris.Count;
            if (!EarClip(left) | !EarClip(right))
            {
                _tris.RemoveRange(before, _tris.Count - before);
                _tris.AddRange(saved);
                return false;
            }
            return true;
        }

        private static List<int> Walk(Dictionary<int, int> next, int from, int to)
        {
            var path = new List<int> { from };
            int cur = from;
            for (int guard = 0; guard <= next.Count; guard++)
            {
                if (!next.TryGetValue(cur, out cur)) return null;
                path.Add(cur);
                if (cur == to) return path;
            }
            return null;
        }

        private int FindTri(int u, int v, out int opposite)
        {
            for (int t = 0; t < _tris.Count; t++)
            {
                int[] tr = _tris[t];
                for (int e = 0; e < 3; e++)
                    if (tr[e] == u && tr[(e + 1) % 3] == v) { opposite = tr[(e + 2) % 3]; return t; }
            }
            opposite = -1;
            return -1;
        }

        private double SignedDist(int a, int b, int p)
        {
            double ex = _x[b] - _x[a], ey = _y[b] - _y[a];
            double len = Math.Sqrt(ex * ex + ey * ey);
            return len > 0 ? (ex * (_y[p] - _y[a]) - ey * (_x[p] - _x[a])) / len : 0;
        }

        private double Dist2(int a, int b)
        {
            double dx = _x[a] - _x[b], dy = _y[a] - _y[b];
            return dx * dx + dy * dy;
        }

        private static V3 Cross(V3 a, V3 b) => new V3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    }
}
