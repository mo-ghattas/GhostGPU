using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using GhostGPU.Core;

namespace GhostGPU.Booleans
{
    /// <summary>
    /// Builds the conforming subdivision of every operand triangle by all intersection curves, with ONE global vertex
    /// table shared by every triangle (so neighbours agree on every shared point by construction):
    /// - original vertices: welded per operand where positions are bit-identical (recovers the topology of unwelded
    ///   Rhino meshes), so each original edge has one id and knows all its triangles;
    /// - every intersection point has a symbolic key from which its coordinates are computed canonically, once:
    ///   input vertex | edge x face (EF) | edge x edge (EE) | face x face x face (triple). Which key applies is decided by
    ///   exact orientation predicates on the input coordinates, so the two triangles of an edge always derive the same key;
    /// - points keyed on an edge are inserted into every triangle of that edge (no T-junctions along original edges);
    /// - points closer than a relative epsilon are merged globally (union-find), before any triangle is re-triangulated,
    ///   so the merge is consistent across neighbours;
    /// - each triangle is then re-triangulated by <see cref="Arrangement"/> (exact predicates, cannot fold).
    /// </summary>
    internal sealed class Intersector
    {
        private const int KVertex = 0, KEdgeFace = 1, KEdgeEdge = 2, KTriple = 3;

        private readonly struct PKey : IEquatable<PKey>
        {
            public readonly int Kind, A, B, C;
            public PKey(int kind, int a, int b = 0, int c = 0) { Kind = kind; A = a; B = b; C = c; }
            public bool Equals(PKey o) => Kind == o.Kind && A == o.A && B == o.B && C == o.C;
            public override bool Equals(object obj) => obj is PKey k && Equals(k);
            public override int GetHashCode() { unchecked { return ((Kind * 397 ^ A) * 397 ^ B) * 397 ^ C; } }
        }

        private struct PPoint
        {
            public PKey Key;
            public V3 Pos;
            public int E1, E2; // original edges the point lies on (-1 = none)
        }

        private struct SegReq
        {
            public PPoint P, Q;
            public int F1, F2, LineEdge;
            public PPoint[] Ties; // unchosen candidates within the merge epsilon of an endpoint (registered so they merge)
        }

        private sealed class Seg
        {
            public int P, Q, F1, F2, LineEdge; // LineEdge >= 0: the segment is a piece of that original edge (coplanar case)
            public List<int> Splits;
            public int[] Chain;
        }

        // Inputs / topology.
        public readonly List<V3> Pts = new List<V3>();
        public readonly int TriCount;
        public readonly int[] Tv;       // 3 original (pre-merge) vertex ids per triangle
        public readonly int[] TriOwner; // operand of each triangle
        private readonly int[] _triEdge;
        private readonly List<int> _edgeU = new List<int>(), _edgeV = new List<int>();
        private readonly List<List<int>> _edgeTris = new List<List<int>>();
        private readonly V3[] _normal;
        private readonly bool[] _degenerate;

        // Points and segments.
        private readonly Dictionary<PKey, int> _keyToId = new Dictionary<PKey, int>();
        private readonly Dictionary<int, List<int>> _edgePoints = new Dictionary<int, List<int>>();
        private readonly List<Seg> _segs = new List<Seg>();
        private readonly List<int>[] _faceSegs;
        private readonly Dictionary<long, int> _pairSeg = new Dictionary<long, int>(KeyComparer.Instance);
        public int[] Rep;

        // Outputs.
        public readonly List<int> SubTris = new List<int>();   // 3 global ids per sub-triangle
        public readonly List<int> SubOwner = new List<int>();  // operand per sub-triangle
        public readonly HashSet<long> CutEdges = new HashSet<long>(KeyComparer.Instance);
        public int Degenerate, DroppedTriangles;

        private readonly double _eps;

        public Intersector(List<TriMesh> operands, int[] triOffset, double mergeEps)
        {
            _eps = mergeEps;
            TriCount = triOffset[operands.Count];
            Tv = new int[3 * TriCount];
            TriOwner = new int[TriCount];
            _triEdge = new int[3 * TriCount];
            _normal = new V3[TriCount];
            _degenerate = new bool[TriCount];
            _faceSegs = new List<int>[TriCount];

            for (int k = 0; k < operands.Count; k++)
            {
                var m = operands[k];
                var weld = new Dictionary<(long, long, long), int>(KeyComparer.Instance);
                var local = new int[m.Vertices.Length];
                for (int v = 0; v < local.Length; v++)
                {
                    V3 p = m.Vertices[v];
                    var key = (BitConverter.DoubleToInt64Bits(p.X + 0.0), BitConverter.DoubleToInt64Bits(p.Y + 0.0), BitConverter.DoubleToInt64Bits(p.Z + 0.0));
                    if (!weld.TryGetValue(key, out int id)) { id = Pts.Count; Pts.Add(p); weld[key] = id; }
                    local[v] = id;
                }
                for (int t = 0; t < m.TriangleCount; t++)
                {
                    int g = triOffset[k] + t;
                    TriOwner[g] = k;
                    for (int c = 0; c < 3; c++) Tv[3 * g + c] = local[m.Triangles[3 * t + c]];
                }
            }

            var edgeIndex = new Dictionary<long, int>(KeyComparer.Instance);
            for (int g = 0; g < TriCount; g++)
            {
                int a = Tv[3 * g], b = Tv[3 * g + 1], c = Tv[3 * g + 2];
                V3 pa = Pts[a], pb = Pts[b], pc = Pts[c];
                _normal[g] = Cross(pb - pa, pc - pa);
                _degenerate[g] = a == b || b == c || c == a ||
                                 (Predicates.Orient2D(pa.X, pa.Y, pb.X, pb.Y, pc.X, pc.Y) == 0 &&
                                  Predicates.Orient2D(pa.Y, pa.Z, pb.Y, pb.Z, pc.Y, pc.Z) == 0 &&
                                  Predicates.Orient2D(pa.Z, pa.X, pb.Z, pb.X, pc.Z, pc.X) == 0);
                for (int e = 0; e < 3; e++)
                {
                    int u = Tv[3 * g + e], v = Tv[3 * g + (e + 1) % 3];
                    if (u > v) { int tmp = u; u = v; v = tmp; }
                    long key = ((long)u << 32) | (uint)v;
                    if (!edgeIndex.TryGetValue(key, out int id))
                    {
                        id = _edgeU.Count;
                        edgeIndex[key] = id;
                        _edgeU.Add(u); _edgeV.Add(v); _edgeTris.Add(new List<int>());
                    }
                    _triEdge[3 * g + e] = id;
                    _edgeTris[id].Add(g);
                }
            }
        }

        public bool IsDegenerate(int g) => _degenerate[g];

        // ================================================================ narrowphase
        /// <summary>Intersects candidate pairs (global triangle ids) in parallel, then registers them serially in order.</summary>
        public void AddPairs(int[] pairs, int offsetK, int offsetL)
        {
            int count = pairs.Length / 2;
            var results = new List<SegReq>[count];
            Parallel.For(0, count, p =>
            {
                int t = offsetK + pairs[2 * p], u = offsetL + pairs[2 * p + 1];
                if (!_degenerate[t] && !_degenerate[u]) results[p] = Narrow(t, u);
            });
            foreach (var list in results)
            {
                if (list == null) continue;
                foreach (var r in list)
                {
                    if (r.Ties != null) foreach (var tie in r.Ties) Register(tie);
                    int p = Register(r.P), q = Register(r.Q);
                    if (p == q) continue;
                    int id = _segs.Count;
                    _segs.Add(new Seg { P = p, Q = q, F1 = r.F1, F2 = r.F2, LineEdge = r.LineEdge });
                    (_faceSegs[r.F1] ?? (_faceSegs[r.F1] = new List<int>())).Add(id);
                    (_faceSegs[r.F2] ?? (_faceSegs[r.F2] = new List<int>())).Add(id);
                    if (r.LineEdge < 0) _pairSeg[PairKey(r.F1, r.F2)] = id;
                }
            }
        }

        private List<SegReq> Narrow(int t, int u)
        {
            V3 t0 = Pts[Tv[3 * t]], t1 = Pts[Tv[3 * t + 1]], t2 = Pts[Tv[3 * t + 2]];
            V3 u0 = Pts[Tv[3 * u]], u1 = Pts[Tv[3 * u + 1]], u2 = Pts[Tv[3 * u + 2]];
            int st0 = Predicates.Side3D(u0, u1, u2, t0), st1 = Predicates.Side3D(u0, u1, u2, t1), st2 = Predicates.Side3D(u0, u1, u2, t2);
            if (st0 == st1 && st1 == st2 && st0 != 0) return null;
            int su0 = Predicates.Side3D(t0, t1, t2, u0), su1 = Predicates.Side3D(t0, t1, t2, u1), su2 = Predicates.Side3D(t0, t1, t2, u2);
            if (su0 == su1 && su1 == su2 && su0 != 0) return null;
            if (st0 == 0 && st1 == 0 && st2 == 0) return Coplanar(t, u);

            var a = new List<PPoint>(2);
            var b = new List<PPoint>(2);
            PlaneCut(t, u, new[] { st0, st1, st2 }, a);
            PlaneCut(u, t, new[] { su0, su1, su2 }, b);
            if (a.Count < 2 || b.Count < 2) return null; // touches the other plane in at most one point

            V3 d = Cross(_normal[t], _normal[u]);
            double ta0 = V3.Dot(a[0].Pos, d), ta1 = V3.Dot(a[1].Pos, d), tb0 = V3.Dot(b[0].Pos, d), tb1 = V3.Dot(b[1].Pos, d);
            if (ta0 > ta1) { (ta0, ta1) = (ta1, ta0); a.Reverse(); }
            if (tb0 > tb1) { (tb0, tb1) = (tb1, tb0); b.Reverse(); }
            if (Math.Max(ta0, tb0) >= Math.Min(ta1, tb1)) return null;
            PPoint p = ta0 >= tb0 ? a[0] : b[0], pOther = ta0 >= tb0 ? b[0] : a[0];
            PPoint q = ta1 <= tb1 ? a[1] : b[1], qOther = ta1 <= tb1 ? b[1] : a[1];
            if (p.Key.Equals(q.Key)) return null;
            // Near ties between the two triangles' interval ends are decided by rounded parameters; register the losing
            // candidate too when it is within the merge epsilon, so the merge makes both sides use one vertex (with the
            // edge attachments of both, e.g. a cut crossing an original edge exactly).
            List<PPoint> ties = null;
            if ((pOther.Pos - p.Pos).Length <= _eps) (ties = new List<PPoint>()).Add(pOther);
            if ((qOther.Pos - q.Pos).Length <= _eps) (ties ?? (ties = new List<PPoint>())).Add(qOther);
            return new List<SegReq> { new SegReq { P = p, Q = q, F1 = t, F2 = u, LineEdge = -1, Ties = ties?.ToArray() } };
        }

        /// <summary>Points of triangle x on the plane of triangle y (x is not coplanar with y), given exact vertex sides.</summary>
        private void PlaneCut(int x, int y, int[] side, List<PPoint> outPts)
        {
            for (int i = 0; i < 3; i++)
            {
                if (side[i] == 0)
                {
                    int vid = Tv[3 * x + i];
                    var pp = new PPoint { Key = new PKey(KVertex, vid), Pos = Pts[vid], E1 = -1, E2 = -1 };
                    pp.E1 = VertexOnEdgeOf(y, vid);
                    outPts.Add(pp);
                }
            }
            for (int i = 0; i < 3; i++)
            {
                int j = (i + 1) % 3;
                if (side[i] == 0 || side[j] == 0 || side[i] == side[j]) continue;
                int e = _triEdge[3 * x + i];
                outPts.Add(EdgeCrossing(e, y));
            }
        }

        /// <summary>
        /// Point where edge e (strictly crossing y's plane) meets that plane. If it lies exactly on an edge of y the key
        /// is EE (or the vertex of y it passes through), so y's neighbours derive the same key; else EF. Exact: e and an
        /// edge f of y are coplanar iff Side3D == 0, and then the crossing is on segment f iff f's endpoints are not
        /// strictly on one side of line e within that plane.
        /// </summary>
        private PPoint EdgeCrossing(int e, int y)
        {
            V3 ea = Pts[_edgeU[e]], eb = Pts[_edgeV[e]];
            for (int k = 0; k < 3; k++)
            {
                int f = _triEdge[3 * y + k];
                V3 fa = Pts[_edgeU[f]], fb = Pts[_edgeV[f]];
                if (Predicates.Side3D(ea, eb, fa, fb) != 0) continue;
                int axis = DominantAxis(Cross(eb - ea, fb - fa));
                Proj(ea, axis, out double ax, out double ay); Proj(eb, axis, out double bx, out double by);
                Proj(fa, axis, out double cx, out double cy); Proj(fb, axis, out double dx, out double dy);
                int o1 = Predicates.Orient2D(ax, ay, bx, by, cx, cy), o2 = Predicates.Orient2D(ax, ay, bx, by, dx, dy);
                if (o1 == 0) return new PPoint { Key = new PKey(KVertex, _edgeU[f]), Pos = fa, E1 = e, E2 = -1 };
                if (o2 == 0) return new PPoint { Key = new PKey(KVertex, _edgeV[f]), Pos = fb, E1 = e, E2 = -1 };
                if (o1 != o2) return EdgeEdge(e, f);
            }
            return new PPoint { Key = new PKey(KEdgeFace, e, y), Pos = EdgeFacePos(e, y), E1 = e, E2 = -1 };
        }

        /// <summary>Edge of triangle y whose open interior contains vertex vid (vid known to lie in y's plane), else -1.</summary>
        private int VertexOnEdgeOf(int y, int vid)
        {
            int axis = DominantAxis(_normal[y]);
            V3 p = Pts[vid];
            for (int k = 0; k < 3; k++)
            {
                int a = Tv[3 * y + k], b = Tv[3 * y + (k + 1) % 3];
                if (a == vid || b == vid) continue;
                if (OnOpenSegment(Pts[a], Pts[b], p, axis)) return _triEdge[3 * y + k];
            }
            return -1;
        }

        private static bool OnOpenSegment(V3 a, V3 b, V3 p, int axis)
        {
            Proj(a, axis, out double ax, out double ay);
            Proj(b, axis, out double bx, out double by);
            Proj(p, axis, out double px, out double py);
            if (Predicates.Orient2D(ax, ay, bx, by, px, py) != 0) return false;
            bool useX = Math.Abs(bx - ax) >= Math.Abs(by - ay);
            double lo = useX ? Math.Min(ax, bx) : Math.Min(ay, by), hi = useX ? Math.Max(ax, bx) : Math.Max(ay, by), c = useX ? px : py;
            return c > lo && c < hi;
        }

        private List<SegReq> Coplanar(int t, int u)
        {
            int axis = DominantAxis(_normal[t]);
            List<SegReq> list = null;
            ClipEdges(t, u, axis, ref list);
            ClipEdges(u, t, axis, ref list);
            return list;
        }

        /// <summary>Coplanar pair: the pieces of x's edges inside the closed triangle y.</summary>
        private void ClipEdges(int x, int y, int axis, ref List<SegReq> list)
        {
            var yv = new double[6];
            for (int k = 0; k < 3; k++) Proj(Pts[Tv[3 * y + k]], axis, out yv[2 * k], out yv[2 * k + 1]);
            int ys = Predicates.Orient2D(yv[0], yv[1], yv[2], yv[3], yv[4], yv[5]);
            if (ys == 0) return;

            for (int i = 0; i < 3; i++)
            {
                int e = _triEdge[3 * x + i];
                int va = Tv[3 * x + i], vb = Tv[3 * x + (i + 1) % 3];
                Proj(Pts[va], axis, out double ax, out double ay);
                Proj(Pts[vb], axis, out double bx, out double by);
                var cand = new List<PPoint>(4);
                foreach (int v in new[] { va, vb })
                {
                    Proj(Pts[v], axis, out double px, out double py);
                    int inside = 1, onEdge = -1, zeros = 0;
                    for (int k = 0; k < 3 && inside > 0; k++)
                    {
                        int o = Predicates.Orient2D(yv[2 * k], yv[2 * k + 1], yv[2 * ((k + 1) % 3)], yv[2 * ((k + 1) % 3) + 1], px, py) * ys;
                        if (o < 0) inside = 0;
                        else if (o == 0) { zeros++; onEdge = _triEdge[3 * y + k]; }
                    }
                    if (inside > 0)
                        cand.Add(new PPoint { Key = new PKey(KVertex, v), Pos = Pts[v], E1 = zeros == 1 ? onEdge : -1, E2 = -1 });
                }
                for (int k = 0; k < 3; k++)
                {
                    int ya = Tv[3 * y + k], yb = Tv[3 * y + (k + 1) % 3];
                    double cx = yv[2 * k], cy = yv[2 * k + 1], dx = yv[2 * ((k + 1) % 3)], dy = yv[2 * ((k + 1) % 3) + 1];
                    int o1 = Predicates.Orient2D(ax, ay, bx, by, cx, cy), o2 = Predicates.Orient2D(ax, ay, bx, by, dx, dy);
                    int o3 = Predicates.Orient2D(cx, cy, dx, dy, ax, ay), o4 = Predicates.Orient2D(cx, cy, dx, dy, bx, by);
                    if (o1 != 0 && o2 != 0 && o1 != o2 && o3 != 0 && o4 != 0 && o3 != o4)
                        cand.Add(EdgeEdge(e, _triEdge[3 * y + k]));
                    if (ya != va && ya != vb && OnOpenSegment(Pts[va], Pts[vb], Pts[ya], axis))
                        cand.Add(new PPoint { Key = new PKey(KVertex, ya), Pos = Pts[ya], E1 = e, E2 = -1 });
                }
                if (cand.Count < 2) continue;
                V3 dir = Pts[vb] - Pts[va];
                int lo = 0, hi = 0;
                for (int c = 1; c < cand.Count; c++)
                {
                    double tc = V3.Dot(cand[c].Pos - Pts[va], dir);
                    if (tc < V3.Dot(cand[lo].Pos - Pts[va], dir)) lo = c;
                    if (tc > V3.Dot(cand[hi].Pos - Pts[va], dir)) hi = c;
                }
                if (cand[lo].Key.Equals(cand[hi].Key)) continue;
                (list ?? (list = new List<SegReq>())).Add(new SegReq { P = cand[lo], Q = cand[hi], F1 = x, F2 = y, LineEdge = e });
            }
        }

        private PPoint EdgeEdge(int e, int f)
        {
            if (e > f) { int tmp = e; e = f; f = tmp; }
            V3 a = Pts[_edgeU[e]], b = Pts[_edgeV[e]], c = Pts[_edgeU[f]], d = Pts[_edgeV[f]];
            V3 r = b - a, s = d - c, w = Cross(r, s);
            double den = V3.Dot(w, w);
            double t = den > 0 ? V3.Dot(Cross(c - a, s), w) / den : 0.5;
            t = Math.Max(0.0, Math.Min(1.0, t));
            return new PPoint { Key = new PKey(KEdgeEdge, e, f), Pos = a + r * t, E1 = e, E2 = f };
        }

        private V3 EdgeFacePos(int e, int f)
        {
            V3 a = Pts[_edgeU[e]], b = Pts[_edgeV[e]], c0 = Pts[Tv[3 * f]], n = _normal[f];
            double da = V3.Dot(a - c0, n), db = V3.Dot(b - c0, n);
            double t = da / (da - db);
            t = double.IsNaN(t) ? 0.5 : Math.Max(0.0, Math.Min(1.0, t));
            return a + (b - a) * t;
        }

        private int Register(PPoint p)
        {
            int id;
            if (p.Key.Kind == KVertex) id = p.Key.A;
            else if (!_keyToId.TryGetValue(p.Key, out id))
            {
                id = Pts.Count;
                Pts.Add(p.Pos);
                _keyToId[p.Key] = id;
            }
            if (p.E1 >= 0) AttachEdge(p.E1, id);
            if (p.E2 >= 0) AttachEdge(p.E2, id);
            return id;
        }

        private void AttachEdge(int e, int id)
        {
            if (!_edgePoints.TryGetValue(e, out var l)) _edgePoints[e] = l = new List<int>();
            if (!l.Contains(id)) l.Add(id);
        }

        // ================================================================ triple points
        /// <summary>
        /// Cut segments of one triangle that cross each other meet at a point shared by three surfaces. It is given a
        /// canonical key (triple of faces, or EF / EE when one segment is a piece of an original edge) and split into
        /// every segment through it, in every triangle, so all three triangles use the same vertex.
        /// </summary>
        public void SplitCrossings(double diag)
        {
            var found = new List<(int T, int S1, int S2)>[TriCount];
            Parallel.For(0, TriCount, t =>
            {
                var segs = _faceSegs[t];
                if (segs == null || segs.Count < 2) return;
                int axis = DominantAxis(_normal[t]);
                for (int i = 0; i < segs.Count; i++)
                    for (int j = i + 1; j < segs.Count; j++)
                    {
                        Seg a = _segs[segs[i]], b = _segs[segs[j]];
                        if (a.P == b.P || a.P == b.Q || a.Q == b.P || a.Q == b.Q) continue;
                        if (!ProperCross(Pts[a.P], Pts[a.Q], Pts[b.P], Pts[b.Q], axis)) continue;
                        (found[t] ?? (found[t] = new List<(int, int, int)>())).Add((t, segs[i], segs[j]));
                    }
            });

            foreach (var list in found)
            {
                if (list == null) continue;
                foreach (var (t, i, j) in list)
                {
                    Seg s1 = _segs[i], s2 = _segs[j];
                    int o1 = s1.F1 == t ? s1.F2 : s1.F1, o2 = s2.F1 == t ? s2.F2 : s2.F1;
                    PPoint pp;
                    int third = -1;
                    if (s1.LineEdge < 0 && s2.LineEdge < 0)
                    {
                        int[] f = { t, o1, o2 };
                        Array.Sort(f);
                        pp = new PPoint { Key = new PKey(KTriple, f[0], f[1], f[2]), Pos = TriplePos(f, s1, s2, t, diag), E1 = -1, E2 = -1 };
                        if (!_pairSeg.TryGetValue(PairKey(o1, o2), out third)) third = -1;
                    }
                    else if (s1.LineEdge >= 0 && s2.LineEdge >= 0) pp = EdgeEdge(s1.LineEdge, s2.LineEdge);
                    else
                    {
                        int e = s1.LineEdge >= 0 ? s1.LineEdge : s2.LineEdge, face = s1.LineEdge >= 0 ? o2 : o1;
                        pp = new PPoint { Key = new PKey(KEdgeFace, e, face), Pos = EdgeFacePos(e, face), E1 = e, E2 = -1 };
                    }
                    int id = Register(pp);
                    AddSplit(s1, id); AddSplit(s2, id);
                    if (third >= 0) AddSplit(_segs[third], id);
                }
            }
        }

        private static void AddSplit(Seg s, int id)
        {
            if (s.P == id || s.Q == id) return;
            if (s.Splits == null) s.Splits = new List<int>();
            if (!s.Splits.Contains(id)) s.Splits.Add(id);
        }

        private V3 TriplePos(int[] f, Seg s1, Seg s2, int t, double diag)
        {
            // Crossing of the two segments in t's projection, lifted along s1 (fallback and sanity reference).
            int axis = DominantAxis(_normal[t]);
            Proj(Pts[s1.P], axis, out double ax, out double ay); Proj(Pts[s1.Q], axis, out double bx, out double by);
            Proj(Pts[s2.P], axis, out double cx, out double cy); Proj(Pts[s2.Q], axis, out double dx, out double dy);
            double rx = bx - ax, ry = by - ay, sx = dx - cx, sy = dy - cy;
            double tt = ((cx - ax) * sy - (cy - ay) * sx) / (rx * sy - ry * sx);
            tt = double.IsNaN(tt) ? 0.5 : Math.Max(0.0, Math.Min(1.0, tt));
            V3 cross = Pts[s1.P] + (Pts[s1.Q] - Pts[s1.P]) * tt;

            V3 o = Pts[Tv[3 * f[0]]];
            V3 n1 = _normal[f[0]], n2 = _normal[f[1]], n3 = _normal[f[2]];
            double d1 = V3.Dot(n1, Pts[Tv[3 * f[0]]] - o), d2 = V3.Dot(n2, Pts[Tv[3 * f[1]]] - o), d3 = V3.Dot(n3, Pts[Tv[3 * f[2]]] - o);
            V3 c23 = Cross(n2, n3), c31 = Cross(n3, n1), c12 = Cross(n1, n2);
            double det = V3.Dot(n1, c23);
            if (Math.Abs(det) <= 1e-10 * n1.Length * n2.Length * n3.Length) return cross;
            V3 x = o + (c23 * d1 + c31 * d2 + c12 * d3) * (1.0 / det);
            return (x - cross).Length <= 1e-6 * diag ? x : cross;
        }

        // ================================================================ merge + re-triangulation
        /// <summary>Union-find merge of all points closer than eps; the representative is the smallest id.</summary>
        public void Merge(double eps)
        {
            int n = Pts.Count;
            Rep = new int[n];
            for (int i = 0; i < n; i++) Rep[i] = i;
            double cell = 2 * eps, epsSq = eps * eps;
            var grid = new Dictionary<(long, long, long), List<int>>(KeyComparer.Instance);
            for (int i = 0; i < n; i++)
            {
                V3 p = Pts[i];
                long cx = (long)Math.Floor(p.X / cell), cy = (long)Math.Floor(p.Y / cell), cz = (long)Math.Floor(p.Z / cell);
                for (long dx = -1; dx <= 1; dx++)
                    for (long dy = -1; dy <= 1; dy++)
                        for (long dz = -1; dz <= 1; dz++)
                            if (grid.TryGetValue((cx + dx, cy + dy, cz + dz), out var l))
                                foreach (int j in l)
                                {
                                    V3 d = Pts[j] - p;
                                    if (V3.Dot(d, d) <= epsSq) Union(i, j);
                                }
                var key = (cx, cy, cz);
                if (!grid.TryGetValue(key, out var list)) grid[key] = list = new List<int>();
                list.Add(i);
            }
            for (int i = 0; i < n; i++) Rep[i] = Find(i);
        }

        private int Find(int i)
        {
            while (Rep[i] != i) { Rep[i] = Rep[Rep[i]]; i = Rep[i]; }
            return i;
        }

        private void Union(int a, int b)
        {
            a = Find(a); b = Find(b);
            if (a == b) return;
            if (a < b) Rep[b] = a; else Rep[a] = b;
        }

        /// <summary>Re-triangulates every triangle (Parallel.For) and assembles sub-triangles over the global vertex table.</summary>
        public void Triangulate()
        {
            // Segment chains (endpoints + split points, ordered along the segment, as representatives).
            foreach (var s in _segs)
            {
                var ids = new List<int> { s.P };
                if (s.Splits != null)
                {
                    V3 a = Pts[s.P], dir = Pts[s.Q] - a;
                    var sorted = new List<int>(s.Splits);
                    sorted.Sort((x, y) => V3.Dot(Pts[x] - a, dir).CompareTo(V3.Dot(Pts[y] - a, dir)));
                    ids.AddRange(sorted);
                }
                ids.Add(s.Q);
                var chain = new List<int>();
                foreach (int id in ids)
                {
                    int r = Rep[id];
                    if (chain.Count == 0 || chain[chain.Count - 1] != r) chain.Add(r);
                }
                s.Chain = chain.ToArray();
            }

            // Points on each original edge, as representatives ordered from _edgeU to _edgeV.
            var edgeLists = new Dictionary<int, List<int>>();
            foreach (var kv in _edgePoints)
            {
                int e = kv.Key, ru = Rep[_edgeU[e]], rv = Rep[_edgeV[e]];
                var l = new List<int>();
                foreach (int id in kv.Value)
                {
                    int r = Rep[id];
                    if (r != ru && r != rv && !l.Contains(r)) l.Add(r);
                }
                if (l.Count == 0) continue;
                V3 a = Pts[_edgeU[e]], dir = Pts[_edgeV[e]] - a;
                l.Sort((x, y) => V3.Dot(Pts[x] - a, dir).CompareTo(V3.Dot(Pts[y] - a, dir)));
                edgeLists[e] = l;
            }

            var results = new Arrangement[TriCount];
            var dropped = new bool[TriCount];
            Parallel.For(0, TriCount, g =>
            {
                int[] c = { Rep[Tv[3 * g]], Rep[Tv[3 * g + 1]], Rep[Tv[3 * g + 2]] };
                if (c[0] == c[1] || c[1] == c[2] || c[2] == c[0]) return; // collapsed by the merge: nothing to emit
                var edgePts = new List<int>[3];
                bool needed = _faceSegs[g] != null;
                for (int e = 0; e < 3; e++)
                {
                    if (!edgeLists.TryGetValue(_triEdge[3 * g + e], out var l)) continue;
                    bool forward = Tv[3 * g + e] == _edgeU[_triEdge[3 * g + e]];
                    edgePts[e] = forward ? l : Reversed(l);
                    needed = true;
                }
                if (!needed) return;
                var cons = new List<(int, int)>();
                if (_faceSegs[g] != null)
                    foreach (int si in _faceSegs[g])
                    {
                        var ch = _segs[si].Chain;
                        for (int k = 0; k + 1 < ch.Length; k++) cons.Add((ch[k], ch[k + 1]));
                    }
                results[g] = Arrangement.Build(c, Pts, edgePts, cons);
                if (results[g] == null) dropped[g] = true;
            });

            for (int g = 0; g < TriCount; g++)
            {
                int a = Rep[Tv[3 * g]], b = Rep[Tv[3 * g + 1]], c = Rep[Tv[3 * g + 2]];
                if (dropped[g]) { DroppedTriangles++; continue; }
                if (a == b || b == c || c == a) continue;
                var r = results[g];
                if (r == null)
                {
                    SubTris.Add(a); SubTris.Add(b); SubTris.Add(c);
                    SubOwner.Add(TriOwner[g]);
                    continue;
                }
                Degenerate += r.Degenerate;
                int baseId = Pts.Count;
                Pts.AddRange(r.NewPoints);
                int Map(int id) => id >= 0 ? id : baseId + (-id - 1);
                for (int k = 0; k < r.Triangles.Count; k += 3)
                {
                    SubTris.Add(Map(r.Triangles[k])); SubTris.Add(Map(r.Triangles[k + 1])); SubTris.Add(Map(r.Triangles[k + 2]));
                    SubOwner.Add(TriOwner[g]);
                }
                foreach (var (u, v) in r.Constrained) CutEdges.Add(UndirectedKey(Map(u), Map(v)));
            }
        }

        private static List<int> Reversed(List<int> l)
        {
            var r = new List<int>(l);
            r.Reverse();
            return r;
        }

        // ================================================================ helpers
        public static long UndirectedKey(int a, int b) => a < b ? ((long)a << 32) | (uint)b : ((long)b << 32) | (uint)a;
        private static long PairKey(int a, int b) => UndirectedKey(a, b);

        private static int DominantAxis(V3 n)
        {
            double ax = Math.Abs(n.X), ay = Math.Abs(n.Y), az = Math.Abs(n.Z);
            return ax >= ay && ax >= az ? 0 : ay >= az ? 1 : 2;
        }

        private static void Proj(V3 p, int axis, out double u, out double v)
        {
            u = axis == 0 ? p.Y : axis == 1 ? p.Z : p.X;
            v = axis == 0 ? p.Z : axis == 1 ? p.X : p.Y;
        }

        private static bool ProperCross(V3 a, V3 b, V3 c, V3 d, int axis)
        {
            Proj(a, axis, out double ax, out double ay); Proj(b, axis, out double bx, out double by);
            Proj(c, axis, out double cx, out double cy); Proj(d, axis, out double dx, out double dy);
            int o1 = Predicates.Orient2D(ax, ay, bx, by, cx, cy), o2 = Predicates.Orient2D(ax, ay, bx, by, dx, dy);
            if (o1 == 0 || o2 == 0 || o1 == o2) return false;
            int o3 = Predicates.Orient2D(cx, cy, dx, dy, ax, ay), o4 = Predicates.Orient2D(cx, cy, dx, dy, bx, by);
            return o3 != 0 && o4 != 0 && o3 != o4;
        }

        private static V3 Cross(V3 a, V3 b) => new V3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    }
}
