using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    /// <summary>
    /// M3: splits a KBrep along SSI branches (Brep.Split semantics: returns the connected pieces).
    /// Per affected face: UV arrangement of the existing trims and the branch pullbacks (polylines for topology,
    /// Newton refinement on the exact NURBS for every vertex), DCEL cycle tracing (face on the left), material /
    /// outer / hole classification, hole containment. Edge splits are made globally consistent across faces
    /// (seam twins and neighbour faces) before the final arrangement. New faces reuse the surface; trims are exact
    /// sub-curves of the original trims / branch pullbacks; new edges are exact sub-curves of the SSI 3D curves.
    /// Pieces = faces connected through original, non-cut edges.
    /// </summary>
    public static class FaceSplit
    {
        public static List<KBrep> Split(KBrep brep, IReadOnlyList<SsiBranch> branches, bool branchesOnA, double tolerance)
            => Split(brep, branches, branchesOnA, tolerance, null);

        /// <summary>Same as <see cref="Split(KBrep, IReadOnlyList{SsiBranch}, bool, double)"/>; diagnostics go to warnings.</summary>
        public static List<KBrep> Split(KBrep brep, IReadOnlyList<SsiBranch> branches, bool branchesOnA, double tolerance, List<string> warnings)
        {
            if (brep == null) throw new ArgumentNullException(nameof(brep));
            if (branches == null) throw new ArgumentNullException(nameof(branches));
            if (!(tolerance > 0)) throw new ArgumentOutOfRangeException(nameof(tolerance));
            var w = warnings ?? new List<string>();
            return new Ctx(brep, tolerance, w).Run(branches, branchesOnA);
        }

        // ------------------------------------------------------------------ per-face arrangement types

        struct SplitPt { public double T; public int V; }

        sealed class Arc
        {
            public bool IsTrim; public int Trim = -1, Branch = -1;
            public NurbsCurve C2; public double T0, T1;
            public double[] PT; public V3[] PUv; public V3 Min, Max;
            public List<SplitPt> Splits = new List<SplitPt>();
            public V3 At(double t) => NurbsEval.CurvePoint(C2, t);
        }

        sealed class Vtx { public V3 Uv, X; public int Global = -1, Parent; public bool OnTrim; }

        sealed class Piece
        {
            public Arc Arc; public double Ta, Tb; public int Va, Vb; public bool Cut, Dead;
            public int Edge = -1; public bool EdgeRev;
        }

        sealed class HalfEdge
        {
            public Piece P; public bool Fwd; public int From, To; public double Ang, Ang2; public bool Used; public HalfEdge Twin;
        }

        sealed class FaceJob
        {
            public int Face; public NurbsSurface S; public double Guard;
            public List<Arc> Arcs = new List<Arc>();
            public List<Vtx> Verts = new List<Vtx>();
            public List<(int trim, double s, V3 x)> Required = new List<(int, double, V3)>();
            public List<Piece> Pieces = new List<Piece>();
            public List<List<List<HalfEdge>>> Regions = new List<List<List<HalfEdge>>>();
            public List<List<V3>> OrigLoops = new List<List<V3>>();
            public List<string> Warn = new List<string>();
        }

        sealed class EdgeSplit { public List<(double e, int gv)> Pts = new List<(double, int)>(); }

        // ------------------------------------------------------------------ context

        sealed class Ctx
        {
            readonly KBrep B; readonly double Tol; readonly List<string> W;
            readonly List<KVertex> GV = new List<KVertex>();
            readonly Dictionary<int, EdgeSplit> EdgeSplits = new Dictionary<int, EdgeSplit>();
            Dictionary<int, List<(double s, int gv)>> TrimSplits = new Dictionary<int, List<(double, int)>>();
            IReadOnlyList<SsiBranch> Branches; bool OnA;
            readonly int[] TrimFace;

            public Ctx(KBrep b, double tol, List<string> w)
            {
                B = b; Tol = tol; W = w;
                foreach (var v in b.Vertices) GV.Add(new KVertex { Point = v.Point, Tolerance = v.Tolerance });
                TrimFace = new int[b.Trims.Count];
                for (int i = 0; i < b.Trims.Count; i++) TrimFace[i] = b.Trims[i].Loop >= 0 ? b.Loops[b.Trims[i].Loop].Face : -1;
            }

            NurbsCurve Uv(SsiBranch br) => OnA ? br.UvA : br.UvB;

            public List<KBrep> Run(IReadOnlyList<SsiBranch> branches, bool onA)
            {
                Branches = branches; OnA = onA;
                var byFace = new Dictionary<int, List<int>>();
                for (int i = 0; i < branches.Count; i++)
                {
                    var br = branches[i];
                    int f = onA ? br.FaceA : br.FaceB;
                    if (f < 0 || f >= B.Faces.Count || Uv(br) == null || br.Curve3 == null) { W.Add($"branch {i}: no face/pullback on this side, ignored"); continue; }
                    if (!byFace.TryGetValue(f, out var l)) byFace[f] = l = new List<int>();
                    l.Add(i);
                }

                // Iterate: arrangement per face -> required trim splits -> global edge splits -> re-run affected faces.
                var jobs = new Dictionary<int, FaceJob>();
                var todo = new HashSet<int>(byFace.Keys);
                for (int round = 0; round < 4 && todo.Count > 0; round++)
                {
                    var list = todo.ToList();
                    var res = new FaceJob[list.Count];
                    Parallel.For(0, list.Count, i =>
                    {
                        byFace.TryGetValue(list[i], out var bl);
                        res[i] = BuildFace(list[i], bl ?? new List<int>());
                    });
                    var required = new List<(int trim, double s, V3 x)>();
                    todo.Clear();
                    for (int i = 0; i < list.Count; i++)
                    {
                        jobs[list[i]] = res[i];
                        if (res[i].Required.Count > 0) { required.AddRange(res[i].Required); todo.Add(list[i]); }
                    }
                    if (required.Count == 0) break;
                    foreach (var e in MergeRequired(required)) foreach (var t in B.Edges[e].Trims) if (TrimFace[t] >= 0) todo.Add(TrimFace[t]);
                    RebuildTrimSplits();
                    if (round == 3) W.Add("edge split iteration did not converge");
                }
                foreach (var j in jobs.Values)
                {
                    W.AddRange(j.Warn.Select(s => $"face {j.Face}: {s}"));
                    if (j.Required.Count > 0) throw new InvalidOperationException($"face {j.Face}: inconsistent edge splits");
                }
                return Assemble(jobs);
            }

            // ---------------------------------------------------------- phase B: global edge splits

            IEnumerable<int> MergeRequired(List<(int trim, double s, V3 x)> req)
            {
                var touched = new HashSet<int>();
                foreach (var (ti, s, x) in req)
                {
                    int ei = B.Trims[ti].Edge;
                    if (ei < 0) { W.Add($"split requested on edgeless trim {ti}"); continue; }
                    var E = B.Edges[ei]; var c3 = B.Curves3[E.Curve3];
                    double e = SplitGeom.ClosestOnCurve(c3, x, E.Domain.T0, E.Domain.T1);
                    var xe = NurbsEval.CurvePoint(c3, e);
                    if (SplitGeom.Dist(xe, x) > Tol) W.Add($"edge {ei}: split point {SplitGeom.Dist(xe, x):E2} off the edge");
                    if (SplitGeom.Dist(xe, GV[E.V0].Point) <= Tol || SplitGeom.Dist(xe, GV[E.V1].Point) <= Tol) continue;
                    if (!EdgeSplits.TryGetValue(ei, out var es)) EdgeSplits[ei] = es = new EdgeSplit();
                    bool dup = false;
                    foreach (var p in es.Pts) if (SplitGeom.Dist(GV[p.gv].Point, xe) <= Tol) { dup = true; break; }
                    if (dup) continue;
                    GV.Add(new KVertex { Point = xe, Tolerance = 0 });
                    es.Pts.Add((e, GV.Count - 1));
                    touched.Add(ei);
                }
                foreach (var es in EdgeSplits.Values) es.Pts.Sort((a, b) => a.e.CompareTo(b.e));
                return touched;
            }

            void RebuildTrimSplits()
            {
                TrimSplits = new Dictionary<int, List<(double, int)>>();
                foreach (var kv in EdgeSplits)
                {
                    var E = B.Edges[kv.Key];
                    foreach (var ti in E.Trims)
                    {
                        var T = B.Trims[ti]; var c2 = B.Curves2[T.Curve2];
                        var srf = B.Surfaces[B.Faces[TrimFace[ti]].Surface];
                        var l = new List<(double, int)>();
                        foreach (var (e, gv) in kv.Value.Pts)
                        {
                            var x = GV[gv].Point;
                            double fr = (e - E.Domain.T0) / E.Domain.Length; if (T.Reversed3d) fr = 1 - fr;
                            double s = T.Domain.ParameterAt(fr);
                            double d = SplitGeom.ProjectCos(srf, c2, T.Domain.T0, T.Domain.T1, x, ref s);
                            if (d > Tol)
                            {   // global restart from the closest polyline sample
                                SplitGeom.Sample(c2, T.Domain.T0, T.Domain.T1, out var ts, out var ps);
                                double bd = double.MaxValue;
                                for (int i = 0; i < ts.Length; i++) { double dd = SplitGeom.Dist(SplitGeom.Eval(srf, ps[i]), x); if (dd < bd) { bd = dd; s = ts[i]; } }
                                d = SplitGeom.ProjectCos(srf, c2, T.Domain.T0, T.Domain.T1, x, ref s);
                            }
                            if (d > Tol) W.Add($"trim {ti}: edge split point {d:E2} off the trim");
                            l.Add((s, gv));
                        }
                        TrimSplits[ti] = l;
                    }
                }
            }

            // ---------------------------------------------------------- phase A/C: face arrangement

            static int Find(FaceJob j, int v) { while (j.Verts[v].Parent != v) { j.Verts[v].Parent = j.Verts[j.Verts[v].Parent].Parent; v = j.Verts[v].Parent; } return v; }

            static int Prio(Vtx v) => v.Global >= 0 ? 2 : v.OnTrim ? 1 : 0;

            void Union(FaceJob j, int a, int b)
            {
                a = Find(j, a); b = Find(j, b);
                if (a == b) return;
                var va = j.Verts[a]; var vb = j.Verts[b];
                if (va.Global >= 0 && vb.Global >= 0 && va.Global != vb.Global)
                    j.Warn.Add($"merging distinct vertices {va.Global}/{vb.Global} ({SplitGeom.Dist(va.X, vb.X):E2})");
                if (Prio(vb) > Prio(va)) { var t = a; a = b; b = t; }
                j.Verts[b].Parent = a;
            }

            int NewVertex(FaceJob j, V3 uv, int global, bool onTrim)
            {
                j.Verts.Add(new Vtx { Uv = uv, X = SplitGeom.Eval(j.S, uv), Global = global, Parent = j.Verts.Count, OnTrim = onTrim });
                return j.Verts.Count - 1;
            }

            /// <summary>Splits arc a at s, reusing an existing split within tolerance. hint &gt;= 0 forces that vertex.</summary>
            int SplitAt(FaceJob j, Arc a, double s, int hint)
            {
                var uv = a.At(s); var x = SplitGeom.Eval(j.S, uv);
                foreach (var sp in a.Splits)
                {
                    var xs = SplitGeom.Eval(j.S, a.At(sp.T));
                    if (SplitGeom.Dist(xs, x) > Tol) continue;
                    if (SplitGeom.Dist(SplitGeom.Eval(j.S, a.At(0.5 * (s + sp.T))), x) > Tol) continue;
                    if (hint >= 0) Union(j, hint, sp.V);
                    return Find(j, sp.V);
                }
                int v = hint >= 0 ? hint : NewVertex(j, uv, -1, a.IsTrim);
                a.Splits.Add(new SplitPt { T = s, V = v });
                if (a.IsTrim) j.Required.Add((a.Trim, s, x));
                return v;
            }

            /// <summary>Closest point on arc q to the 3D point x (UV hint uv). Returns 3D distance or +inf when the UV guard fails.</summary>
            double Project(FaceJob j, Arc q, V3 uv, V3 x, out double s)
            {
                var d = NurbsEval.SurfaceDerivatives(j.S, uv.X, uv.Y, 1);
                V3 Lin(V3 p) => d[1, 0] * (p.X - uv.X) + d[0, 1] * (p.Y - uv.Y);
                double bd = double.MaxValue; s = q.T0;
                for (int i = 0; i + 1 < q.PT.Length; i++)
                {
                    var a = Lin(q.PUv[i]); var ab = Lin(q.PUv[i + 1]) - a;
                    double L2 = V3.Dot(ab, ab);
                    double lam = L2 > 0 ? Math.Max(0, Math.Min(1, -V3.Dot(a, ab) / L2)) : 0;
                    double dd = (a + ab * lam).Length;
                    if (dd < bd) { bd = dd; s = q.PT[i] + lam * (q.PT[i + 1] - q.PT[i]); }
                }
                double dist = SplitGeom.ProjectCos(j.S, q.C2, q.T0, q.T1, x, ref s);
                if (SplitGeom.Dist(q.At(s), uv) > j.Guard) return double.PositiveInfinity;
                return dist;
            }

            FaceJob BuildFace(int fi, List<int> branchIdx)
            {
                var F = B.Faces[fi];
                var j = new FaceJob { Face = fi, S = B.Surfaces[F.Surface] };
                j.Guard = 0.25 * Math.Min(j.S.DomainU.Length, j.S.DomainV.Length);

                // existing loops
                foreach (var li in F.Loops)
                {
                    var L = B.Loops[li];
                    var arcs = new List<Arc>();
                    foreach (var ti in L.Trims)
                    {
                        var T = B.Trims[ti];
                        arcs.Add(new Arc { IsTrim = true, Trim = ti, C2 = B.Curves2[T.Curve2], T0 = T.Domain.T0, T1 = T.Domain.T1 });
                    }
                    int n = arcs.Count;
                    var sv = new int[n];
                    for (int k = 0; k < n; k++)
                    {
                        int g = StartVertex(L.Trims[k]);
                        if (g < 0) g = EndVertex(L.Trims[(k - 1 + n) % n]);
                        sv[k] = NewVertex(j, arcs[k].At(arcs[k].T0), g, true);
                    }
                    var poly = new List<V3>();
                    for (int k = 0; k < n; k++)
                    {
                        var a = arcs[k];
                        a.Splits.Add(new SplitPt { T = a.T0, V = sv[k] });
                        a.Splits.Add(new SplitPt { T = a.T1, V = sv[(k + 1) % n] });
                        if (TrimSplits.TryGetValue(a.Trim, out var ts))
                            foreach (var (s, gv) in ts) a.Splits.Add(new SplitPt { T = s, V = NewVertex(j, a.At(s), gv, true) });
                        Poly(a);
                        for (int i = 0; i + 1 < a.PUv.Length; i++) poly.Add(a.PUv[i]);
                        j.Arcs.Add(a);
                    }
                    j.OrigLoops.Add(poly);
                }
                int nTrimArcs = j.Arcs.Count;

                // branch pullbacks, split at periodic seams
                foreach (var bi in branchIdx)
                {
                    foreach (var c in SeamSplit(Uv(Branches[bi]), j.S))
                    {
                        var a = new Arc { Branch = bi, C2 = c, T0 = c.Domain.T0, T1 = c.Domain.T1 };
                        Poly(a);
                        double len = 0;
                        for (int i = 0; i + 1 < a.PUv.Length; i++) len += SplitGeom.Dist(SplitGeom.Eval(j.S, a.PUv[i]), SplitGeom.Eval(j.S, a.PUv[i + 1]));
                        if (len <= Tol) continue;
                        j.Arcs.Add(a);
                    }
                }
                var newArcs = j.Arcs.Skip(nTrimArcs).ToList();

                // (a) new-arc end points: snap to vertices, else T-junction on an arc, else free vertex
                foreach (var a in newArcs)
                    foreach (var t in new[] { a.T0, a.T1 })
                    {
                        var uv = a.At(t); var x = SplitGeom.Eval(j.S, uv);
                        int bestV = -1; double bd = double.MaxValue;
                        for (int v = 0; v < j.Verts.Count; v++)
                        {
                            if (Find(j, v) != v) continue;
                            double d = SplitGeom.Dist(j.Verts[v].X, x);
                            if (d <= Tol && d < bd && SplitGeom.Dist(j.Verts[v].Uv, uv) <= j.Guard) { bd = d; bestV = v; }
                        }
                        if (bestV < 0)
                        {
                            Arc bq = null; double bs = 0; bd = double.MaxValue;
                            foreach (var q in j.Arcs)
                            {
                                if (q == a) continue;
                                double d = Project(j, q, uv, x, out double s);
                                if (d <= Tol && d < bd) { bd = d; bq = q; bs = s; }
                            }
                            bestV = bq != null ? SplitAt(j, bq, bs, -1) : NewVertex(j, uv, -1, false);
                        }
                        a.Splits.Add(new SplitPt { T = t, V = bestV });
                    }

                // (b) existing vertices lying on new-arc interiors
                for (int v = 0; v < j.Verts.Count; v++)
                {
                    if (Find(j, v) != v) continue;
                    foreach (var a in newArcs)
                    {
                        if (a.Splits.Any(sp => Find(j, sp.V) == v)) continue;
                        double d = Project(j, a, j.Verts[v].Uv, j.Verts[v].X, out double s);
                        if (d <= Tol) SplitAt(j, a, s, v);
                    }
                }

                // (c) transversal crossings
                for (int ia = 0; ia < newArcs.Count; ia++)
                {
                    var a = newArcs[ia];
                    foreach (var q in j.Arcs)
                    {
                        if (q == a || (!q.IsTrim && newArcs.IndexOf(q) < ia)) continue;
                        foreach (var (sa, sq) in Crossings(a, q))
                        {
                            if (q.IsTrim) { int v = SplitAt(j, q, sq, -1); SplitAt(j, a, sa, v); }
                            else { int v = SplitAt(j, a, sa, -1); SplitAt(j, q, sq, v); }
                        }
                    }
                }

                // (d) merge coincident vertices
                for (int a = 0; a < j.Verts.Count; a++)
                    for (int b = a + 1; b < j.Verts.Count; b++)
                    {
                        int ra = Find(j, a), rb = Find(j, b);
                        if (ra == rb) continue;
                        if (SplitGeom.Dist(j.Verts[ra].X, j.Verts[rb].X) <= Tol && SplitGeom.Dist(j.Verts[ra].Uv, j.Verts[rb].Uv) <= j.Guard)
                            Union(j, ra, rb);
                    }

                if (j.Required.Count > 0) return j; // edge splits must be made global first

                BuildPieces(j);
                TraceRegions(j);
                return j;
            }

            int StartVertex(int ti) { var T = B.Trims[ti]; if (T.Edge < 0) return -1; var E = B.Edges[T.Edge]; return T.Reversed3d ? E.V1 : E.V0; }
            int EndVertex(int ti) { var T = B.Trims[ti]; if (T.Edge < 0) return -1; var E = B.Edges[T.Edge]; return T.Reversed3d ? E.V0 : E.V1; }

            static void Poly(Arc a)
            {
                SplitGeom.Sample(a.C2, a.T0, a.T1, out a.PT, out a.PUv);
                double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
                foreach (var p in a.PUv) { x0 = Math.Min(x0, p.X); y0 = Math.Min(y0, p.Y); x1 = Math.Max(x1, p.X); y1 = Math.Max(y1, p.Y); }
                a.Min = new V3(x0, y0, 0); a.Max = new V3(x1, y1, 0);
            }

            /// <summary>Splits a pullback where it crosses a periodic seam line and shifts each piece into the domain.</summary>
            static List<NurbsCurve> SeamSplit(NurbsCurve c, NurbsSurface s)
            {
                var res = new List<NurbsCurve>();
                var dom = c.Domain;
                if (!s.IsClosedU && !s.IsClosedV) { res.Add(c); return res; }
                SplitGeom.Sample(c, dom.T0, dom.T1, out var ts, out var ps);
                var cuts = new List<double>();
                for (int dir = 0; dir < 2; dir++)
                {
                    if (dir == 0 ? !s.IsClosedU : !s.IsClosedV) continue;
                    var sd = dir == 0 ? s.DomainU : s.DomainV; double P = sd.Length;
                    double Co(V3 p) => dir == 0 ? p.X : p.Y;
                    for (int i = 0; i + 1 < ts.Length; i++)
                    {
                        double k0 = Math.Floor((Co(ps[i]) - sd.T0) / P), k1 = Math.Floor((Co(ps[i + 1]) - sd.T0) / P);
                        if (k0 == k1) continue;
                        // A non-finite or runaway pullback (e.g. at a pole) would make K++ a no-op and loop forever.
                        if (double.IsNaN(k0) || double.IsNaN(k1) || double.IsInfinity(k0) || double.IsInfinity(k1) || Math.Abs(k1 - k0) > 64)
                            throw new InvalidOperationException($"seam crossing: invalid pullback at sample {i} (k {k0}..{k1})");
                        for (double K = Math.Min(k0, k1) + 1; K <= Math.Max(k0, k1); K++)
                        {
                            double line = sd.T0 + K * P, a = ts[i], b = ts[i + 1];
                            double fa = Co(ps[i]) - line;
                            for (int it = 0; it < 80; it++)
                            {
                                double m = 0.5 * (a + b), fm = Co(NurbsEval.CurvePoint(c, m)) - line;
                                if ((fm < 0) == (fa < 0)) { a = m; fa = fm; } else b = m;
                            }
                            cuts.Add(0.5 * (a + b));
                        }
                    }
                }
                cuts.Sort();
                // Cuts closer than eps to an existing break collapse inside SubCurve's knot snapping; skip them.
                double cutEps = 1e-9 * Math.Max(1.0, Math.Abs(dom.T0) + Math.Abs(dom.T1));
                var br = new List<double> { dom.T0 };
                foreach (var t in cuts) if (t > br[br.Count - 1] + cutEps && t < dom.T1 - cutEps) br.Add(t);
                br.Add(dom.T1);
                for (int i = 0; i + 1 < br.Count; i++)
                {
                    double a = br[i], b = br[i + 1];
                    if (!(b > a)) continue;
                    var piece = (a == dom.T0 && b == dom.T1) ? c : SplitGeom.SubCurve(c, a, b);
                    var m = NurbsEval.CurvePoint(c, 0.5 * (a + b));
                    double du = s.IsClosedU ? -Math.Floor((m.X - s.DomainU.T0) / s.DomainU.Length) * s.DomainU.Length : 0;
                    double dv = s.IsClosedV ? -Math.Floor((m.Y - s.DomainV.T0) / s.DomainV.Length) * s.DomainV.Length : 0;
                    res.Add(SplitGeom.Translate(piece, du, dv));
                }
                return res;
            }

            /// <summary>Transversal crossings of two arcs: polyline segment intersections refined by 2D Newton.</summary>
            static List<(double, double)> Crossings(Arc a, Arc q)
            {
                var r = new List<(double, double)>();
                double m = 1e-9 * (1 + Math.Abs(a.Max.X) + Math.Abs(a.Max.Y));
                if (a.Max.X < q.Min.X - m || q.Max.X < a.Min.X - m || a.Max.Y < q.Min.Y - m || q.Max.Y < a.Min.Y - m) return r;
                for (int i = 0; i + 1 < a.PT.Length; i++)
                {
                    var p0 = a.PUv[i]; var p1 = a.PUv[i + 1];
                    double ax0 = Math.Min(p0.X, p1.X), ax1 = Math.Max(p0.X, p1.X), ay0 = Math.Min(p0.Y, p1.Y), ay1 = Math.Max(p0.Y, p1.Y);
                    for (int k = 0; k + 1 < q.PT.Length; k++)
                    {
                        var q0 = q.PUv[k]; var q1 = q.PUv[k + 1];
                        if (Math.Max(q0.X, q1.X) < ax0 - m || Math.Min(q0.X, q1.X) > ax1 + m || Math.Max(q0.Y, q1.Y) < ay0 - m || Math.Min(q0.Y, q1.Y) > ay1 + m) continue;
                        double dx = p1.X - p0.X, dy = p1.Y - p0.Y, ex = q1.X - q0.X, ey = q1.Y - q0.Y;
                        double den = dx * ey - dy * ex;
                        double nn = Math.Sqrt((dx * dx + dy * dy) * (ex * ex + ey * ey));
                        if (Math.Abs(den) <= 1e-10 * nn || nn == 0) continue;
                        double fx = q0.X - p0.X, fy = q0.Y - p0.Y;
                        double lam = (fx * ey - fy * ex) / den, mu = (fx * dy - fy * dx) / den;
                        if (lam < -1e-9 || lam > 1 + 1e-9 || mu < -1e-9 || mu > 1 + 1e-9) continue;
                        double s = a.PT[i] + lam * (a.PT[i + 1] - a.PT[i]), t = q.PT[k] + mu * (q.PT[k + 1] - q.PT[k]);
                        if (!Newton2(a, q, ref s, ref t)) continue;
                        bool dup = false;
                        foreach (var (s0, t0) in r) if (Math.Abs(s0 - s) <= 1e-9 * (a.T1 - a.T0) && Math.Abs(t0 - t) <= 1e-9 * (q.T1 - q.T0)) { dup = true; break; }
                        if (!dup) r.Add((s, t));
                    }
                }
                return r;
            }

            static bool Newton2(Arc a, Arc q, ref double s, ref double t)
            {
                for (int it = 0; it < 40; it++)
                {
                    var da = NurbsEval.CurveDerivatives(a.C2, s, 1); var dq = NurbsEval.CurveDerivatives(q.C2, t, 1);
                    double fx = da[0].X - dq[0].X, fy = da[0].Y - dq[0].Y;
                    double j11 = da[1].X, j12 = -dq[1].X, j21 = da[1].Y, j22 = -dq[1].Y;
                    double det = j11 * j22 - j12 * j21;
                    double na = Math.Sqrt(j11 * j11 + j21 * j21), nq = Math.Sqrt(j12 * j12 + j22 * j22);
                    if (Math.Abs(det) <= 1e-8 * na * nq || na == 0 || nq == 0) return false; // tangential: handled by projections
                    double ds = (j22 * fx - j12 * fy) / det, dt = (-j21 * fx + j11 * fy) / det;
                    s = Math.Max(a.T0, Math.Min(a.T1, s - ds)); t = Math.Max(q.T0, Math.Min(q.T1, t - dt));
                    if (Math.Abs(ds) <= 1e-15 * (1 + Math.Abs(s)) && Math.Abs(dt) <= 1e-15 * (1 + Math.Abs(t))) break;
                }
                var pa = a.At(s); var pq = q.At(t);
                double scale = 1 + Math.Abs(pa.X) + Math.Abs(pa.Y);
                return SplitGeom.Dist(pa, pq) <= 1e-10 * scale;
            }

            double Len3(FaceJob j, Arc a, double t0, double t1)
            {
                double L = 0; var prev = SplitGeom.Eval(j.S, a.At(t0));
                for (int i = 1; i <= 8; i++) { var p = SplitGeom.Eval(j.S, a.At(t0 + (t1 - t0) * i / 8)); L += SplitGeom.Dist(prev, p); prev = p; }
                return L;
            }

            void BuildPieces(FaceJob j)
            {
                foreach (var a in j.Arcs)
                {
                    var sp = a.Splits.OrderBy(x => x.T).ToList();
                    var keep = new List<SplitPt>();
                    foreach (var x in sp)
                    {
                        int v = Find(j, x.V);
                        if (keep.Count > 0)
                        {
                            var last = keep[keep.Count - 1];
                            if (Find(j, last.V) == v && Len3(j, a, last.T, x.T) <= Tol) continue;
                            if (x.T - last.T <= 1e-13 * (a.T1 - a.T0)) { Union(j, last.V, v); continue; }
                        }
                        keep.Add(new SplitPt { T = x.T, V = v });
                    }
                    if (a.IsTrim && (keep[0].T != a.T0 || keep[keep.Count - 1].T != a.T1)) j.Warn.Add($"trim {a.Trim}: end split lost");
                    for (int i = 0; i + 1 < keep.Count; i++)
                        j.Pieces.Add(new Piece { Arc = a, Ta = keep[i].T, Tb = keep[i + 1].T, Va = keep[i].V, Vb = keep[i + 1].V });
                }
                foreach (var p in j.Pieces) { p.Va = Find(j, p.Va); p.Vb = Find(j, p.Vb); }

                // overlaps: a new piece running along a trim piece (or a duplicate new piece) is dropped; the trim becomes a cut
                foreach (var p in j.Pieces.Where(p => !p.Arc.IsTrim))
                {
                    var mid = SplitGeom.Eval(j.S, p.Arc.At(0.5 * (p.Ta + p.Tb)));
                    foreach (var q in j.Pieces)
                    {
                        if (q == p || q.Dead) continue;
                        if (!((q.Va == p.Va && q.Vb == p.Vb) || (q.Va == p.Vb && q.Vb == p.Va))) continue;
                        double s = 0.5 * (q.Ta + q.Tb);
                        if (SplitGeom.ProjectCos(j.S, q.Arc.C2, q.Ta, q.Tb, mid, ref s) > Tol) continue;
                        p.Dead = true; if (q.Arc.IsTrim) q.Cut = true;
                        break;
                    }
                }

                // dangling new pieces do not separate regions
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    var deg = new Dictionary<int, int>();
                    foreach (var p in j.Pieces.Where(p => !p.Dead)) { deg[p.Va] = deg.TryGetValue(p.Va, out var d0) ? d0 + 1 : 1; deg[p.Vb] = deg.TryGetValue(p.Vb, out var d1) ? d1 + 1 : 1; }
                    foreach (var p in j.Pieces.Where(p => !p.Dead && !p.Arc.IsTrim))
                        if (deg[p.Va] == 1 || deg[p.Vb] == 1) { p.Dead = true; changed = true; }
                }
            }

            static double Ang(V3 d) => Math.Atan2(d.Y, d.X);

            static List<V3> PiecePoly(Piece p, bool fwd)
            {
                var a = p.Arc; var l = new List<V3> { a.At(p.Ta) };
                for (int i = 0; i < a.PT.Length; i++) if (a.PT[i] > p.Ta && a.PT[i] < p.Tb) l.Add(a.PUv[i]);
                l.Add(a.At(p.Tb));
                if (!fwd) l.Reverse();
                return l;
            }

            void TraceRegions(FaceJob j)
            {
                var hes = new List<HalfEdge>();
                var outg = new Dictionary<int, List<HalfEdge>>();
                foreach (var p in j.Pieces.Where(p => !p.Dead))
                {
                    var h1 = new HalfEdge { P = p, Fwd = true, From = p.Va, To = p.Vb };
                    var h2 = new HalfEdge { P = p, Fwd = false, From = p.Vb, To = p.Va };
                    h1.Twin = h2; h2.Twin = h1;
                    double dt = p.Tb - p.Ta;
                    var t0 = NurbsEval.CurveDerivatives(p.Arc.C2, p.Ta, 1)[1];
                    var t1 = NurbsEval.CurveDerivatives(p.Arc.C2, p.Tb, 1)[1] * -1;
                    var c0 = p.Arc.At(p.Ta + 0.02 * dt) - p.Arc.At(p.Ta);
                    var c1 = p.Arc.At(p.Tb - 0.02 * dt) - p.Arc.At(p.Tb);
                    h1.Ang2 = Ang(c0); h2.Ang2 = Ang(c1);
                    h1.Ang = t0.Length > 1e-14 ? Ang(t0) : h1.Ang2;
                    h2.Ang = t1.Length > 1e-14 ? Ang(t1) : h2.Ang2;
                    foreach (var h in new[] { h1, h2 })
                    {
                        hes.Add(h);
                        if (!outg.TryGetValue(h.From, out var l)) outg[h.From] = l = new List<HalfEdge>();
                        l.Add(h);
                    }
                }
                int Cmp(HalfEdge x, HalfEdge y)
                {
                    if (Math.Abs(x.Ang - y.Ang) > 1e-9) return x.Ang.CompareTo(y.Ang);
                    double rx = Math.IEEERemainder(x.Ang2 - x.Ang, 2 * Math.PI), ry = Math.IEEERemainder(y.Ang2 - y.Ang, 2 * Math.PI);
                    return rx.CompareTo(ry);
                }
                foreach (var l in outg.Values) l.Sort(Cmp);
                HalfEdge Next(HalfEdge h)
                {
                    var l = outg[h.To];
                    int k = l.IndexOf(h.Twin);
                    return l[(k - 1 + l.Count) % l.Count];
                }

                var outers = new List<(List<HalfEdge> c, List<V3> poly, double area)>();
                var holes = new List<(List<HalfEdge> c, List<V3> poly, double area)>();
                foreach (var h0 in hes)
                {
                    if (h0.Used) continue;
                    var cyc = new List<HalfEdge>();
                    var h = h0;
                    while (!h.Used) { h.Used = true; cyc.Add(h); h = Next(h); }
                    if (h != h0) { j.Warn.Add("open cycle in arrangement"); continue; }
                    if (cyc.Any(e => e.P.Arc.IsTrim && !e.Fwd)) continue; // outside of the face
                    var poly = new List<V3>();
                    foreach (var e in cyc) { var pp = PiecePoly(e.P, e.Fwd); for (int i = 0; i + 1 < pp.Count; i++) poly.Add(pp[i]); }
                    if (!cyc.Any(e => e.P.Arc.IsTrim))
                    {
                        var pm = cyc[0].P.Arc.At(0.5 * (cyc[0].P.Ta + cyc[0].P.Tb));
                        int wsum = 0; foreach (var lp in j.OrigLoops) wsum += SplitGeom.Winding(lp, pm);
                        if (wsum == 0) continue;
                    }
                    double ar = SplitGeom.SignedArea(poly);
                    if (ar > 0) outers.Add((cyc, poly, ar)); else holes.Add((cyc, poly, ar));
                }
                var regions = outers.Select(o => new List<List<HalfEdge>> { o.c }).ToList();
                foreach (var hcy in holes)
                {
                    var p = hcy.c[0].P; var pm = p.Arc.At(0.5 * (p.Ta + p.Tb));
                    int best = -1;
                    var hp = new HashSet<Piece>(hcy.c.Select(e => e.P));
                    for (int i = 0; i < outers.Count; i++)
                        if (!outers[i].c.Any(e => hp.Contains(e.P)) && SplitGeom.Winding(outers[i].poly, pm) != 0 && (best < 0 || outers[i].area < outers[best].area)) best = i;
                    if (best < 0) { j.Warn.Add($"hole cycle without container (area {hcy.area:E2}) dropped"); continue; }
                    regions[best].Add(hcy.c);
                }
                j.Regions = regions;
            }

            // ---------------------------------------------------------- assembly

            double MapParam(SsiBranch br, double t, double tolDom)
            {
                var uvd = Uv(br).Domain; var c3 = br.Curve3; var d3 = c3.Domain;
                if (Math.Abs(t - uvd.T0) <= tolDom) return d3.T0;
                if (Math.Abs(t - uvd.T1) <= tolDom) return d3.T1;
                double seed = d3.ParameterAt((t - uvd.T0) / uvd.Length);
                var srf = B.Surfaces[B.Faces[OnA ? br.FaceA : br.FaceB].Surface];
                var uv = NurbsEval.CurvePoint(Uv(br), t);
                double u = srf.IsClosedU ? SplitGeom.Wrap(uv.X, srf.DomainU) : uv.X, v = srf.IsClosedV ? SplitGeom.Wrap(uv.Y, srf.DomainV) : uv.Y;
                var x = NurbsEval.SurfacePoint(srf, u, v);
                double w = 0.05 * d3.Length;
                double tau = SplitGeom.ClosestOnCurve(c3, x, Math.Max(d3.T0, seed - w), Math.Min(d3.T1, seed + w));
                if (SplitGeom.Dist(NurbsEval.CurvePoint(c3, tau), x) > Tol)
                {
                    tau = SplitGeom.ClosestOnCurve(c3, x, d3.T0, d3.T1);
                    if (SplitGeom.Dist(NurbsEval.CurvePoint(c3, tau), x) > Tol) W.Add($"branch pullback point {SplitGeom.Dist(NurbsEval.CurvePoint(c3, tau), x):E2} off Curve3");
                }
                return tau;
            }

            List<KBrep> Assemble(Dictionary<int, FaceJob> jobs)
            {
                var G = new KBrep();
                G.Surfaces.AddRange(B.Surfaces); G.Curves2.AddRange(B.Curves2); G.Curves3.AddRange(B.Curves3);
                G.Vertices = GV;
                var isNew = new List<bool>(); var cut = new List<bool>();
                int AddEdge(KEdge e, bool nw) { G.Edges.Add(e); isNew.Add(nw); cut.Add(false); return G.Edges.Count - 1; }

                // original edges (split into sub-edges where needed)
                var single = new int[B.Edges.Count];
                var subs = new Dictionary<int, (double[] P, int[] Ids)>();
                for (int ei = 0; ei < B.Edges.Count; ei++)
                {
                    var E = B.Edges[ei];
                    if (!EdgeSplits.TryGetValue(ei, out var es) || es.Pts.Count == 0)
                    {
                        single[ei] = AddEdge(new KEdge { Curve3 = E.Curve3, Domain = E.Domain, V0 = E.V0, V1 = E.V1, Tolerance = E.Tolerance }, false);
                        continue;
                    }
                    single[ei] = -1;
                    // Split points within eps of a neighbour or an end would make a zero-length sub-edge; skip them.
                    double splitEps = 1e-9 * Math.Max(1.0, Math.Abs(E.Domain.T0) + Math.Abs(E.Domain.T1));
                    var P = new List<double> { E.Domain.T0 }; var V = new List<int> { E.V0 };
                    foreach (var (e, gv) in es.Pts.OrderBy(x => x.Item1))
                        if (e > P[P.Count - 1] + splitEps && e < E.Domain.T1 - splitEps) { P.Add(e); V.Add(gv); }
                    P.Add(E.Domain.T1); V.Add(E.V1);
                    var ids = new int[P.Count - 1];
                    for (int k = 0; k + 1 < P.Count; k++)
                    {
                        var c = SplitGeom.SubCurve(B.Curves3[E.Curve3], P[k], P[k + 1]);
                        G.Curves3.Add(c);
                        ids[k] = AddEdge(new KEdge { Curve3 = G.Curves3.Count - 1, Domain = c.Domain, V0 = V[k], V1 = V[k + 1], Tolerance = E.Tolerance }, false);
                    }
                    subs[ei] = (P.ToArray(), ids);
                }

                int MapOrigEdge(int ei, NurbsSurface srf, V3 uvMid)
                {
                    if (single[ei] >= 0) return single[ei];
                    var (P, ids) = subs[ei]; var E = B.Edges[ei];
                    double e = SplitGeom.ClosestOnCurve(B.Curves3[E.Curve3], SplitGeom.Eval(srf, uvMid), E.Domain.T0, E.Domain.T1);
                    for (int k = 0; k + 1 < P.Length; k++) if (e <= P[k + 1]) return ids[k];
                    return ids[ids.Length - 1];
                }

                int GlobalOf(FaceJob j, int v)
                {
                    var x = j.Verts[v];
                    if (x.Global < 0) { GV.Add(new KVertex { Point = x.X, Tolerance = 0 }); x.Global = GV.Count - 1; }
                    return x.Global;
                }

                for (int fi = 0; fi < B.Faces.Count; fi++)
                {
                    var F = B.Faces[fi];
                    if (!jobs.TryGetValue(fi, out var j))
                    {
                        var nf = new KFace { Surface = F.Surface, Reversed = F.Reversed };
                        G.Faces.Add(nf);
                        foreach (var li in F.Loops)
                        {
                            var nl = new KLoop { Face = G.Faces.Count - 1, Type = B.Loops[li].Type };
                            G.Loops.Add(nl); nf.Loops.Add(G.Loops.Count - 1);
                            foreach (var ti in B.Loops[li].Trims)
                            {
                                var T = B.Trims[ti];
                                var nt = new KTrim { Curve2 = T.Curve2, Domain = T.Domain, Edge = T.Edge < 0 ? -1 : single[T.Edge], Loop = G.Loops.Count - 1, Reversed3d = T.Reversed3d, Type = T.Type, TolU = T.TolU, TolV = T.TolV };
                                if (T.Edge >= 0 && nt.Edge < 0) throw new InvalidOperationException("untouched face references a split edge");
                                G.Trims.Add(nt); nl.Trims.Add(G.Trims.Count - 1);
                                if (nt.Edge >= 0) G.Edges[nt.Edge].Trims.Add(G.Trims.Count - 1);
                            }
                        }
                        continue;
                    }
                    var srf = j.S;
                    foreach (var region in j.Regions)
                    {
                        var nf = new KFace { Surface = F.Surface, Reversed = F.Reversed };
                        G.Faces.Add(nf);
                        for (int li = 0; li < region.Count; li++)
                        {
                            var nl = new KLoop { Face = G.Faces.Count - 1, Type = li == 0 ? LoopType.Outer : LoopType.Inner };
                            G.Loops.Add(nl); nf.Loops.Add(G.Loops.Count - 1);
                            foreach (var h in region[li])
                            {
                                var p = h.P; var a = p.Arc;
                                var nt = new KTrim { Loop = G.Loops.Count - 1 };
                                if (a.IsTrim)
                                {
                                    var T = B.Trims[a.Trim];
                                    if (p.Ta == a.T0 && p.Tb == a.T1) { nt.Curve2 = T.Curve2; nt.Domain = T.Domain; }
                                    else { var c = SplitGeom.SubCurve(a.C2, p.Ta, p.Tb); G.Curves2.Add(c); nt.Curve2 = G.Curves2.Count - 1; nt.Domain = c.Domain; }
                                    nt.Reversed3d = T.Reversed3d; nt.Type = T.Type; nt.TolU = T.TolU; nt.TolV = T.TolV;
                                    nt.Edge = T.Edge < 0 ? -1 : MapOrigEdge(T.Edge, srf, a.At(0.5 * (p.Ta + p.Tb)));
                                    if (p.Cut && nt.Edge >= 0) cut[nt.Edge] = true;
                                }
                                else
                                {
                                    var br = Branches[a.Branch];
                                    if (p.Edge < 0)
                                    {
                                        double eps = 1e-12 * (1 + Math.Abs(Uv(br).Domain.T0) + Math.Abs(Uv(br).Domain.T1));
                                        double ta = MapParam(br, p.Ta, eps), tb = MapParam(br, p.Tb, eps);
                                        int g0 = GlobalOf(j, p.Va), g1 = GlobalOf(j, p.Vb);
                                        if (!(tb > ta)) { W.Add($"branch {a.Branch}: non-increasing Curve3 parameter ({ta}..{tb})"); var t = ta; ta = tb; tb = t; var g = g0; g0 = g1; g1 = g; p.EdgeRev = true; }
                                        var c3 = SplitGeom.SnapEnds(SplitGeom.SubCurve(br.Curve3, ta, tb), GV[g0].Point, GV[g1].Point);
                                        G.Curves3.Add(c3);
                                        p.Edge = AddEdge(new KEdge { Curve3 = G.Curves3.Count - 1, Domain = c3.Domain, V0 = g0, V1 = g1 }, true);
                                    }
                                    var c2 = (p.Ta == a.T0 && p.Tb == a.T1) ? a.C2 : SplitGeom.SubCurve(a.C2, p.Ta, p.Tb);
                                    c2 = SplitGeom.SnapEnds(c2, j.Verts[p.Va].Uv, j.Verts[p.Vb].Uv);
                                    if (!h.Fwd) c2 = SplitGeom.Reverse(c2);
                                    G.Curves2.Add(c2); nt.Curve2 = G.Curves2.Count - 1; nt.Domain = c2.Domain;
                                    nt.Edge = p.Edge;
                                    nt.Reversed3d = h.Fwd == p.EdgeRev;
                                    nt.Type = TrimType.Mated;
                                }
                                G.Trims.Add(nt); nl.Trims.Add(G.Trims.Count - 1);
                                if (nt.Edge >= 0) G.Edges[nt.Edge].Trims.Add(G.Trims.Count - 1);
                            }
                        }
                    }
                }

                // tolerances of the new edges: measured trim-to-edge deviation
                for (int e = 0; e < G.Edges.Count; e++)
                {
                    if (!isNew[e]) continue;
                    var E = G.Edges[e]; var c3 = G.Curves3[E.Curve3]; double dev = 0;
                    foreach (var ti in E.Trims)
                    {
                        var T = G.Trims[ti]; var srf = G.Surfaces[G.Faces[G.Loops[T.Loop].Face].Surface];
                        for (int k = 0; k <= 16; k++)
                        {
                            var x = SplitGeom.Eval(srf, NurbsEval.CurvePoint(G.Curves2[T.Curve2], T.Domain.ParameterAt(k / 16.0)));
                            dev = Math.Max(dev, SplitGeom.Dist(NurbsEval.CurvePoint(c3, SplitGeom.ClosestOnCurve(c3, x, E.Domain.T0, E.Domain.T1)), x));
                        }
                    }
                    E.Tolerance = dev;
                }

                // connected pieces: faces joined through original-derived, uncut edges
                int nF = G.Faces.Count;
                var par = Enumerable.Range(0, nF).ToArray();
                int Fd(int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
                for (int e = 0; e < G.Edges.Count; e++)
                {
                    if (isNew[e] || cut[e]) continue;
                    var fs = G.Edges[e].Trims.Select(t => Fd(G.Loops[G.Trims[t].Loop].Face)).Distinct().ToList();
                    for (int k = 1; k < fs.Count; k++) par[Fd(fs[k])] = Fd(fs[0]);
                }
                var comps = new Dictionary<int, List<int>>();
                for (int f = 0; f < nF; f++) { int r = Fd(f); if (!comps.TryGetValue(r, out var l)) comps[r] = l = new List<int>(); l.Add(f); }
                return comps.Values.Select(c => Compact(G, c)).ToList();
            }

            static KBrep Compact(KBrep G, List<int> faces)
            {
                var r = new KBrep();
                var ms = new Dictionary<int, int>(); var m2 = new Dictionary<int, int>(); var m3 = new Dictionary<int, int>();
                var mv = new Dictionary<int, int>(); var me = new Dictionary<int, int>();
                int Map<T>(Dictionary<int, int> m, List<T> src, List<T> dst, int i) { if (i < 0) return -1; if (!m.TryGetValue(i, out var k)) { dst.Add(src[i]); m[i] = k = dst.Count - 1; } return k; }
                int MapV(int i)
                {
                    if (i < 0) return -1;
                    if (!mv.TryGetValue(i, out var k)) { r.Vertices.Add(new KVertex { Point = G.Vertices[i].Point, Tolerance = G.Vertices[i].Tolerance }); mv[i] = k = r.Vertices.Count - 1; }
                    return k;
                }
                int MapE(int i)
                {
                    if (i < 0) return -1;
                    if (!me.TryGetValue(i, out var k))
                    {
                        var E = G.Edges[i];
                        r.Edges.Add(new KEdge { Curve3 = Map(m3, G.Curves3, r.Curves3, E.Curve3), Domain = E.Domain, V0 = MapV(E.V0), V1 = MapV(E.V1), Tolerance = E.Tolerance });
                        me[i] = k = r.Edges.Count - 1;
                    }
                    return k;
                }
                foreach (var fi in faces)
                {
                    var F = G.Faces[fi];
                    var nf = new KFace { Surface = Map(ms, G.Surfaces, r.Surfaces, F.Surface), Reversed = F.Reversed };
                    r.Faces.Add(nf);
                    foreach (var li in F.Loops)
                    {
                        var L = G.Loops[li];
                        var nl = new KLoop { Face = r.Faces.Count - 1, Type = L.Type };
                        r.Loops.Add(nl); nf.Loops.Add(r.Loops.Count - 1);
                        foreach (var ti in L.Trims)
                        {
                            var T = G.Trims[ti];
                            var nt = new KTrim { Curve2 = Map(m2, G.Curves2, r.Curves2, T.Curve2), Domain = T.Domain, Edge = MapE(T.Edge), Loop = r.Loops.Count - 1, Reversed3d = T.Reversed3d, Type = T.Type, TolU = T.TolU, TolV = T.TolV };
                            r.Trims.Add(nt); nl.Trims.Add(r.Trims.Count - 1);
                            if (nt.Edge >= 0) r.Edges[nt.Edge].Trims.Add(r.Trims.Count - 1);
                        }
                    }
                }
                for (int e = 0; e < r.Edges.Count; e++)
                {
                    var E = r.Edges[e];
                    if (E.V0 >= 0) r.Vertices[E.V0].Edges.Add(e);
                    if (E.V1 >= 0) r.Vertices[E.V1].Edges.Add(e);
                }
                foreach (var T in r.Trims)
                {
                    if (T.Edge < 0) { if (T.Type != TrimType.Singular && T.Type != TrimType.PointOnSurface) T.Type = TrimType.Singular; continue; }
                    var et = r.Edges[T.Edge].Trims;
                    if (et.Count == 1) T.Type = TrimType.Boundary;
                    else if (et.Count == 2 && r.Trims[et[0]].Loop == r.Trims[et[1]].Loop) T.Type = TrimType.Seam;
                    else T.Type = TrimType.Mated;
                }
                return r;
            }
        }
    }
}
