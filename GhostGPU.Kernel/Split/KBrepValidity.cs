using System;
using System.Collections.Generic;
using System.Linq;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    /// <summary>ON_Brep-style validity checks for a KBrep. Returns an empty list when valid.</summary>
    public static class KBrepValidity
    {
        /// <param name="faceAreas">optional, length Faces.Count: receives each face's area (Green + Gauss quadrature)</param>
        public static List<string> Check(KBrep b, double tol, double[] faceAreas = null)
        {
            var err = new List<string>();
            void E(string s) { if (err.Count < 200) err.Add(s); }
            bool In(int i, int n) => i >= 0 && i < n;

            for (int fi = 0; fi < b.Faces.Count; fi++)
            {
                var F = b.Faces[fi];
                if (!In(F.Surface, b.Surfaces.Count)) { E($"face {fi}: bad surface"); continue; }
                if (F.Loops.Count == 0) E($"face {fi}: no loops");
                for (int k = 0; k < F.Loops.Count; k++)
                {
                    int li = F.Loops[k];
                    if (!In(li, b.Loops.Count)) { E($"face {fi}: bad loop index"); continue; }
                    var L = b.Loops[li];
                    if (L.Face != fi) E($"loop {li}: face back-pointer {L.Face} != {fi}");
                    var want = k == 0 ? LoopType.Outer : LoopType.Inner;
                    if (L.Type != want) E($"loop {li}: type {L.Type}, expected {want}");
                }
            }

            for (int li = 0; li < b.Loops.Count; li++)
            {
                var L = b.Loops[li];
                if (L.Trims.Count == 0) { E($"loop {li}: no trims"); continue; }
                if (!In(L.Face, b.Faces.Count) || !In(b.Faces[L.Face].Surface, b.Surfaces.Count)) continue;
                var srf = b.Surfaces[b.Faces[L.Face].Surface];
                double scale = Math.Max(srf.DomainU.Length, srf.DomainV.Length);
                var poly = new List<V3>();
                for (int k = 0; k < L.Trims.Count; k++)
                {
                    int ti = L.Trims[k], tn = L.Trims[(k + 1) % L.Trims.Count];
                    if (!In(ti, b.Trims.Count) || !In(tn, b.Trims.Count)) { E($"loop {li}: bad trim index"); continue; }
                    var T = b.Trims[ti]; var Tn = b.Trims[tn];
                    if (T.Loop != li) E($"trim {ti}: loop back-pointer {T.Loop} != {li}");
                    if (!In(T.Curve2, b.Curves2.Count) || !In(Tn.Curve2, b.Curves2.Count)) { E($"trim {ti}: bad curve2"); continue; }
                    var c = b.Curves2[T.Curve2];
                    var pe = NurbsEval.CurvePoint(c, T.Domain.T1);
                    var ps = NurbsEval.CurvePoint(b.Curves2[Tn.Curve2], Tn.Domain.T0);
                    double gap = SplitGeom.Dist(pe, ps);
                    if (gap > 1e-9 * Math.Max(1, scale)) E($"loop {li}: UV gap {gap:E2} between trims {ti} and {tn}");
                    double gap3 = SplitGeom.Dist(SplitGeom.Eval(srf, pe), SplitGeom.Eval(srf, ps));
                    if (gap3 > tol) E($"loop {li}: 3D gap {gap3:E2} between trims {ti} and {tn}");
                    SplitGeom.Sample(c, T.Domain.T0, T.Domain.T1, out var ts, out var pts);
                    for (int i = 0; i + 1 < pts.Length; i++) poly.Add(pts[i]);
                }
                double area = SplitGeom.SignedArea(poly);
                if (L.Type == LoopType.Outer && !(area > 0)) E($"loop {li}: outer loop not CCW (UV area {area:E3})");
                if (L.Type == LoopType.Inner && !(area < 0)) E($"loop {li}: inner loop not CW (UV area {area:E3})");
            }

            for (int ti = 0; ti < b.Trims.Count; ti++)
            {
                var T = b.Trims[ti];
                if (!In(T.Curve2, b.Curves2.Count) || !In(T.Loop, b.Loops.Count)) { E($"trim {ti}: bad curve/loop index"); continue; }
                var c2 = b.Curves2[T.Curve2];
                if (c2.Dimension != 2) E($"trim {ti}: curve dimension {c2.Dimension}");
                if (T.Domain.T0 < c2.Domain.T0 - 1e-12 || T.Domain.T1 > c2.Domain.T1 + 1e-12 || !(T.Domain.Length > 0)) E($"trim {ti}: domain outside curve");
                var L = b.Loops[T.Loop];
                if (!In(L.Face, b.Faces.Count)) continue;
                var F = b.Faces[L.Face]; var srf = b.Surfaces[F.Surface];
                if (T.Edge < 0)
                {
                    if (T.Type != TrimType.Singular) E($"trim {ti}: no edge but type {T.Type}");
                    var a = SplitGeom.Eval(srf, NurbsEval.CurvePoint(c2, T.Domain.T0));
                    var m = SplitGeom.Eval(srf, NurbsEval.CurvePoint(c2, T.Domain.ParameterAt(0.5)));
                    if (SplitGeom.Dist(a, m) > tol) E($"trim {ti}: singular trim is not a point in 3D");
                    continue;
                }
                if (!In(T.Edge, b.Edges.Count)) { E($"trim {ti}: bad edge"); continue; }
                var Ed = b.Edges[T.Edge];
                if (!Ed.Trims.Contains(ti)) E($"trim {ti}: edge {T.Edge} does not list it");
                var et = Ed.Trims.Where(t => In(t, b.Trims.Count)).ToList();
                TrimType want = et.Count == 1 ? TrimType.Boundary
                    : (et.Count == 2 && b.Trims[et[0]].Loop == b.Trims[et[1]].Loop) ? TrimType.Seam : TrimType.Mated;
                if (T.Type != want) E($"trim {ti}: type {T.Type}, expected {want}");
                if (!In(Ed.Curve3, b.Curves3.Count)) continue;
                var c3 = b.Curves3[Ed.Curve3];
                double etol = Math.Max(tol, Ed.Tolerance);
                // end points against the edge's vertices
                int vs = T.Reversed3d ? Ed.V1 : Ed.V0, ve = T.Reversed3d ? Ed.V0 : Ed.V1;
                if (In(vs, b.Vertices.Count) && In(ve, b.Vertices.Count))
                {
                    var p0 = SplitGeom.Eval(srf, NurbsEval.CurvePoint(c2, T.Domain.T0));
                    var p1 = SplitGeom.Eval(srf, NurbsEval.CurvePoint(c2, T.Domain.T1));
                    if (SplitGeom.Dist(p0, b.Vertices[vs].Point) > etol) E($"trim {ti}: start {SplitGeom.Dist(p0, b.Vertices[vs].Point):E2} from vertex {vs}");
                    if (SplitGeom.Dist(p1, b.Vertices[ve].Point) > etol) E($"trim {ti}: end {SplitGeom.Dist(p1, b.Vertices[ve].Point):E2} from vertex {ve}");
                }
                // 3D image on the edge, same direction as the edge unless Reversed3d
                double maxd = 0; int dirBad = 0;
                for (int k = 0; k <= 16; k++)
                {
                    double t = T.Domain.ParameterAt(k / 16.0);
                    var cd = NurbsEval.CurveDerivatives(c2, t, 1);
                    var sd = NurbsEval.SurfaceDerivatives(srf, cd[0].X, cd[0].Y, 1);
                    var x = sd[0, 0];
                    double e = SplitGeom.ClosestOnCurve(c3, x, Ed.Domain.T0, Ed.Domain.T1);
                    var ed = NurbsEval.CurveDerivatives(c3, e, 1);
                    maxd = Math.Max(maxd, SplitGeom.Dist(ed[0], x));
                    if (k > 0 && k < 16)
                    {
                        var tan = sd[1, 0] * cd[1].X + sd[0, 1] * cd[1].Y;
                        double dot = V3.Dot(tan, ed[1]);
                        if (tan.Length > 1e-12 && ed[1].Length > 1e-12 && (dot > 0) == T.Reversed3d) dirBad++;
                    }
                }
                if (maxd > etol) E($"trim {ti}: 3D image {maxd:E2} away from edge {T.Edge} (tol {etol:E2})");
                if (dirBad > 0) E($"trim {ti}: direction disagrees with Reversed3d={T.Reversed3d} at {dirBad} samples");
            }

            for (int ei = 0; ei < b.Edges.Count; ei++)
            {
                var Ed = b.Edges[ei];
                if (!In(Ed.Curve3, b.Curves3.Count) || !In(Ed.V0, b.Vertices.Count) || !In(Ed.V1, b.Vertices.Count)) { E($"edge {ei}: bad curve/vertex index"); continue; }
                var c3 = b.Curves3[Ed.Curve3];
                if (Ed.Domain.T0 < c3.Domain.T0 - 1e-12 || Ed.Domain.T1 > c3.Domain.T1 + 1e-12 || !(Ed.Domain.Length > 0)) E($"edge {ei}: domain outside curve");
                double etol = Math.Max(tol, Ed.Tolerance);
                if (SplitGeom.Dist(NurbsEval.CurvePoint(c3, Ed.Domain.T0), b.Vertices[Ed.V0].Point) > etol) E($"edge {ei}: start off vertex {Ed.V0}");
                if (SplitGeom.Dist(NurbsEval.CurvePoint(c3, Ed.Domain.T1), b.Vertices[Ed.V1].Point) > etol) E($"edge {ei}: end off vertex {Ed.V1}");
                if (!b.Vertices[Ed.V0].Edges.Contains(ei) || !b.Vertices[Ed.V1].Edges.Contains(ei)) E($"edge {ei}: vertex does not list it");
                if (Ed.Trims.Count == 0) E($"edge {ei}: no trims");
                foreach (var t in Ed.Trims) if (!In(t, b.Trims.Count) || b.Trims[t].Edge != ei) E($"edge {ei}: trim {t} does not point back");
                if (Ed.Trims.Count == 2 && Ed.Trims.All(t => In(t, b.Trims.Count)))
                {
                    var t0 = b.Trims[Ed.Trims[0]]; var t1 = b.Trims[Ed.Trims[1]];
                    bool d0 = t0.Reversed3d ^ b.Faces[b.Loops[t0.Loop].Face].Reversed, d1 = t1.Reversed3d ^ b.Faces[b.Loops[t1.Loop].Face].Reversed;
                    if (d0 == d1) E($"edge {ei}: both trims run the same way (orientation inconsistent)");
                }
            }

            for (int vi = 0; vi < b.Vertices.Count; vi++)
                foreach (var e in b.Vertices[vi].Edges)
                    if (!In(e, b.Edges.Count) || (b.Edges[e].V0 != vi && b.Edges[e].V1 != vi)) E($"vertex {vi}: edge {e} does not use it");

            for (int fi = 0; fi < b.Faces.Count; fi++)
            {
                double a;
                try { a = SplitGeom.FaceArea(b, fi); } catch (Exception ex) { E($"face {fi}: area failed: {ex.Message}"); continue; }
                if (faceAreas != null) faceAreas[fi] = a;
                if (!(a > 0)) E($"face {fi}: area {a:E3} not positive");
            }
            return err;
        }
    }
}
