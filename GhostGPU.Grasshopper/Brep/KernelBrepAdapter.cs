using System;
using System.Text;
using Rhino;
using Rhino.Geometry;
using GhostGPU.Core;
using K = GhostGPU.Kernel;

namespace GhostGPU.Grasshopper.BrepPipeline
{
    /// <summary>
    /// Rhino <see cref="Brep"/> &lt;-&gt; <see cref="K.KBrep"/> (docs/contracts/kernel.md / KernelModel.cs conventions:
    /// full clamped knots, homogeneous CVs, surface CVs row-major over U, ON_Brep topology with identical
    /// vertex/edge/trim/loop/face indices on the way in).
    /// Edges and trims are converted through their proxies (BrepEdge/BrepTrim.ToNurbsCurve), so every KEdge/KTrim
    /// owns one curve whose domain is exactly the proxy domain (handles proxy sub-domains and reversal).
    /// </summary>
    public static class KernelBrepAdapter
    {
        // ------------------------------------------------------------------ Rhino -> kernel

        /// <summary>
        /// Converts a Rhino Brep to a KBrep. Throws <see cref="NotSupportedException"/> when a surface cannot be
        /// represented exactly for the kernel (no NURBS form, unclamped/periodic knots, or a NURBS form whose
        /// parameterization differs from the original while the face has non-boundary trims).
        /// </summary>
        public static K.KBrep ToKernel(Brep brep)
        {
            if (brep == null) throw new ArgumentNullException(nameof(brep));
            // Convert every face to its NURBS form with trims re-mapped (rev/sphere/cylinder surfaces),
            // so the HasNurbsForm == 2 restriction below only triggers if Rhino could not convert.
            var v2 = brep.DuplicateBrep();
            if (v2.MakeValidForV2()) brep = v2;
            var k = new K.KBrep();

            for (int i = 0; i < brep.Surfaces.Count; i++)
            {
                var src = brep.Surfaces[i];
                var s = src.ToNurbsSurface() ?? throw new NotSupportedException($"surface {i} has no NURBS form");
                // HasNurbsForm == 2: same domain, different parameterization (rev/sphere/cylinder surfaces).
                // UV trims then only stay exact if they run along the domain boundary.
                if (src.HasNurbsForm() == 2 && !OnlyBoundaryTrims(brep, i))
                    throw new NotSupportedException($"surface {i}: NURBS parameterization differs from the source surface and its face has interior trims");
                // Limitation: periodic (unclamped) surfaces are rejected; RhinoCommon has no surface ClampEnd. Upgrade: knot-insert to clamp.
                if (!s.KnotsU.ClampedAtStart || !s.KnotsU.ClampedAtEnd || !s.KnotsV.ClampedAtStart || !s.KnotsV.ClampedAtEnd)
                    throw new NotSupportedException($"surface {i} has unclamped (periodic) knots");

                int nu = s.Points.CountU, nv = s.Points.CountV;
                var cvs = new double[nu * nv * 4];
                int p = 0;
                for (int a = 0; a < nu; a++)
                    for (int b = 0; b < nv; b++)
                    {
                        ControlPoint cp = s.Points.GetControlPoint(a, b); // X,Y,Z are homogeneous (x*w)
                        cvs[p++] = cp.X; cvs[p++] = cp.Y; cvs[p++] = cp.Z; cvs[p++] = cp.Weight;
                    }
                k.Surfaces.Add(new K.NurbsSurface(s.Degree(0), s.Degree(1), nu, nv, s.IsRational,
                    FullKnots(s.KnotsU), FullKnots(s.KnotsV), cvs)
                {
                    IsClosedU = s.IsClosed(0),
                    IsClosedV = s.IsClosed(1)
                });
            }

            foreach (var v in brep.Vertices)
            {
                var kv = new K.KVertex { Point = new V3(v.Location.X, v.Location.Y, v.Location.Z), Tolerance = v.Tolerance };
                kv.Edges.AddRange(v.EdgeIndices() ?? new int[0]);
                k.Vertices.Add(kv);
            }

            foreach (var e in brep.Edges)
            {
                var c = ToKernelCurve(e.ToNurbsCurve(), 3, $"edge {e.EdgeIndex}");
                k.Curves3.Add(c);
                var ke = new K.KEdge
                {
                    Curve3 = k.Curves3.Count - 1,
                    Domain = c.Domain,
                    V0 = e.StartVertex?.VertexIndex ?? -1,
                    V1 = e.EndVertex?.VertexIndex ?? -1,
                    Tolerance = e.Tolerance
                };
                ke.Trims.AddRange(e.TrimIndices() ?? new int[0]);
                k.Edges.Add(ke);
            }

            foreach (var t in brep.Trims)
            {
                var c = ToKernelCurve(t.ToNurbsCurve(), 2, $"trim {t.TrimIndex}");
                k.Curves2.Add(c);
                t.GetTolerances(out double tolU, out double tolV);
                k.Trims.Add(new K.KTrim
                {
                    Curve2 = k.Curves2.Count - 1,
                    Domain = c.Domain,
                    Edge = t.Edge?.EdgeIndex ?? -1,
                    Loop = t.Loop?.LoopIndex ?? -1,
                    Reversed3d = t.IsReversed(),
                    Type = (K.TrimType)Enum.Parse(typeof(K.TrimType), t.TrimType.ToString()),
                    TolU = tolU,
                    TolV = tolV
                });
            }

            foreach (var l in brep.Loops)
            {
                var kl = new K.KLoop
                {
                    Face = l.Face?.FaceIndex ?? -1,
                    Type = (K.LoopType)Enum.Parse(typeof(K.LoopType), l.LoopType.ToString())
                };
                for (int i = 0; i < l.Trims.Count; i++) kl.Trims.Add(l.Trims[i].TrimIndex);
                k.Loops.Add(kl);
            }

            foreach (var f in brep.Faces)
            {
                var kf = new K.KFace { Surface = f.SurfaceIndex, Reversed = f.OrientationIsReversed };
                if (f.OuterLoop != null) kf.Loops.Add(f.OuterLoop.LoopIndex);
                for (int i = 0; i < f.Loops.Count; i++)
                    if (!kf.Loops.Contains(f.Loops[i].LoopIndex)) kf.Loops.Add(f.Loops[i].LoopIndex);
                k.Faces.Add(kf);
            }

            return k;
        }

        private static bool OnlyBoundaryTrims(Brep brep, int surfaceIndex)
        {
            foreach (var f in brep.Faces)
            {
                if (f.SurfaceIndex != surfaceIndex) continue;
                foreach (var l in f.Loops)
                    foreach (var t in l.Trims)
                    {
                        var iso = t.IsoStatus;
                        if (iso != IsoStatus.West && iso != IsoStatus.East && iso != IsoStatus.South && iso != IsoStatus.North) return false;
                    }
            }
            return true;
        }

        private static K.NurbsCurve ToKernelCurve(Rhino.Geometry.NurbsCurve nc, int dimension, string what)
        {
            if (nc == null) throw new NotSupportedException($"{what} has no NURBS form");
            if (!nc.Knots.IsClampedStart || !nc.Knots.IsClampedEnd) nc.Knots.ClampEnd(CurveEnd.Both);
            int n = nc.Points.Count;
            var cvs = new double[n * 4];
            for (int i = 0; i < n; i++)
            {
                ControlPoint cp = nc.Points[i]; // X,Y,Z are homogeneous (x*w)
                cvs[4 * i] = cp.X; cvs[4 * i + 1] = cp.Y; cvs[4 * i + 2] = dimension == 2 ? 0.0 : cp.Z; cvs[4 * i + 3] = cp.Weight;
            }
            return new K.NurbsCurve(dimension, nc.Degree, nc.IsRational, FullKnots(nc.Knots), cvs);
        }

        // opennurbs omits the two "superfluous" end knots; KBrep stores the full vector.
        private static double[] FullKnots(Rhino.Geometry.Collections.NurbsCurveKnotList knots)
        {
            var full = new double[knots.Count + 2];
            full[0] = knots.SuperfluousKnot(true);
            for (int i = 0; i < knots.Count; i++) full[i + 1] = knots[i];
            full[full.Length - 1] = knots.SuperfluousKnot(false);
            return full;
        }

        private static double[] FullKnots(Rhino.Geometry.Collections.NurbsSurfaceKnotList knots)
        {
            var full = new double[knots.Count + 2];
            full[0] = knots.SuperfluousKnot(true);
            for (int i = 0; i < knots.Count; i++) full[i + 1] = knots[i];
            full[full.Length - 1] = knots.SuperfluousKnot(false);
            return full;
        }

        // ------------------------------------------------------------------ kernel -> Rhino

        /// <summary>Converts a KBrep to a Rhino Brep (may be invalid; check IsValid). Tolerances and flags are recomputed by Rhino.</summary>
        public static Brep ToRhino(K.KBrep k) => ToRhino(k, out _);

        /// <summary>As <see cref="ToRhino(K.KBrep)"/>; <paramref name="invalidLog"/> is null when the result is valid, else Rhino's IsValidWithLog report.</summary>
        public static Brep ToRhino(K.KBrep k, out string invalidLog)
        {
            if (k == null) throw new ArgumentNullException(nameof(k));
            var b = new Brep();

            foreach (var s in k.Surfaces) b.AddSurface(ToRhinoSurface(s));
            foreach (var c in k.Curves2) b.AddTrimCurve(ToRhinoCurve(c));
            foreach (var c in k.Curves3) b.AddEdgeCurve(ToRhinoCurve(c));

            // Unset tolerances: SetTolerancesBoxesAndFlags computes the real ones below.
            foreach (var v in k.Vertices) b.Vertices.Add(new Point3d(v.Point.X, v.Point.Y, v.Point.Z), RhinoMath.UnsetValue);
            foreach (var e in k.Edges) b.Edges.Add(e.V0, e.V1, e.Curve3, new Interval(e.Domain.T0, e.Domain.T1), RhinoMath.UnsetValue);

            for (int fi = 0; fi < k.Faces.Count; fi++)
            {
                var kf = k.Faces[fi];
                var face = b.Faces.Add(kf.Surface);
                face.OrientationIsReversed = kf.Reversed;
                foreach (int li in kf.Loops)
                {
                    var kl = k.Loops[li];
                    var loop = b.Loops.Add((BrepLoopType)Enum.Parse(typeof(BrepLoopType), kl.Type.ToString()), face);
                    for (int p = 0; p < kl.Trims.Count; p++)
                    {
                        var kt = k.Trims[kl.Trims[p]];
                        BrepTrim t = kt.Edge >= 0
                            ? b.Trims.Add(b.Edges[kt.Edge], kt.Reversed3d, loop, kt.Curve2)
                            : b.Trims.AddSingularTrim(b.Vertices[SingularVertex(k, kl, p)], loop,
                                                      SingularIso(k.Curves2[kt.Curve2], kt.Domain, k.Surfaces[kf.Surface]), kt.Curve2);
                        var full = k.Curves2[kt.Curve2].Domain;
                        if (kt.Domain.T0 != full.T0 || kt.Domain.T1 != full.T1)
                            t.SetTrimCurve(kt.Curve2, new Interval(kt.Domain.T0, kt.Domain.T1));
                    }
                }
            }

            // Not lazy: recompute vertex/edge/trim tolerances, trim iso + type flags, loop types and trim boxes.
            b.SetTolerancesBoxesAndFlags(false, true, true, true, true, true, true, true);
            invalidLog = b.IsValidWithLog(out string log) ? null : log;
            return b;
        }

        // Vertex of a singular trim = end vertex of the previous edge trim in the loop, else start of the next.
        private static int SingularVertex(K.KBrep k, K.KLoop loop, int pos)
        {
            int n = loop.Trims.Count;
            var prev = k.Trims[loop.Trims[(pos - 1 + n) % n]];
            if (prev.Edge >= 0) { var e = k.Edges[prev.Edge]; return prev.Reversed3d ? e.V0 : e.V1; }
            var next = k.Trims[loop.Trims[(pos + 1) % n]];
            if (next.Edge >= 0) { var e = k.Edges[next.Edge]; return next.Reversed3d ? e.V1 : e.V0; }
            throw new InvalidOperationException("singular trim has no neighbouring edge trim to take its vertex from");
        }

        private static IsoStatus SingularIso(K.NurbsCurve c2, K.Interval dom, K.NurbsSurface s)
        {
            V3 a = K.NurbsEval.CurvePoint(c2, dom.T0), b = K.NurbsEval.CurvePoint(c2, dom.T1);
            K.Interval du = s.DomainU, dv = s.DomainV;
            double eu = 1e-9 * Math.Max(1.0, Math.Abs(du.Length)), ev = 1e-9 * Math.Max(1.0, Math.Abs(dv.Length));
            bool AtU(double u) => Math.Abs(a.X - u) <= eu && Math.Abs(b.X - u) <= eu;
            bool AtV(double v) => Math.Abs(a.Y - v) <= ev && Math.Abs(b.Y - v) <= ev;
            if (AtU(du.T0)) return IsoStatus.West;
            if (AtU(du.T1)) return IsoStatus.East;
            if (AtV(dv.T0)) return IsoStatus.South;
            if (AtV(dv.T1)) return IsoStatus.North;
            return IsoStatus.None; // SetTolerancesBoxesAndFlags recomputes iso flags anyway
        }

        private static Rhino.Geometry.NurbsCurve ToRhinoCurve(K.NurbsCurve c)
        {
            var nc = new Rhino.Geometry.NurbsCurve(c.Dimension, c.IsRational, c.Degree + 1, c.CvCount);
            for (int i = 0; i < nc.Knots.Count; i++) nc.Knots[i] = c.Knots[i + 1];
            for (int i = 0; i < c.CvCount; i++)
                nc.Points.SetPoint(i, new Point4d(c.Cvs[4 * i], c.Cvs[4 * i + 1], c.Cvs[4 * i + 2], c.Cvs[4 * i + 3]));
            return nc;
        }

        private static Rhino.Geometry.NurbsSurface ToRhinoSurface(K.NurbsSurface s)
        {
            var ns = Rhino.Geometry.NurbsSurface.Create(3, s.IsRational, s.DegreeU + 1, s.DegreeV + 1, s.CvCountU, s.CvCountV)
                     ?? throw new InvalidOperationException("NurbsSurface.Create failed");
            for (int i = 0; i < ns.KnotsU.Count; i++) ns.KnotsU[i] = s.KnotsU[i + 1];
            for (int i = 0; i < ns.KnotsV.Count; i++) ns.KnotsV[i] = s.KnotsV[i + 1];
            for (int i = 0; i < s.CvCountU; i++)
                for (int j = 0; j < s.CvCountV; j++)
                {
                    int q = (i * s.CvCountV + j) * 4;
                    ns.Points.SetPoint(i, j, new Point4d(s.Cvs[q], s.Cvs[q + 1], s.Cvs[q + 2], s.Cvs[q + 3]));
                }
            return ns;
        }

        // ------------------------------------------------------------------ diagnostics

        /// <summary>ToKernel then ToRhino; reports IsValid/IsSolid, face/edge counts and area/volume relative error.</summary>
        public static string RoundTripCheck(Brep brep)
        {
            if (brep == null) return "RoundTrip: FAIL (null brep)";
            Brep r; string log;
            try { r = ToRhino(ToKernel(brep), out log); }
            catch (Exception ex) { return $"RoundTrip: FAIL ({ex.GetType().Name}: {ex.Message})"; }

            double areaIn = AreaMassProperties.Compute(brep)?.Area ?? double.NaN;
            double areaOut = AreaMassProperties.Compute(r)?.Area ?? double.NaN;
            double areaErr = Math.Abs(areaOut - areaIn) / Math.Max(Math.Abs(areaIn), 1e-300);
            double volErr = 0.0;
            string vol = "n/a (not solid)";
            if (brep.IsSolid && r.IsSolid)
            {
                double vIn = VolumeMassProperties.Compute(brep)?.Volume ?? double.NaN;
                double vOut = VolumeMassProperties.Compute(r)?.Volume ?? double.NaN;
                volErr = Math.Abs(vOut - vIn) / Math.Max(Math.Abs(vIn), 1e-300);
                vol = $"{vIn:G10} -> {vOut:G10} (rel {volErr:E2})";
            }

            bool pass = r.IsValid == brep.IsValid && r.IsSolid == brep.IsSolid
                        && r.Faces.Count == brep.Faces.Count && r.Edges.Count == brep.Edges.Count
                        && areaErr < 1e-6 && volErr < 1e-6;
            var sb = new StringBuilder();
            sb.AppendLine($"RoundTrip: {(pass ? "PASS" : "FAIL")}");
            sb.AppendLine($"  IsValid  {brep.IsValid} -> {r.IsValid}; IsSolid {brep.IsSolid} -> {r.IsSolid}");
            sb.AppendLine($"  Faces {brep.Faces.Count} -> {r.Faces.Count}; Edges {brep.Edges.Count} -> {r.Edges.Count}; Trims {brep.Trims.Count} -> {r.Trims.Count}; Vertices {brep.Vertices.Count} -> {r.Vertices.Count}");
            sb.AppendLine($"  Area {areaIn:G10} -> {areaOut:G10} (rel {areaErr:E2}); Volume {vol}");
            if (log != null) sb.AppendLine("  IsValidWithLog: " + log.Trim());
            return sb.ToString();
        }
    }
}
