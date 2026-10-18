using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    /// <summary>Per-face data for SSI: surface, Bezier patches, domain/periodicity and the trim boundary as UV segments.</summary>
    internal sealed class SsiFace
    {
        public int Index;
        public NurbsSurface Srf;
        public List<BezierPatch> Patches;
        public double U0, U1, V0, V1;
        public bool PerU, PerV;
        public V2[] SegA = Array.Empty<V2>(), SegB = Array.Empty<V2>();   // trim boundary segments (a,b) in UV
        public V3 BoxMin, BoxMax;
        public bool Planar;
        public V3 PlaneN;
        public double PlaneD;

        public double Diag => (BoxMax - BoxMin).Length;

        public static SsiFace Build(KBrep b, int fi)
        {
            var face = b.Faces[fi];
            var srf = b.Surfaces[face.Surface];
            var f = new SsiFace
            {
                Index = fi, Srf = srf, Patches = BezierDecomposition.Decompose(srf, fi),
                U0 = srf.DomainU.T0, U1 = srf.DomainU.T1, V0 = srf.DomainV.T0, V1 = srf.DomainV.T1,
                PerU = srf.IsClosedU, PerV = srf.IsClosedV,
            };
            f.BoxMin = new V3(double.MaxValue, double.MaxValue, double.MaxValue);
            f.BoxMax = new V3(-double.MaxValue, -double.MaxValue, -double.MaxValue);
            foreach (var p in f.Patches) { f.BoxMin = V3.Min(f.BoxMin, p.BoxMin); f.BoxMax = V3.Max(f.BoxMax, p.BoxMax); }

            // trims -> UV segments (a segment soup: crossing parity needs no loop ordering)
            double uvTol = 1e-6 * Math.Max(f.U1 - f.U0, f.V1 - f.V0);
            var sa = new List<V2>(); var sb = new List<V2>();
            foreach (int li in face.Loops)
                foreach (int ti in b.Loops[li].Trims)
                {
                    var tr = b.Trims[ti];
                    if (tr.Curve2 < 0) continue;
                    var c = b.Curves2[tr.Curve2];
                    double t0 = Math.Max(tr.Domain.T0, c.Domain.T0), t1 = Math.Min(tr.Domain.T1, c.Domain.T1);
                    if (!(t1 > t0)) { t0 = c.Domain.T0; t1 = c.Domain.T1; }
                    var pts = new List<V2>();
                    Tessellate(c, t0, t1, uvTol, pts);
                    for (int k = 0; k + 1 < pts.Count; k++) { sa.Add(pts[k]); sb.Add(pts[k + 1]); }
                }
            f.SegA = sa.ToArray(); f.SegB = sb.ToArray();

            // planarity (for coincident-face detection)
            double um = 0.5 * (f.U0 + f.U1), vm = 0.5 * (f.V0 + f.V1);
            var n = NurbsEval.SurfaceNormal(srf, um, vm);
            var p0 = NurbsEval.SurfacePoint(srf, um, vm);
            if (n.Length > 0.5)
            {
                double d = V3.Dot(n, p0), eps = 1e-9 * (1.0 + f.Diag + p0.Length);
                bool planar = true;
                for (int i = 0; i < srf.Cvs.Length && planar; i += 4)
                {
                    double w = srf.Cvs[i + 3];
                    var x = new V3(srf.Cvs[i] / w, srf.Cvs[i + 1] / w, srf.Cvs[i + 2] / w);
                    planar = Math.Abs(V3.Dot(n, x) - d) <= eps;
                }
                f.Planar = planar; f.PlaneN = n; f.PlaneD = d;
            }
            return f;
        }

        static void Tessellate(NurbsCurve c, double t0, double t1, double uvTol, List<V2> pts)
        {
            int n = Math.Max(4, 4 * (c.CvCount - c.Degree));
            V2 a = P2(c, t0);
            pts.Add(a);
            for (int k = 0; k < n; k++)
            {
                double ta = t0 + (t1 - t0) * k / n, tb = t0 + (t1 - t0) * (k + 1) / n;
                V2 b = P2(c, tb);
                Rec(c, ta, tb, a, b, uvTol, 0, pts);
                a = b;
            }
        }

        static void Rec(NurbsCurve c, double ta, double tb, V2 a, V2 b, double uvTol, int depth, List<V2> pts)
        {
            double tm = 0.5 * (ta + tb);
            V2 m = P2(c, tm);
            double dx = m.X - 0.5 * (a.X + b.X), dy = m.Y - 0.5 * (a.Y + b.Y);
            if (depth < 20 && dx * dx + dy * dy > uvTol * uvTol)
            {
                Rec(c, ta, tm, a, m, uvTol, depth + 1, pts);
                Rec(c, tm, tb, m, b, uvTol, depth + 1, pts);
            }
            else pts.Add(b);
        }

        static V2 P2(NurbsCurve c, double t) { var p = NurbsEval.CurvePoint(c, t); return new V2(p.X, p.Y); }

        public double WrapU(double u) => PerU ? Wrap(u, U0, U1) : u;
        public double WrapV(double v) => PerV ? Wrap(v, V0, V1) : v;

        static double Wrap(double x, double lo, double hi)
        {
            double per = hi - lo;
            double r = x - per * Math.Floor((x - lo) / per);
            return r;
        }

        /// <summary>Point-in-trim test (crossing parity over all trim segments) with a 3D boundary band of bandTol.</summary>
        public bool Inside(double u, double v, V3 su, V3 sv, double bandTol)
        {
            u = WrapU(u); v = WrapV(v);
            if (SegA.Length == 0) return u >= U0 && u <= U1 && v >= V0 && v <= V1;
            int cross = 0;
            for (int i = 0; i < SegA.Length; i++)
            {
                V2 a = SegA[i], b = SegB[i];
                double ex = b.X - a.X, ey = b.Y - a.Y, l2 = ex * ex + ey * ey;
                double t = l2 > 0 ? ((u - a.X) * ex + (v - a.Y) * ey) / l2 : 0.0;
                t = t < 0 ? 0 : (t > 1 ? 1 : t);
                double du = u - (a.X + t * ex), dv = v - (a.Y + t * ey);
                if ((su * du + sv * dv).Length <= bandTol) return true;
                if ((a.Y > v) != (b.Y > v))
                {
                    double x = a.X + (v - a.Y) * ex / ey;
                    if (x > u) cross++;
                }
            }
            return (cross & 1) == 1;
        }
    }
}
