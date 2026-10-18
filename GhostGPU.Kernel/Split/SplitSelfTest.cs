using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    /// <summary>
    /// M3 validation against Rhino golden data (tests/fixtures/golden/ssi.json + split.json). Until M2 is available
    /// the SsiBranch inputs are rebuilt from Rhino's sampled intersection curves: C0 kinks are recovered (corner =
    /// closest approach of the cubic extrapolations from both sides), each smooth run is assigned to the face that
    /// contains it, samples are pulled back by Newton closest point, and 3D + UV curves are interpolated (cubic, or
    /// lines for degree-1 Rhino curves) with a shared parameterization.
    /// </summary>
    public static class SplitSelfTest
    {
        public static string Run(string goldenDir, Action<string> progress = null)
        {
            var sb = new StringBuilder();
            var ssi = (List<object>)MiniJson.Parse(File.ReadAllText(Path.Combine(goldenDir, "ssi.json")));
            var split = (List<object>)MiniJson.Parse(File.ReadAllText(Path.Combine(goldenDir, "split.json")));
            int pass = 0, areaPass = 0, total = 0;
            double worstRel = 0;
            foreach (Dictionary<string, object> c in ssi)
            {
                string name = (string)c["Name"]; double tol = D(c["Tolerance"]);
                var exp = split.Cast<Dictionary<string, object>>().First(x => (string)x["Name"] == name);
                var A = ReadBrep((Dictionary<string, object>)c["BrepA"]); var Bb = ReadBrep((Dictionary<string, object>)c["BrepB"]);
                var curves = ((List<object>)c["Curves"]).Cast<Dictionary<string, object>>().ToList();
                foreach (var onA in new[] { true, false })
                {
                    total++;
                    var target = onA ? A : Bb;
                    var e = (Dictionary<string, object>)exp[onA ? "SplitAByB" : "SplitBByA"];
                    var ePieces = ((List<object>)e["Pieces"]).Cast<Dictionary<string, object>>()
                        .Select(p => (faces: (int)D(p["FaceCount"]), area: D(p["Area"]), valid: (bool)p["IsValid"])).OrderBy(p => p.area).ToList();
                    var log = new List<string>();
                    string line = $"{name,-22} {(onA ? "A/B" : "B/A")}  rhino {ePieces.Count} [{string.Join(", ", ePieces.Select(p => $"{p.faces}f {p.area:F6}"))}]";
                    try
                    {
                        var sw = Stopwatch.StartNew();
                        var branches = BuildBranches(target, curves, tol, onA, log);
                        double maxDev = branches.Count == 0 ? 0 : branches.Max(b => b.MaxDeviation);
                        var tb = sw.Elapsed.TotalMilliseconds; sw.Restart();
                        var pieces = FaceSplit.Split(target, branches, onA, tol, log);
                        var ts = sw.Elapsed.TotalMilliseconds; sw.Restart();
                        var ours = new List<(int faces, double area, List<string> errs)>();
                        foreach (var p in pieces)
                        {
                            var fa = new double[p.Faces.Count];
                            var errs = KBrepValidity.Check(p, tol, fa);
                            ours.Add((p.Faces.Count, fa.Sum(), errs));
                        }
                        ours = ours.OrderBy(o => o.area).ToList();
                        bool ok = ours.Count == ePieces.Count && ours.All(o => o.errs.Count == 0);
                        double rel = 0;
                        if (ours.Count == ePieces.Count)
                            for (int i = 0; i < ours.Count; i++)
                            {
                                rel = Math.Max(rel, Math.Abs(ours[i].area - ePieces[i].area) / ePieces[i].area);
                                if (ours[i].faces != ePieces[i].faces) ok = false;
                            }
                        else ok = false;
                        if (ok) { pass++; worstRel = Math.Max(worstRel, rel); if (rel <= 1e-6) areaPass++; }
                        line += $"\n{"",-26} ours  {ours.Count} [{string.Join(", ", ours.Select(o => $"{o.faces}f {o.area:F6}"))}]  maxRelArea {rel:E2}  valid {ours.Count(o => o.errs.Count == 0)}/{ours.Count}"
                              + $"  branches {branches.Count} (fit dev {maxDev:E1})  {(ok ? (rel <= 1e-6 ? "PASS" : "PASS(topology)") : "FAIL")}  [{tb:F0}+{ts:F0}+{sw.Elapsed.TotalMilliseconds:F0} ms]";
                        foreach (var o in ours) foreach (var er in o.errs.Take(6)) line += $"\n{"",-30}invalid: {er}";
                    }
                    catch (Exception ex) { line += $"\n{"",-26} EXCEPTION {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}"; }
                    foreach (var l in log.Distinct().Take(12)) line += $"\n{"",-30}note: {l}";
                    sb.AppendLine(line); progress?.Invoke(line);
                }
            }
            sb.AppendLine($"M3 split: {pass}/{total} topology pass (piece count, faces per piece, validity); {areaPass}/{total} also within 1e-6 relative area; worst relative area error among topology passes {worstRel:E2}");
            return sb.ToString();
        }

        static double D(object o) => o is double d ? d : Convert.ToDouble(o, CultureInfo.InvariantCulture);
        static double[] Arr(object o) => ((List<object>)o).Select(D).ToArray();
        static List<int> Ints(object o) => ((List<object>)o).Select(x => (int)D(x)).ToList();

        // ------------------------------------------------------------------ golden KBrep reader

        public static KBrep ReadBrep(Dictionary<string, object> j)
        {
            var b = new KBrep();
            foreach (Dictionary<string, object> s in (List<object>)j["Surfaces"])
                b.Surfaces.Add(new NurbsSurface((int)D(s["DegreeU"]), (int)D(s["DegreeV"]), (int)D(s["CvCountU"]), (int)D(s["CvCountV"]), (bool)s["IsRational"],
                    Arr(s["KnotsU"]), Arr(s["KnotsV"]), Arr(s["Cvs"])) { IsClosedU = (bool)s["IsClosedU"], IsClosedV = (bool)s["IsClosedV"] });
            NurbsCurve Crv(Dictionary<string, object> c) => new NurbsCurve((int)D(c["Dimension"]), (int)D(c["Degree"]), (bool)c["IsRational"], Arr(c["Knots"]), Arr(c["Cvs"]));
            foreach (Dictionary<string, object> c in (List<object>)j["Curves3"]) b.Curves3.Add(Crv(c));
            foreach (Dictionary<string, object> c in (List<object>)j["Curves2"]) b.Curves2.Add(Crv(c));
            foreach (Dictionary<string, object> v in (List<object>)j["Vertices"])
            { var p = Arr(v["Point"]); b.Vertices.Add(new KVertex { Point = new V3(p[0], p[1], p[2]), Tolerance = D(v["Tolerance"]), Edges = Ints(v["Edges"]) }); }
            foreach (Dictionary<string, object> e in (List<object>)j["Edges"])
            { var d = Arr(e["Domain"]); b.Edges.Add(new KEdge { Curve3 = (int)D(e["Curve3"]), Domain = new Interval(d[0], d[1]), V0 = (int)D(e["V0"]), V1 = (int)D(e["V1"]), Trims = Ints(e["Trims"]), Tolerance = D(e["Tolerance"]) }); }
            foreach (Dictionary<string, object> t in (List<object>)j["Trims"])
            {
                var d = Arr(t["Domain"]);
                b.Trims.Add(new KTrim { Curve2 = (int)D(t["Curve2"]), Domain = new Interval(d[0], d[1]), Edge = (int)D(t["Edge"]), Loop = (int)D(t["Loop"]), Reversed3d = (bool)t["Reversed3d"],
                    Type = (TrimType)Enum.Parse(typeof(TrimType), (string)t["Type"]), TolU = D(t["TolU"]), TolV = D(t["TolV"]) });
            }
            foreach (Dictionary<string, object> l in (List<object>)j["Loops"])
                b.Loops.Add(new KLoop { Face = (int)D(l["Face"]), Type = (LoopType)Enum.Parse(typeof(LoopType), (string)l["Type"]), Trims = Ints(l["Trims"]) });
            foreach (Dictionary<string, object> f in (List<object>)j["Faces"])
                b.Faces.Add(new KFace { Surface = (int)D(f["Surface"]), Reversed = (bool)f["Reversed"], Loops = Ints(f["Loops"]) });
            return b;
        }

        // ------------------------------------------------------------------ test branches from sampled curves

        static V3 Lagrange(List<double> s, List<V3> p, double x)
        {
            var r = new V3(0, 0, 0);
            for (int i = 0; i < s.Count; i++)
            {
                double w = 1;
                for (int k = 0; k < s.Count; k++) if (k != i) w *= (x - s[k]) / (s[i] - s[k]);
                r = r + p[i] * w;
            }
            return r;
        }

        /// <summary>Corner between the end of `left` (ordered towards the corner) and the start of `right` (ordered away).</summary>
        static V3 Corner(List<V3> left, List<V3> right)
        {
            List<double> Chord(List<V3> p) { var s = new List<double> { 0 }; for (int i = 1; i < p.Count; i++) s.Add(s[i - 1] + SplitGeom.Dist(p[i], p[i - 1])); return s; }
            var sl = Chord(left); var rr = new List<V3>(right); rr.Reverse(); var sr = Chord(rr);
            double a = sl[sl.Count - 1], b = sr[sr.Count - 1];
            double h = SplitGeom.Dist(left[left.Count - 1], right[0]);
            double x = a + 0.5 * h, y = b + 0.5 * h;
            for (int it = 0; it < 50; it++)
            {
                var F = Lagrange(sl, left, x) - Lagrange(sr, rr, y);
                double e = 1e-7 * (1 + h);
                var Lx = (Lagrange(sl, left, x + e) - Lagrange(sl, left, x - e)) * (0.5 / e);
                var Ry = (Lagrange(sr, rr, y + e) - Lagrange(sr, rr, y - e)) * (-0.5 / e);
                double a11 = V3.Dot(Lx, Lx), a12 = V3.Dot(Lx, Ry), a22 = V3.Dot(Ry, Ry), g1 = V3.Dot(Lx, F), g2 = V3.Dot(Ry, F);
                double det = a11 * a22 - a12 * a12;
                if (Math.Abs(det) < 1e-30) break;
                double dx = (a22 * g1 - a12 * g2) / det, dy = (a11 * g2 - a12 * g1) / det;
                x -= dx; y -= dy;
                if (Math.Abs(dx) + Math.Abs(dy) < 1e-14 * (1 + a + b)) break;
            }
            return (Lagrange(sl, left, x) + Lagrange(sr, rr, y)) * 0.5;
        }

        public static List<SsiBranch> BuildBranches(KBrep brep, List<Dictionary<string, object>> curves, double tol, bool onA, List<string> log)
        {
            var res = new List<SsiBranch>();
            foreach (var cj in curves)
            {
                int deg = (int)D(cj["Degree"]); bool closed = (bool)cj["IsClosed"];
                var pts = ((List<object>)cj["Points"]).Select(o => { var a = Arr(o); return new V3(a[0], a[1], a[2]); }).ToList();
                if (closed && SplitGeom.Dist(pts[0], pts[pts.Count - 1]) < 1e-9) pts.RemoveAt(pts.Count - 1);
                int n = pts.Count;
                V3 P(int i) => pts[((i % n) + n) % n];

                // turning angles and kinks
                var turn = new double[n];
                for (int i = 0; i < n; i++)
                {
                    if (!closed && (i == 0 || i == n - 1)) continue;
                    var d0 = P(i) - P(i - 1); var d1 = P(i + 1) - P(i);
                    double c = V3.Dot(d0, d1) / (d0.Length * d1.Length);
                    turn[i] = Math.Acos(Math.Max(-1, Math.Min(1, c)));
                }
                var sorted = turn.OrderBy(x => x).ToArray();
                double thr = deg == 1 ? 1e-7 : Math.Max(0.25, 8 * sorted[n / 2]);
                // corners: (index of last sample before the corner, corner point, exact-at-sample)
                var corners = new List<(int i, V3 p, bool atSample)>();
                for (int i = 0; i < n; i++)
                {
                    if (turn[i] <= thr) continue;
                    int ip = ((i - 1) % n + n) % n;
                    if (turn[ip] > thr && (closed || i > 0)) continue; // second of a pair, handled below
                    int inx = (i + 1) % n;
                    bool pair = turn[inx] > thr && (closed || i + 1 < n);
                    if (pair && turn[(i + 2) % n] > thr) log.Add($"kink cluster at sample {i} (more than two large turns)");
                    var left = new List<V3>(); var right = new List<V3>();
                    int la = pair ? i : i - 1;
                    for (int k = 3; k >= 0; k--) left.Add(P(la - k));
                    for (int k = 0; k < 4; k++) right.Add(P(i + 1 + k));
                    corners.Add(pair ? (i, Corner(left, right), false) : (i, P(i), true));
                }

                // smooth runs between corners
                var runs = new List<(List<V3> p, bool closedRun)>();
                if (corners.Count == 0)
                {
                    var r = new List<V3>(pts); if (closed) r.Add(pts[0]);
                    runs.Add((r, closed));
                }
                else
                {
                    int m = corners.Count;
                    for (int k = closed ? 0 : -1; k < m; k++)
                    {
                        var r = new List<V3>();
                        int from = k < 0 ? 0 : corners[k].i + 1;
                        if (k >= 0) r.Add(corners[k].p);
                        bool toEnd = !closed && k == m - 1;
                        var cn = toEnd ? default : corners[(k + 1) % m];
                        int to = toEnd ? n - 1 : (cn.atSample ? cn.i - 1 : cn.i);
                        int count = closed ? ((to - from) % n + n) % n + 1 : to - from + 1;
                        if (closed && m >= 2 && ((to + 1) % n + n) % n == ((from % n) + n) % n) count = 0;
                        for (int q = 0; q < count; q++) r.Add(P(from + q));
                        if (!toEnd) r.Add(cn.p);
                        for (int q = r.Count - 2; q >= 1; q--)
                            if (SplitGeom.Dist(r[q], r[q + 1]) < 1e-9 || SplitGeom.Dist(r[q], r[q - 1]) < 1e-9) r.RemoveAt(q);
                        runs.Add((r, false));
                    }
                }

                foreach (var (rp, closedRun) in runs)
                {
                    var br = FitRun(brep, rp, closedRun, deg, tol, log);
                    if (br == null) continue;
                    if (!onA) { br.FaceB = br.FaceA; br.UvB = br.UvA; br.FaceA = -1; br.UvA = null; }
                    res.Add(br);
                }
            }
            return res;
        }

        static void Unwrap(NurbsSurface s, double pu, double pv, ref double u, ref double v)
        {
            if (s.IsClosedU) { double P = s.DomainU.Length; u += Math.Round((pu - u) / P) * P; }
            if (s.IsClosedV) { double P = s.DomainV.Length; v += Math.Round((pv - v) / P) * P; }
        }

        static List<V3> Pull(NurbsSurface s, List<V3> pts, out double maxD)
        {
            var uv = new List<V3>(); maxD = 0;
            double u = 0, v = 0, pu = 0, pv = 0;
            for (int i = 0; i < pts.Count; i++)
            {
                double d;
                if (i == 0) d = SplitGeom.ClosestOnSurfaceGlobal(s, pts[i], out u, out v);
                else
                {
                    u = s.IsClosedU ? SplitGeom.Wrap(pu, s.DomainU) : pu; v = s.IsClosedV ? SplitGeom.Wrap(pv, s.DomainV) : pv;
                    d = SplitGeom.ClosestOnSurface(s, pts[i], ref u, ref v);
                    Unwrap(s, pu, pv, ref u, ref v);
                }
                maxD = Math.Max(maxD, d);
                uv.Add(new V3(u, v, 0)); pu = u; pv = v;
            }
            return uv;
        }

        static bool InFace(KBrep b, int fi, V3 uvIn, V3 x, double tol)
        {
            var F = b.Faces[fi]; var s = b.Surfaces[F.Surface];
            var uv = new V3(s.IsClosedU ? SplitGeom.Wrap(uvIn.X, s.DomainU) : uvIn.X, s.IsClosedV ? SplitGeom.Wrap(uvIn.Y, s.DomainV) : uvIn.Y, 0);
            int w = 0;
            foreach (var li in F.Loops)
            {
                var poly = new List<V3>();
                foreach (var ti in b.Loops[li].Trims)
                {
                    var T = b.Trims[ti];
                    SplitGeom.Sample(b.Curves2[T.Curve2], T.Domain.T0, T.Domain.T1, out var ts, out var ps);
                    for (int i = 0; i + 1 < ps.Length; i++) poly.Add(ps[i]);
                }
                w += SplitGeom.Winding(poly, uv);
            }
            if (w != 0) return true;
            foreach (var li in F.Loops)
                foreach (var ti in b.Loops[li].Trims)
                {
                    var T = b.Trims[ti]; var c2 = b.Curves2[T.Curve2];
                    SplitGeom.Sample(c2, T.Domain.T0, T.Domain.T1, out var ts, out var ps);
                    int bi = 0; double bd = double.MaxValue;
                    for (int i = 0; i < ts.Length; i++) { double d = SplitGeom.Dist(SplitGeom.Eval(s, ps[i]), x); if (d < bd) { bd = d; bi = i; } }
                    double t = ts[bi];
                    if (SplitGeom.ProjectCos(s, c2, T.Domain.T0, T.Domain.T1, x, ref t) <= tol) return true;
                }
            return false;
        }

        static SsiBranch FitRun(KBrep b, List<V3> pts, bool closedRun, int deg, double tol, List<string> log)
        {
            // face that contains every sample of the run (most samples on ties)
            int bestF = -1, bestCount = -1; List<V3> bestUv = null;
            for (int fi = 0; fi < b.Faces.Count; fi++)
            {
                var s = b.Surfaces[b.Faces[fi].Surface];
                var uv = Pull(s, pts, out double maxD);
                if (maxD > tol) continue;
                int cnt = 0;
                for (int i = 0; i < pts.Count; i++) if (InFace(b, fi, uv[i], pts[i], tol)) cnt++;
                if (cnt > bestCount) { bestCount = cnt; bestF = fi; bestUv = uv; }
            }
            if (bestF < 0) { log.Add("run lies on no face"); return null; }
            if (bestCount < pts.Count) log.Add($"run of {pts.Count} samples only {bestCount} inside face {bestF} (runs crossing face boundaries are not supported by the test builder)");
            var srf = b.Surfaces[b.Faces[bestF].Surface];
            var par = new List<double> { 0 };
            for (int i = 1; i < pts.Count; i++) par.Add(par[i - 1] + SplitGeom.Dist(pts[i], pts[i - 1]));
            NurbsCurve c3, c2;
            if (deg == 1)
            {
                c3 = SplitGeom.Line(pts[0], pts[pts.Count - 1], 3, 0, par[par.Count - 1]);
                c2 = SplitGeom.Line(bestUv[0], bestUv[bestUv.Count - 1], 2, 0, par[par.Count - 1]);
            }
            else
            {
                c3 = SplitGeom.Interpolate(pts, par, 3, 3);
                c2 = SplitGeom.Interpolate(bestUv, par, 3, 2);
            }
            double dev = 0;
            for (int k = 0; k <= 2000; k++)
            {
                double t = c3.Domain.ParameterAt(k / 2000.0);
                var uv = NurbsEval.CurvePoint(c2, t);
                double u = srf.IsClosedU ? SplitGeom.Wrap(uv.X, srf.DomainU) : uv.X, v = srf.IsClosedV ? SplitGeom.Wrap(uv.Y, srf.DomainV) : uv.Y;
                dev = Math.Max(dev, SplitGeom.Dist(NurbsEval.SurfacePoint(srf, u, v), NurbsEval.CurvePoint(c3, t)));
            }
            if (dev > tol) log.Add($"fitted branch deviates {dev:E2} > tol");
            return new SsiBranch { FaceA = bestF, FaceB = -1, Curve3 = c3, UvA = c2, IsClosed = closedRun, MaxDeviation = dev };
        }

        // ------------------------------------------------------------------ minimal JSON reader (net48 has no System.Text.Json)

        static class MiniJson
        {
            public static object Parse(string s) { int i = 0; var v = Val(s, ref i); return v; }
            static void Ws(string s, ref int i) { while (i < s.Length && char.IsWhiteSpace(s[i])) i++; }
            static object Val(string s, ref int i)
            {
                Ws(s, ref i);
                char c = s[i];
                if (c == '{')
                {
                    var d = new Dictionary<string, object>(); i++; Ws(s, ref i);
                    if (s[i] == '}') { i++; return d; }
                    while (true)
                    {
                        Ws(s, ref i); var k = Str(s, ref i); Ws(s, ref i); i++; // ':'
                        d[k] = Val(s, ref i); Ws(s, ref i);
                        if (s[i++] == '}') return d;
                    }
                }
                if (c == '[')
                {
                    var l = new List<object>(); i++; Ws(s, ref i);
                    if (s[i] == ']') { i++; return l; }
                    while (true) { l.Add(Val(s, ref i)); Ws(s, ref i); if (s[i++] == ']') return l; }
                }
                if (c == '"') return Str(s, ref i);
                if (string.CompareOrdinal(s, i, "true", 0, 4) == 0) { i += 4; return true; }
                if (string.CompareOrdinal(s, i, "false", 0, 5) == 0) { i += 5; return false; }
                if (string.CompareOrdinal(s, i, "null", 0, 4) == 0) { i += 4; return null; }
                int st = i;
                while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
                return double.Parse(s.Substring(st, i - st), CultureInfo.InvariantCulture);
            }
            static string Str(string s, ref int i)
            {
                var sb = new StringBuilder(); i++;
                while (s[i] != '"')
                {
                    if (s[i] == '\\') { i++; char e = s[i]; sb.Append(e == 'n' ? '\n' : e == 't' ? '\t' : e == 'u' ? (char)Convert.ToInt32(s.Substring(i + 1, 4), 16) : e); if (e == 'u') i += 4; }
                    else sb.Append(s[i]);
                    i++;
                }
                i++;
                return sb.ToString();
            }
        }
    }
}
