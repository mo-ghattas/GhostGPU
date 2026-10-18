using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Kernel
{
    /// <summary>
    /// Validates Ssi.Intersect against live-Rhino Intersection.BrepBrep golden data (tests/fixtures/golden/ssi.json).
    /// Per case: connected curve count and closed flags after joining curves at shared endpoints (applied identically
    /// to our branches and to Rhino's curves), two-sided distance (every Rhino sample within 2*tol of our curves; every
    /// sample of our curves within 2*tol of Rhino's curve reconstructed from its 200 samples), MaxDeviation &lt;= tol.
    /// </summary>
    public static class SsiSelfTest
    {
        public static string Run(string ssiJsonPath = null, GpuSession session = null)
        {
            ssiJsonPath = ssiJsonPath ?? FindGolden();
            var root = (List<object>)new Json(File.ReadAllText(ssiJsonPath)).Parse();
            var sb = new StringBuilder();
            sb.AppendLine($"SSI self-test ({(session == null ? "host" : session.Options.Backend.ToString())}) on {ssiJsonPath}");
            sb.AppendLine("case                     rhino(crv/comp) ours(br/comp) maxDev     R->ours    ours->R    closed  ms      phases(sub/seed/march/fit)  result");
            int pass = 0;
            foreach (Dictionary<string, object> c in root)
            {
                string name = (string)c["Name"];
                double tol = Num(c["Tolerance"]);
                var A = LoadBrep((Dictionary<string, object>)c["BrepA"]);
                var B = LoadBrep((Dictionary<string, object>)c["BrepB"]);
                var rhino = new List<(V3[] pts, bool closed, int degree)>();
                foreach (Dictionary<string, object> rc in (List<object>)c["Curves"])
                    rhino.Add((((List<object>)rc["Points"]).Select(p => V(p)).ToArray(), (bool)rc["IsClosed"], (int)Num(rc["Degree"])));

                SsiResult r; double ms;
                try
                {
                    var sw = Stopwatch.StartNew();
                    r = Ssi.Intersect(A, B, tol, session);
                    ms = sw.Elapsed.TotalMilliseconds;
                }
                catch (Exception ex) { sb.AppendLine($"{name,-24} EXCEPTION {ex}"); continue; }

                var tm = Ssi.LastTimings;
                string phases = string.Format(CultureInfo.InvariantCulture, "{0:F1}/{1:F1}/{2:F1}/{3:F1}", tm[0], tm[1] - tm[0], tm[2] - tm[1], tm[3] - tm[2]);
                // our curves sampled densely
                var ours = r.Branches.Select(br => SampleCurve(br.Curve3)).ToList();
                double maxDev = r.Branches.Count == 0 ? 0 : r.Branches.Max(br => br.MaxDeviation);

                double dRO = 0;
                foreach (var rc in rhino) foreach (var p in rc.pts) dRO = Math.Max(dRO, DistToOurs(p, r.Branches, ours));
                double dOR = 0;
                foreach (var s in ours) foreach (var p in s.pts) dOR = Math.Max(dOR, DistToRhino(p, rhino));
                if (rhino.Count == 0 && ours.Count == 0) { dRO = dOR = 0; }

                // connected components of the curve set (grouping-invariant: Rhino may return a 4-way junction
                // as one curve or two), and per component whether it has free ends
                var oc = Components(ours.Count, i => ours[i].pts, i => r.Branches[i].IsClosed, (i, q) => DistToOurs(q, new List<SsiBranch> { r.Branches[i] }, new List<Sampled> { ours[i] }), 10 * tol);
                var rcmp = Components(rhino.Count, i => rhino[i].pts, i => rhino[i].closed, (i, q) => DistToRhino(q, new List<(V3[], bool, int)> { rhino[i] }), 10 * tol);
                var oj = oc; var rj = rcmp;
                bool countOk = oj.Count == rj.Count;
                bool closedOk = countOk && oj.OrderBy(x => x).SequenceEqual(rj.OrderBy(x => x));
                bool ok = countOk && closedOk && dRO <= 2 * tol && dOR <= 2 * tol && maxDev <= tol;
                if (ok) pass++;
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                    "{0,-24} {1,5}/{10,-9} {2,6}/{3,-6}  {4,-10:E2} {5,-10:E2} {6,-10:E2} {7,-7} {8,-7:F1} {11,-27} {9}",
                    name, rhino.Count, r.Branches.Count, oj.Count, maxDev, dRO, dOR,
                    closedOk ? "ok" : $"{string.Join("", oj.Select(x => x ? 'C' : 'o'))}|{string.Join("", rj.Select(x => x ? 'C' : 'o'))}",
                    ms, ok ? "PASS" : "FAIL", rj.Count, phases));
                foreach (var w in r.Warnings.Take(6)) sb.AppendLine("    warning: " + w);
                if (r.Warnings.Count > 6) sb.AppendLine($"    ... {r.Warnings.Count - 6} more warnings");
            }
            sb.AppendLine($"{pass}/{root.Count} cases pass");
            return sb.ToString();
        }

        static string FindGolden()
        {
            var d = new DirectoryInfo(AppContext.BaseDirectory);
            while (d != null)
            {
                var f = Path.Combine(d.FullName, "tests", "fixtures", "golden", "ssi.json");
                if (File.Exists(f)) return f;
                d = d.Parent;
            }
            throw new FileNotFoundException("tests/fixtures/golden/ssi.json not found above " + AppContext.BaseDirectory);
        }

        // ---------------- distances ----------------

        sealed class Sampled { public double[] t; public V3[] pts; }

        static Sampled SampleCurve(NurbsCurve c)
        {
            var ts = new List<double>();
            var k = c.Knots;
            for (int s = c.Degree; s < c.CvCount; s++)
            {
                if (k[s + 1] <= k[s]) continue;
                for (int j = 0; j < 64; j++) ts.Add(k[s] + (k[s + 1] - k[s]) * j / 64.0);
            }
            ts.Add(c.Domain.T1);
            return new Sampled { t = ts.ToArray(), pts = ts.Select(t => NurbsEval.CurvePoint(c, t)).ToArray() };
        }

        static double DistToOurs(V3 p, List<SsiBranch> br, List<Sampled> ours)
        {
            double best = double.MaxValue;
            for (int b = 0; b < ours.Count; b++)
            {
                var s = ours[b];
                int bi = 0; double bd = double.MaxValue;
                for (int i = 0; i < s.pts.Length; i++) { double d = (s.pts[i] - p).Length; if (d < bd) { bd = d; bi = i; } }
                double lo = s.t[Math.Max(0, bi - 1)], hi = s.t[Math.Min(s.t.Length - 1, bi + 1)];
                var c = br[b].Curve3;
                for (int it = 0; it < 80; it++)
                {
                    double m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
                    if ((NurbsEval.CurvePoint(c, m1) - p).Length < (NurbsEval.CurvePoint(c, m2) - p).Length) hi = m2; else lo = m1;
                }
                best = Math.Min(best, Math.Min(bd, (NurbsEval.CurvePoint(c, 0.5 * (lo + hi)) - p).Length));
            }
            return best;
        }

        /// <summary>
        /// Distance to Rhino's curve rebuilt from its uniform-parameter samples: Catmull-Rom on smooth segments;
        /// on kink segments (corners of polylines, tangent discontinuities) the two one-sided quadratic extrapolations.
        /// </summary>
        static double DistToRhino(V3 q, List<(V3[] pts, bool closed, int degree)> rhino)
        {
            double best = double.MaxValue;
            foreach (var rc in rhino)
            {
                var r = rc.pts; int n = r.Length;
                bool closed = rc.closed && (r[0] - r[n - 1]).Length < 1e-9;
                int m = closed ? n - 1 : n;             // distinct samples
                Func<int, int> W = i => closed ? ((i % m) + m) % m : i;
                Func<int, bool> Has = i => closed || (i >= 0 && i < n);
                Func<int, double> Theta = i =>
                {
                    if (!Has(i - 1) || !Has(i + 1)) return 0;
                    V3 a = r[W(i)] - r[W(i - 1)], b = r[W(i + 1)] - r[W(i)];
                    double la = a.Length, lb = b.Length;
                    if (la == 0 || lb == 0) return 0;
                    return Math.Acos(Math.Max(-1, Math.Min(1, V3.Dot(a, b) / (la * lb))));
                };
                // kink samples: turning angle well above the turning two samples away (corners, tangent breaks)
                Func<int, bool> Kink = i => Has(i) && Theta(i) > 3 * Math.Max(Theta(i - 2), Theta(i + 2)) + 0.05;
                int segs = closed ? m : n - 1;
                for (int k = 0; k < segs; k++)
                {
                    V3 p0 = r[W(k)], p1 = r[W(k + 1)];
                    double sl = (p1 - p0).Length;
                    if ((q - p0).Length > best + 2 * sl + 1e-9) continue;
                    var cands = new List<Func<double, V3>>();
                    bool smooth = Has(k - 1) && Has(k + 2) && !Kink(k - 1) && !Kink(k) && !Kink(k + 1) && !Kink(k + 2);
                    if (smooth)
                    {
                        V3 a = r[W(k - 1)], b = p0, c = p1, d = r[W(k + 2)];
                        cands.Add(s => (b * 2 + (c - a) * s + (a * 2 - b * 5 + c * 4 - d) * (s * s) + (b * 3 - a - c * 3 + d) * (s * s * s)) * 0.5);
                    }
                    else
                    {
                        // one-sided quadratics that do not straddle a kink, extrapolated over this segment
                        if (Has(k - 2) && !Kink(k - 1)) { V3 a = r[W(k - 2)], b = r[W(k - 1)], c = p0; cands.Add(s => a * ((s + 1) * s / 2) - b * ((s + 2) * s) + c * ((s + 2) * (s + 1) / 2)); }
                        if (Has(k + 3) && !Kink(k + 2)) { V3 a = p1, b = r[W(k + 2)], c = r[W(k + 3)]; cands.Add(s => a * ((s - 2) * (s - 3) / 2) - b * ((s - 1) * (s - 3)) + c * ((s - 1) * (s - 2) / 2)); }
                        if (Has(k + 2) && !Kink(k) && !Kink(k + 1)) { V3 a = p0, b = p1, c = r[W(k + 2)]; cands.Add(s => a * ((s - 1) * (s - 2) / 2) - b * (s * (s - 2)) + c * (s * (s - 1) / 2)); }
                        if (Has(k - 1) && !Kink(k) && !Kink(k + 1)) { V3 a = r[W(k - 1)], b = p0, c = p1; cands.Add(s => a * (s * (s - 1) / 2) - b * ((s + 1) * (s - 1)) + c * ((s + 1) * s / 2)); }
                        if (cands.Count == 0) cands.Add(s => p0 + (p1 - p0) * s);
                    }
                    foreach (var f in cands) best = Math.Min(best, MinDist(f, q));
                }
            }
            return best;
        }

        static double MinDist(Func<double, V3> f, V3 q)
        {
            int bi = 0; double bd = double.MaxValue;
            for (int i = 0; i <= 32; i++) { double d = (f(i / 32.0) - q).Length; if (d < bd) { bd = d; bi = i; } }
            double lo = Math.Max(0, (bi - 1) / 32.0), hi = Math.Min(1, (bi + 1) / 32.0);
            for (int it = 0; it < 60; it++)
            {
                double m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
                if ((f(m1) - q).Length < (f(m2) - q).Length) hi = m2; else lo = m1;
            }
            return Math.Min(bd, (f(0.5 * (lo + hi)) - q).Length);
        }

        // ---------------- connected components ----------------

        /// <summary>
        /// Union of curves touching within tol (any sample of one near the other). Returns per component whether it is
        /// closed: no endpoint of an open curve is free (free = not near another curve and not near its own other end).
        /// </summary>
        static List<bool> Components(int n, Func<int, V3[]> pts, Func<int, bool> closed, Func<int, V3, double> dist, double tol)
        {
            var parent = Enumerable.Range(0, n).ToArray();
            Func<int, int> Find = null; Find = i => parent[i] == i ? i : (parent[i] = Find(parent[i]));
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    if (Find(i) == Find(j)) continue;
                    var pi = pts(i); int step = Math.Max(1, pi.Length / 400);
                    for (int k = 0; k < pi.Length; k += step)
                        if (dist(j, pi[k]) <= tol) { parent[Find(i)] = Find(j); break; }
                }
            var free = new bool[n];
            for (int i = 0; i < n; i++)
            {
                if (closed(i)) continue;
                var p = pts(i);
                foreach (var e in new[] { p[0], p[p.Length - 1] })
                {
                    bool near = (p[0] - p[p.Length - 1]).Length <= tol;
                    for (int j = 0; j < n && !near; j++) if (j != i && dist(j, e) <= tol) near = true;
                    if (!near) free[i] = true;
                }
            }
            var res = new List<bool>();
            foreach (var g in Enumerable.Range(0, n).GroupBy(i => Find(i))) res.Add(!g.Any(i => free[i]));
            return res;
        }

        // ---------------- golden loader ----------------

        static double Num(object o) => Convert.ToDouble(o, CultureInfo.InvariantCulture);
        static V3 V(object o) { var l = (List<object>)o; return new V3(Num(l[0]), Num(l[1]), Num(l[2])); }
        static double[] Arr(object o) => ((List<object>)o).Select(Num).ToArray();
        static List<int> Ints(object o) => ((List<object>)o).Select(x => (int)Num(x)).ToList();
        static Interval Iv(object o) { var a = Arr(o); return new Interval(a[0], a[1]); }

        public static KBrep LoadBrep(Dictionary<string, object> d)
        {
            var b = new KBrep();
            foreach (Dictionary<string, object> s in (List<object>)d["Surfaces"])
                b.Surfaces.Add(new NurbsSurface((int)Num(s["DegreeU"]), (int)Num(s["DegreeV"]), (int)Num(s["CvCountU"]), (int)Num(s["CvCountV"]),
                    (bool)s["IsRational"], Arr(s["KnotsU"]), Arr(s["KnotsV"]), Arr(s["Cvs"]))
                { IsClosedU = (bool)s["IsClosedU"], IsClosedV = (bool)s["IsClosedV"] });
            foreach (var (key, list) in new[] { ("Curves3", b.Curves3), ("Curves2", b.Curves2) })
                foreach (Dictionary<string, object> c in (List<object>)d[key])
                    list.Add(new NurbsCurve((int)Num(c["Dimension"]), (int)Num(c["Degree"]), (bool)c["IsRational"], Arr(c["Knots"]), Arr(c["Cvs"])));
            foreach (Dictionary<string, object> v in (List<object>)d["Vertices"])
                b.Vertices.Add(new KVertex { Point = V(v["Point"]), Tolerance = Num(v["Tolerance"]), Edges = Ints(v["Edges"]) });
            foreach (Dictionary<string, object> e in (List<object>)d["Edges"])
                b.Edges.Add(new KEdge { Curve3 = (int)Num(e["Curve3"]), Domain = Iv(e["Domain"]), V0 = (int)Num(e["V0"]), V1 = (int)Num(e["V1"]), Trims = Ints(e["Trims"]), Tolerance = Num(e["Tolerance"]) });
            foreach (Dictionary<string, object> t in (List<object>)d["Trims"])
                b.Trims.Add(new KTrim
                {
                    Curve2 = (int)Num(t["Curve2"]), Domain = Iv(t["Domain"]), Edge = (int)Num(t["Edge"]), Loop = (int)Num(t["Loop"]),
                    Reversed3d = (bool)t["Reversed3d"], Type = (TrimType)Enum.Parse(typeof(TrimType), (string)t["Type"]), TolU = Num(t["TolU"]), TolV = Num(t["TolV"])
                });
            foreach (Dictionary<string, object> l in (List<object>)d["Loops"])
                b.Loops.Add(new KLoop { Face = (int)Num(l["Face"]), Type = (LoopType)Enum.Parse(typeof(LoopType), (string)l["Type"]), Trims = Ints(l["Trims"]) });
            foreach (Dictionary<string, object> f in (List<object>)d["Faces"])
                b.Faces.Add(new KFace { Surface = (int)Num(f["Surface"]), Reversed = (bool)f["Reversed"], Loops = Ints(f["Loops"]) });
            return b;
        }

        /// <summary>Minimal JSON reader: objects -> Dictionary, arrays -> List, numbers -> double.</summary>
        sealed class Json
        {
            readonly string _s; int _i;
            public Json(string s) { _s = s; }

            public object Parse()
            {
                Ws();
                char c = _s[_i];
                if (c == '{')
                {
                    _i++; var d = new Dictionary<string, object>();
                    Ws(); if (_s[_i] == '}') { _i++; return d; }
                    while (true)
                    {
                        Ws(); string k = Str(); Ws(); _i++; // ':'
                        d[k] = Parse(); Ws();
                        if (_s[_i++] == '}') return d;
                    }
                }
                if (c == '[')
                {
                    _i++; var l = new List<object>();
                    Ws(); if (_s[_i] == ']') { _i++; return l; }
                    while (true)
                    {
                        l.Add(Parse()); Ws();
                        if (_s[_i++] == ']') return l;
                    }
                }
                if (c == '"') return Str();
                if (string.CompareOrdinal(_s, _i, "true", 0, 4) == 0) { _i += 4; return true; }
                if (string.CompareOrdinal(_s, _i, "false", 0, 5) == 0) { _i += 5; return false; }
                if (string.CompareOrdinal(_s, _i, "null", 0, 4) == 0) { _i += 4; return null; }
                int st = _i;
                while (_i < _s.Length && "+-0123456789.eE".IndexOf(_s[_i]) >= 0) _i++;
                return double.Parse(_s.Substring(st, _i - st), NumberStyles.Float, CultureInfo.InvariantCulture);
            }

            string Str()
            {
                _i++; var sb = new StringBuilder();
                while (_s[_i] != '"')
                {
                    if (_s[_i] == '\\') { _i++; char e = _s[_i]; sb.Append(e == 'n' ? '\n' : e == 't' ? '\t' : e == 'u' ? (char)Convert.ToInt32(_s.Substring(_i + 1, 4), 16) : e); if (e == 'u') _i += 4; }
                    else sb.Append(_s[_i]);
                    _i++;
                }
                _i++;
                return sb.ToString();
            }

            void Ws() { while (_i < _s.Length && char.IsWhiteSpace(_s[_i])) _i++; }
        }
    }
}
