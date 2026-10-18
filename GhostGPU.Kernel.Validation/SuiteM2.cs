using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using GhostGPU.Core;
using GhostGPU.Kernel;

namespace GhostGPU.Kernel.Validation
{
    public static class SuiteM2
    {
        public static MilestoneReport Run(List<GoldenSsiCase> ssiCases)
        {
            var report = new MilestoneReport
            {
                Milestone = "M2",
                Status = "Running"
            };

            if (ssiCases == null || ssiCases.Count == 0)
            {
                report.Status = "not run: golden ssi.json data missing";
                report.NotRun++;
                report.Tests.Add(new TestCaseResult
                {
                    Name = "M2_GoldenData_Available",
                    Milestone = "M2",
                    Backend = "Host",
                    Outcome = TestOutcome.NotRun,
                    Message = "Golden ssi.json not found in tests/fixtures/golden."
                });
                return report;
            }

            foreach (var sc in ssiCases)
            {
                var sw = Stopwatch.StartNew();
                bool fail = false;
                string failMsg = null;
                double maxMeasuredError = 0;
                string details = null;

                try
                {
                    var result = Ssi.Intersect(sc.BrepA, sc.BrepB, sc.Tolerance, null);
                    sw.Stop();

                    if (result == null)
                    {
                        fail = true;
                        failMsg = "Ssi.Intersect returned null";
                    }
                    else
                    {
                        double tol = sc.Tolerance;

                        // 1. Evaluate max deviation reported on branches
                        double maxDev = result.Branches.Count == 0 ? 0 : result.Branches.Max(br => br.MaxDeviation);

                        // 2. Sample our branch curves densely
                        var ourSamples = result.Branches.Select(br => SampleOurCurve(br.Curve3)).ToList();

                        // 3. Prepare Rhino golden curves
                        var rhinoCurves = sc.Curves.Select(rc => (pts: rc.Points.ToArray(), closed: rc.IsClosed, degree: rc.Degree)).ToList();

                        // 4. Two-sided distance checks
                        double dRO = 0; // Rhino samples to our curves
                        foreach (var rc in rhinoCurves)
                        {
                            foreach (var p in rc.pts)
                                dRO = Math.Max(dRO, DistanceToOurCurves(p, result.Branches, ourSamples));
                        }

                        double dOR = 0; // Our curve samples to Rhino curves
                        foreach (var s in ourSamples)
                        {
                            foreach (var p in s.Pts)
                                dOR = Math.Max(dOR, DistanceToRhinoCurves(p, rhinoCurves));
                        }

                        if (rhinoCurves.Count == 0 && ourSamples.Count == 0)
                        {
                            dRO = dOR = 0;
                        }

                        // 5. Grouping-invariant connected components
                        double weldTol = 10 * tol;
                        var ourComponents = ComputeConnectedComponents(
                            ourSamples.Count,
                            i => ourSamples[i].Pts,
                            i => result.Branches[i].IsClosed,
                            (i, q) => DistanceToOurBranch(q, result.Branches[i], ourSamples[i]),
                            weldTol);

                        var rhinoComponents = ComputeConnectedComponents(
                            rhinoCurves.Count,
                            i => rhinoCurves[i].pts,
                            i => rhinoCurves[i].closed,
                            (i, q) => DistanceToRhinoSingleCurve(q, rhinoCurves[i]),
                            weldTol);

                        bool countOk = ourComponents.Count == rhinoComponents.Count;
                        bool closedOk = countOk && ourComponents.OrderBy(x => x).SequenceEqual(rhinoComponents.OrderBy(x => x));
                        bool distOk = (dRO <= 2 * tol) && (dOR <= 2 * tol);
                        bool devOk = (maxDev <= tol);

                        maxMeasuredError = Math.Max(maxDev, Math.Max(dRO, dOR));
                        details = $"Comp={ourComponents.Count}/{rhinoComponents.Count}, Br={result.Branches.Count}, MaxDev={maxDev:E2}, d(R->O)={dRO:E2}, d(O->R)={dOR:E2}, Closed={closedOk}";

                        if (!countOk)
                        {
                            fail = true;
                            failMsg = $"Connected component count mismatch: got {ourComponents.Count}, expected {rhinoComponents.Count}";
                        }
                        else if (!closedOk)
                        {
                            fail = true;
                            failMsg = "Component closedness flag mismatch";
                        }
                        else if (!devOk)
                        {
                            fail = true;
                            failMsg = $"Branch max deviation {maxDev:E2} > tolerance {tol:E2}";
                        }
                        else if (!distOk)
                        {
                            fail = true;
                            failMsg = $"Two-sided distance exceeded 2*tol: d(Rhino->Ours)={dRO:E2}, d(Ours->Rhino)={dOR:E2}, 2*tol={2 * tol:E2}";
                        }
                    }
                }
                catch (NotImplementedException)
                {
                    sw.Stop();
                    report.Tests.Add(new TestCaseResult
                    {
                        Name = $"M2_SSI_{sc.Name}",
                        Milestone = "M2",
                        Backend = "Host",
                        Outcome = TestOutcome.NotRun,
                        DurationMs = sw.Elapsed.TotalMilliseconds,
                        Message = "not run: Ssi.Intersect threw NotImplementedException"
                    });
                    continue;
                }
                catch (Exception ex)
                {
                    sw.Stop();
                    fail = true;
                    failMsg = $"Exception: {ex.GetType().Name}: {ex.Message}";
                }

                report.Tests.Add(new TestCaseResult
                {
                    Name = $"M2_SSI_{sc.Name}",
                    Milestone = "M2",
                    Backend = "Host",
                    Outcome = fail ? TestOutcome.Fail : TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    MaxError = maxMeasuredError,
                    Message = failMsg,
                    Details = details
                });
            }

            report.TotalTests = report.Tests.Count;
            report.Passed = report.Tests.FindAll(t => t.Outcome == TestOutcome.Pass).Count;
            report.Failed = report.Tests.FindAll(t => t.Outcome == TestOutcome.Fail).Count;
            report.NotRun = report.Tests.FindAll(t => t.Outcome == TestOutcome.NotRun).Count;
            report.Skipped = report.Tests.FindAll(t => t.Outcome == TestOutcome.Skipped).Count;
            report.Status = report.Failed > 0 ? "Failed" : (report.Passed > 0 ? "Passed" : "Not Run");
            return report;
        }

        #region Distance & Connected Component Helpers

        private sealed class SampledCurve
        {
            public double[] T;
            public V3[] Pts;
        }

        private static SampledCurve SampleOurCurve(NurbsCurve c)
        {
            var ts = new List<double>();
            var k = c.Knots;
            for (int s = c.Degree; s < c.CvCount; s++)
            {
                if (k[s + 1] <= k[s]) continue;
                for (int j = 0; j < 64; j++)
                    ts.Add(k[s] + (k[s + 1] - k[s]) * j / 64.0);
            }
            ts.Add(c.Domain.T1);
            return new SampledCurve
            {
                T = ts.ToArray(),
                Pts = ts.Select(t => NurbsEval.CurvePoint(c, t)).ToArray()
            };
        }

        private static double DistanceToOurBranch(V3 p, SsiBranch br, SampledCurve sample)
        {
            int bi = 0;
            double bd = double.MaxValue;
            for (int i = 0; i < sample.Pts.Length; i++)
            {
                double d = (sample.Pts[i] - p).Length;
                if (d < bd) { bd = d; bi = i; }
            }

            double lo = sample.T[Math.Max(0, bi - 1)];
            double hi = sample.T[Math.Min(sample.T.Length - 1, bi + 1)];
            var c = br.Curve3;

            for (int it = 0; it < 80; it++)
            {
                double m1 = lo + (hi - lo) / 3.0;
                double m2 = hi - (hi - lo) / 3.0;
                if ((NurbsEval.CurvePoint(c, m1) - p).Length < (NurbsEval.CurvePoint(c, m2) - p).Length)
                    hi = m2;
                else
                    lo = m1;
            }

            return Math.Min(bd, (NurbsEval.CurvePoint(c, 0.5 * (lo + hi)) - p).Length);
        }

        private static double DistanceToOurCurves(V3 p, List<SsiBranch> branches, List<SampledCurve> samples)
        {
            double best = double.MaxValue;
            for (int b = 0; b < samples.Count; b++)
            {
                best = Math.Min(best, DistanceToOurBranch(p, branches[b], samples[b]));
            }
            return best;
        }

        private static double DistanceToRhinoSingleCurve(V3 q, (V3[] pts, bool closed, int degree) rc)
        {
            double best = double.MaxValue;
            var r = rc.pts;
            int n = r.Length;
            if (n == 0) return best;

            bool closed = rc.closed && (r[0] - r[n - 1]).Length < 1e-9;
            int m = closed ? n - 1 : n;
            Func<int, int> W = i => closed ? ((i % m) + m) % m : i;
            Func<int, bool> Has = i => closed || (i >= 0 && i < n);

            Func<int, double> Theta = i =>
            {
                if (!Has(i - 1) || !Has(i + 1)) return 0;
                V3 a = r[W(i)] - r[W(i - 1)];
                V3 b = r[W(i + 1)] - r[W(i)];
                double la = a.Length, lb = b.Length;
                if (la == 0 || lb == 0) return 0;
                return Math.Acos(Math.Max(-1, Math.Min(1, V3.Dot(a, b) / (la * lb))));
            };

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
                    if (Has(k - 2) && !Kink(k - 1)) { V3 a = r[W(k - 2)], b = r[W(k - 1)], c = p0; cands.Add(s => a * ((s + 1) * s / 2) - b * ((s + 2) * s) + c * ((s + 2) * (s + 1) / 2)); }
                    if (Has(k + 3) && !Kink(k + 2)) { V3 a = p1, b = r[W(k + 2)], c = r[W(k + 3)]; cands.Add(s => a * ((s - 2) * (s - 3) / 2) - b * ((s - 1) * (s - 3)) + c * ((s - 1) * (s - 2) / 2)); }
                    if (Has(k + 2) && !Kink(k) && !Kink(k + 1)) { V3 a = p0, b = p1, c = r[W(k + 2)]; cands.Add(s => a * ((s - 1) * (s - 2) / 2) - b * (s * (s - 2)) + c * (s * (s - 1) / 2)); }
                    if (Has(k - 1) && !Kink(k) && !Kink(k + 1)) { V3 a = r[W(k - 1)], b = p0, c = p1; cands.Add(s => a * (s * (s - 1) / 2) - b * ((s + 1) * (s - 1)) + c * ((s + 1) * s / 2)); }
                    if (cands.Count == 0) cands.Add(s => p0 + (p1 - p0) * s);
                }

                foreach (var f in cands)
                    best = Math.Min(best, MinimizeDistanceToCurve(f, q));
            }

            return best;
        }

        private static double DistanceToRhinoCurves(V3 q, List<(V3[] pts, bool closed, int degree)> rhino)
        {
            double best = double.MaxValue;
            foreach (var rc in rhino)
            {
                best = Math.Min(best, DistanceToRhinoSingleCurve(q, rc));
            }
            return best;
        }

        private static double MinimizeDistanceToCurve(Func<double, V3> f, V3 q)
        {
            int bi = 0;
            double bd = double.MaxValue;
            for (int i = 0; i <= 32; i++)
            {
                double d = (f(i / 32.0) - q).Length;
                if (d < bd) { bd = d; bi = i; }
            }

            double lo = Math.Max(0, (bi - 1) / 32.0);
            double hi = Math.Min(1, (bi + 1) / 32.0);
            for (int it = 0; it < 60; it++)
            {
                double m1 = lo + (hi - lo) / 3.0;
                double m2 = hi - (hi - lo) / 3.0;
                if ((f(m1) - q).Length < (f(m2) - q).Length)
                    hi = m2;
                else
                    lo = m1;
            }

            return Math.Min(bd, (f(0.5 * (lo + hi)) - q).Length);
        }

        private static List<bool> ComputeConnectedComponents(
            int n,
            Func<int, V3[]> pts,
            Func<int, bool> closed,
            Func<int, V3, double> dist,
            double tol)
        {
            var parent = Enumerable.Range(0, n).ToArray();
            int Find(int i)
            {
                int root = i;
                while (root != parent[root]) root = parent[root];
                int curr = i;
                while (curr != root) { int next = parent[curr]; parent[curr] = root; curr = next; }
                return root;
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (Find(i) == Find(j)) continue;
                    var pi = pts(i);
                    int step = Math.Max(1, pi.Length / 400);
                    for (int k = 0; k < pi.Length; k += step)
                    {
                        if (dist(j, pi[k]) <= tol)
                        {
                            parent[Find(i)] = Find(j);
                            break;
                        }
                    }
                }
            }

            var free = new bool[n];
            for (int i = 0; i < n; i++)
            {
                if (closed(i)) continue;
                var p = pts(i);
                if (p.Length == 0) continue;

                foreach (var e in new[] { p[0], p[p.Length - 1] })
                {
                    bool near = (p[0] - p[p.Length - 1]).Length <= tol;
                    for (int j = 0; j < n && !near; j++)
                    {
                        if (j != i && dist(j, e) <= tol) near = true;
                    }
                    if (!near) free[i] = true;
                }
            }

            var res = new List<bool>();
            foreach (var g in Enumerable.Range(0, n).GroupBy(i => Find(i)))
            {
                res.Add(!g.Any(i => free[i]));
            }
            return res;
        }

        #endregion
    }
}
