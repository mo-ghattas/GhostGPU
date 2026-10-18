using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using GhostGPU.Kernel;

namespace GhostGPU.Kernel.Validation
{
    public static class SuiteM4
    {
        public static MilestoneReport Run(List<GoldenJoinCase> joinCases)
        {
            var report = new MilestoneReport
            {
                Milestone = "M4",
                Status = "Running"
            };

            var joinType = typeof(NurbsSurface).Assembly.GetType("GhostGPU.Kernel.BrepJoin");
            var joinMethod = joinType?.GetMethod("Join", BindingFlags.Public | BindingFlags.Static);

            if (joinType == null || joinMethod == null)
            {
                report.Status = "not run: BrepJoin API not implemented yet";
                report.NotRun++;
                report.Tests.Add(new TestCaseResult
                {
                    Name = "M4_BrepJoin_Api_Implemented",
                    Milestone = "M4",
                    Backend = "Host",
                    Outcome = TestOutcome.NotRun,
                    Message = "not run: GhostGPU.Kernel.BrepJoin class not implemented yet."
                });
                return report;
            }

            if (joinCases == null || joinCases.Count == 0)
            {
                report.Status = "not run: golden join.json data missing";
                report.NotRun++;
                report.Tests.Add(new TestCaseResult
                {
                    Name = "M4_GoldenData_Available",
                    Milestone = "M4",
                    Backend = "Host",
                    Outcome = TestOutcome.NotRun,
                    Message = "Golden join.json not found in tests/fixtures/golden."
                });
                return report;
            }

            foreach (var jc in joinCases)
            {
                var sw = Stopwatch.StartNew();
                bool fail = false;
                string failMsg = null;
                try
                {
                    var joined = BrepJoin.Join(jc.Inputs, jc.Tolerance);
                    sw.Stop();

                    if (joined == null)
                    {
                        fail = true;
                        failMsg = "BrepJoin.Join returned null";
                    }
                    else if (joined.Count != jc.ResultCount)
                    {
                        fail = true;
                        failMsg = $"Join result count mismatch: got {joined.Count}, expected {jc.ResultCount}";
                    }
                    else
                    {
                        for (int i = 0; i < jc.Results.Count; i++)
                        {
                            bool actSolid = BrepJoin.IsSolid(joined[i]);
                            int actNaked = BrepJoin.GetNakedEdgeCount(joined[i]);
                            bool expSolid = jc.Results[i].IsSolid;
                            int expNaked = jc.Results[i].NakedEdgeCount;

                            if (actSolid != expSolid || actNaked != expNaked)
                            {
                                fail = true;
                                failMsg = $"Result[{i}] mismatch: Solid={actSolid} (exp {expSolid}), Naked={actNaked} (exp {expNaked})";
                                break;
                            }
                        }
                    }
                }
                catch (NotImplementedException)
                {
                    sw.Stop();
                    report.Tests.Add(new TestCaseResult
                    {
                        Name = $"M4_Join_{jc.Name}",
                        Milestone = "M4",
                        Backend = "Host",
                        Outcome = TestOutcome.NotRun,
                        DurationMs = sw.Elapsed.TotalMilliseconds,
                        Message = "not run: BrepJoin.Join threw NotImplementedException"
                    });
                    continue;
                }
                catch (Exception ex)
                {
                    fail = true;
                    failMsg = "Exception: " + ex.Message;
                }

                sw.Stop();
                report.Tests.Add(new TestCaseResult
                {
                    Name = $"M4_Join_{jc.Name}",
                    Milestone = "M4",
                    Backend = "Host",
                    Outcome = fail ? TestOutcome.Fail : TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = failMsg,
                    Details = $"Inputs={jc.InputCount}, ExpectedResults={jc.ResultCount}, Tol={jc.Tolerance}"
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
    }
}
