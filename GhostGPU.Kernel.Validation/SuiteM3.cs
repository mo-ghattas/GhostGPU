using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using GhostGPU.Kernel;

namespace GhostGPU.Kernel.Validation
{
    public static class SuiteM3
    {
        public static MilestoneReport Run(List<GoldenSplitCase> splitCases, string goldenDir = null)
        {
            var report = new MilestoneReport
            {
                Milestone = "M3",
                Status = "Running"
            };

            var splitType = typeof(NurbsSurface).Assembly.GetType("GhostGPU.Kernel.FaceSplit");
            var splitMethod = splitType?.GetMethod("Split", new Type[] { typeof(KBrep), typeof(IReadOnlyList<SsiBranch>), typeof(bool), typeof(double) });

            if (splitType == null || splitMethod == null)
            {
                report.Status = "not run: FaceSplit API not implemented yet";
                report.NotRun++;
                report.Tests.Add(new TestCaseResult
                {
                    Name = "M3_FaceSplit_Api_Implemented",
                    Milestone = "M3",
                    Backend = "Host",
                    Outcome = TestOutcome.NotRun,
                    Message = "not run: GhostGPU.Kernel.FaceSplit class not implemented yet."
                });
                return report;
            }

            if (splitCases == null || splitCases.Count == 0)
            {
                report.Status = "not run: golden split.json data missing";
                report.NotRun++;
                report.Tests.Add(new TestCaseResult
                {
                    Name = "M3_GoldenData_Available",
                    Milestone = "M3",
                    Backend = "Host",
                    Outcome = TestOutcome.NotRun,
                    Message = "Golden split.json not found in tests/fixtures/golden."
                });
                return report;
            }

            foreach (var sc in splitCases)
            {
                var sw = Stopwatch.StartNew();
                report.Tests.Add(new TestCaseResult
                {
                    Name = $"M3_Split_{sc.Name}",
                    Milestone = "M3",
                    Backend = "Host",
                    Outcome = TestOutcome.NotRun,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = "not run: awaiting full SSI branch integration for FaceSplit input."
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
