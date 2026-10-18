using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using GhostGPU.Core;

namespace GhostGPU.Kernel.Validation
{
    public static class Program
    {
        public static int Main(string[] args)
        {
            var swTotal = Stopwatch.StartNew();
            Console.WriteLine("================================================================================");
            Console.WriteLine("GhostGPU.Kernel Validation Harness");
            Console.WriteLine("================================================================================");

            string explicitGoldenDir = null;
            string explicitOutPath = null;
            for (int i = 0; i < args.Length; i++)
            {
                if ((args[i] == "--golden-dir" || args[i] == "-g") && i + 1 < args.Length)
                {
                    explicitGoldenDir = args[++i];
                }
                else if ((args[i] == "--out" || args[i] == "-o") && i + 1 < args.Length)
                {
                    explicitOutPath = args[++i];
                }
            }

            string goldenDir = GoldenDataLoader.ResolveGoldenDir(explicitGoldenDir);
            Console.WriteLine($"Golden data directory: {goldenDir}");
            bool goldenExists = Directory.Exists(goldenDir);
            if (!goldenExists)
            {
                Console.WriteLine($"[WARNING] Golden data directory not found at: {goldenDir}");
            }

            // Load datasets
            var surfaces = GoldenDataLoader.LoadSurfaces(goldenDir);
            var curves = GoldenDataLoader.LoadCurves(goldenDir);
            var ssiCases = GoldenDataLoader.LoadSsi(goldenDir);
            var splitCases = GoldenDataLoader.LoadSplit(goldenDir);
            var joinCases = GoldenDataLoader.LoadJoin(goldenDir);

            Console.WriteLine($"Loaded golden data: " +
                              $"{(surfaces != null ? surfaces.Count.ToString() : "0")} surfaces, " +
                              $"{(curves != null ? curves.Count.ToString() : "0")} curves, " +
                              $"{(ssiCases != null ? ssiCases.Count.ToString() : "0")} SSI pairs, " +
                              $"{(splitCases != null ? splitCases.Count.ToString() : "0")} Split cases, " +
                              $"{(joinCases != null ? joinCases.Count.ToString() : "0")} Join cases.");
            Console.WriteLine();

            var report = new ValidationReport
            {
                TimestampUtc = DateTime.UtcNow.ToString("o"),
                BackendRequested = "CpuDebug + CudaOnly",
                BackendActual = "Host, CpuDebug, CUDA (where available)",
                DeviceName = Environment.MachineName,
                IsHardwareAccelerated = false
            };

            // Run Milestones with per-suite exception isolation
            Console.WriteLine("--- Running Milestone 1: Eval, Derivatives, Bezier Decomposition, GPU ---");
            RunMilestone(report, "M1", () => SuiteM1.Run(surfaces, curves));

            Console.WriteLine("--- Running Milestone 2: Exact SSI ---");
            RunMilestone(report, "M2", () => SuiteM2.Run(ssiCases));

            Console.WriteLine("--- Running Milestone 3: Face Split ---");
            RunMilestone(report, "M3", () => SuiteM3.Run(splitCases, goldenDir));

            Console.WriteLine("--- Running Milestone 4: Brep Join ---");
            RunMilestone(report, "M4", () => SuiteM4.Run(joinCases));

            // Tally overall totals
            foreach (var m in report.Milestones)
            {
                report.TotalTests += m.TotalTests;
                report.Passed += m.Passed;
                report.Failed += m.Failed;
                report.NotRun += m.NotRun;
                report.Skipped += m.Skipped;
                report.TestResults.AddRange(m.Tests);
            }

            swTotal.Stop();
            report.TotalDurationMs = swTotal.Elapsed.TotalMilliseconds;

            // Output JSON report
            string outPath = explicitOutPath;
            if (string.IsNullOrEmpty(outPath))
            {
                string baseDir = AppContext.BaseDirectory;
                outPath = Path.Combine(baseDir, "kernel_validation.results.json");
            }

            var jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                Converters = { new JsonStringEnumConverter() }
            };

            string json = JsonSerializer.Serialize(report, jsonOptions);
            try
            {
                string dir = Path.GetDirectoryName(Path.GetFullPath(outPath));
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(outPath, json);
                Console.WriteLine($"\nValidation JSON report written to: {outPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n[ERROR] Failed to write report file {outPath}: {ex.Message}");
            }

            Console.WriteLine("\n================================================================================");
            Console.WriteLine($"SUMMARY: Total={report.TotalTests}, Passed={report.Passed}, Failed={report.Failed}, NotRun={report.NotRun}, Skipped={report.Skipped} in {report.TotalDurationMs:F1}ms");
            Console.WriteLine("================================================================================");

            if (report.Failed > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"RESULT: FAILED ({report.Failed} failure(s))");
                Console.ResetColor();
                return 1;
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("RESULT: PASSED (all executed suites succeeded)");
            Console.ResetColor();
            return 0;
        }

        private static void RunMilestone(ValidationReport report, string name, Func<MilestoneReport> runSuite)
        {
            MilestoneReport m;
            try
            {
                m = runSuite();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Unhandled exception running suite {name}: {ex.Message}");
                m = new MilestoneReport
                {
                    Milestone = name,
                    Status = $"Failed (Unhandled Exception: {ex.GetType().Name})",
                    Failed = 1,
                    TotalTests = 1
                };
                m.Tests.Add(new TestCaseResult
                {
                    Name = $"{name}_UnhandledException",
                    Milestone = name,
                    Backend = "Host",
                    Outcome = TestOutcome.Fail,
                    Message = $"Unhandled exception in suite {name}: {ex.GetType().Name}: {ex.Message}",
                    Details = ex.ToString()
                });
            }

            report.Milestones.Add(m);
            PrintMilestoneSummary(m);
        }

        private static void PrintMilestoneSummary(MilestoneReport m)
        {
            Console.WriteLine($"[{m.Milestone}] Status: {m.Status} | Passed={m.Passed}, Failed={m.Failed}, NotRun={m.NotRun}, Skipped={m.Skipped}");
            foreach (var t in m.Tests)
            {
                string tag = t.Outcome switch
                {
                    TestOutcome.Pass => "[PASS]",
                    TestOutcome.Fail => "[FAIL]",
                    TestOutcome.NotRun => "[NOT RUN]",
                    TestOutcome.Skipped => "[SKIP]",
                    _ => "[UNKNOWN]"
                };
                string errInfo = t.MaxError > 0 ? $" (maxErr={t.MaxError:E2})" : "";
                Console.WriteLine($"  {tag,-10} {t.Name} [{t.Backend}] in {t.DurationMs:F2}ms{errInfo}");
                if (!string.IsNullOrEmpty(t.Message))
                    Console.WriteLine($"             -> {t.Message}");
            }
            Console.WriteLine();
        }
    }
}
