using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using GhostGPU.Core;
using GhostGPU.Validation.Benchmarks;
using GhostGPU.Validation.Models;
using GhostGPU.Validation.Tests;

namespace GhostGPU.Validation
{
    public sealed class ValidationRunner
    {
        public static int Run(string[] args)
        {
            var sw = Stopwatch.StartNew();

            // Default options
            GpuBackend requestedBackend = GpuBackend.CudaOnly;
            string outputPath = "validation.results.json";
            bool runBenchmarks = true;

            // Parse CLI args: --backend cuda|cpu, --output <path>|JSON
            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i];
                if (arg.Equals("--backend", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    string val = args[++i].ToLowerInvariant();
                    if (val == "cuda" || val == "cudaonly")
                        requestedBackend = GpuBackend.CudaOnly;
                    else if (val == "cpu" || val == "cpudebug")
                        requestedBackend = GpuBackend.CpuDebug;
                    else
                        Console.WriteLine($"[WARN] Unrecognized backend '{val}', defaulting to {requestedBackend}");
                }
                else if (arg.Equals("--output", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    string val = args[++i];
                    if (!val.Equals("JSON", StringComparison.OrdinalIgnoreCase))
                    {
                        outputPath = val;
                    }
                }
                else if (arg.Equals("--no-benchmarks", StringComparison.OrdinalIgnoreCase))
                {
                    runBenchmarks = false;
                }
            }

            Console.WriteLine("================================================================================");
            Console.WriteLine(" GhostGPU Independent Conformance & Benchmark Runner (.NET 8.0)");
            Console.WriteLine("================================================================================");
            Console.WriteLine($" Requested Backend : {requestedBackend}");
            Console.WriteLine($" Output Target     : {outputPath}");
            Console.WriteLine($" Timestamp         : {DateTime.UtcNow:u}");
            Console.WriteLine("--------------------------------------------------------------------------------");

            var report = new ValidationReport
            {
                BackendRequested = requestedBackend.ToString(),
                StrictCudaModeRequired = requestedBackend == GpuBackend.CudaOnly
            };

            IGpuSession? session = null;
            try
            {
                var options = new SessionOptions
                {
                    Backend = requestedBackend,
                    MemoryLimitBytes = 512L * 1024 * 1024,
                    MaximumNearestPairs = 100000000
                };

                session = SessionFactory.CreateSession(options);

                report.BackendActual = session.IsHardwareAccelerated ? "CudaOnly" : "CpuDebug";
                report.DeviceName = session.DeviceName;
                report.IsHardwareAccelerated = session.IsHardwareAccelerated;

                Console.WriteLine($" Device Name       : {report.DeviceName}");
                Console.WriteLine($" HW Accelerated    : {report.IsHardwareAccelerated}");
                Console.WriteLine($" Actual Backend    : {report.BackendActual}");

                if (requestedBackend == GpuBackend.CudaOnly && !session.IsHardwareAccelerated)
                {
                    throw new InvalidOperationException("Strict CudaOnly required, but session is not hardware accelerated! Silent fallback prohibited.");
                }

                Console.WriteLine("--------------------------------------------------------------------------------");
                Console.WriteLine(" Executing Conformance Test Suites...");
                Console.WriteLine("--------------------------------------------------------------------------------");

                void RecordTest(TestResultItem item)
                {
                    report.TestResults.Add(item);
                    string statusTag = item.Outcome == TestOutcome.Pass ? "[PASS]" : "[FAIL]";
                    string dur = $"{item.DurationMs,7:F2} ms";
                    Console.WriteLine($" {statusTag} {dur} | [{item.Category}] {item.Name}");
                    if (item.Outcome == TestOutcome.Fail && !string.IsNullOrEmpty(item.Message))
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine($"        Error: {item.Message}");
                        Console.ResetColor();
                    }
                }

                // 1. Upload/Download Integrity
                UploadDownloadTests.RunAll(session, RecordTest);

                // 2. Closest Points & Strict Ties & 1e155 overflow
                ClosestPointsTests.RunAll(session, RecordTest);

                // 3. Repeat-Last Longest List Rule
                RepeatLastTests.RunAll(session, RecordTest);

                // 4. Math, Vectors, Zero Vectors, Amplitude, Divide by zero
                MathVectorTests.RunAll(session, RecordTest);

                // 5. Bounds, Remap, Map, Colour Ramp, Repeat Colours
                BoundsRemapMapTests.RunAll(session, RecordTest);

                // 6. Lease Lifecycle, Retain, Disposed/Foreign/Reset Rejection
                LeaseLifecycleTests.RunAll(session, RecordTest);

                // 7. Memory Guard Limits
                MemoryGuardTests.RunAll(session, RecordTest);

                // 8. Resident Pipeline Zero Download Metrics
                ResidentChainMetricsTests.RunAll(session, RecordTest);

                // 9. Exact Match with Native Rhino 8.35 Recorded Execution
                NativeReferenceComparisonTests.RunAll(session, RecordTest);

                // 10. V2 Runtime Conformance Tests (Milestone B)
                V2RuntimeTests.RunAll(session, RecordTest);
                V2FormatAndConcurrencyTests.RunAll(session, RecordTest);

                // 11. Bounded Benchmarks
                if (runBenchmarks)
                {
                    Console.WriteLine("--------------------------------------------------------------------------------");
                    Console.WriteLine(" Running Bounded Benchmarks vs Naive Independent CPU Reference...");
                    Console.WriteLine("--------------------------------------------------------------------------------");
                    report.BenchmarkResults = BoundedBenchmarkRunner.RunBenchmarks(session);

                    foreach (var bm in report.BenchmarkResults)
                    {
                        Console.WriteLine($"  {bm.Operation,-30} | Device: {bm.DeviceDurationMs,7:F2} ms | CPU: {bm.NaiveCpuDurationMs,7:F2} ms | Speedup: {bm.Speedup,6:F2}x | MaxDiff: {bm.MaxAbsoluteDifference:E2}");
                    }
                }

                var metrics = session.Metrics;
                report.FinalMetrics = new MetricsSnapshot
                {
                    UploadedBytes = metrics.UploadedBytes,
                    DownloadedBytes = metrics.DownloadedBytes,
                    ActiveBytes = metrics.ActiveBytes,
                    PeakBytes = metrics.PeakBytes,
                    LiveAllocations = metrics.LiveAllocations,
                    KernelDispatches = metrics.KernelDispatches,
                    Synchronizations = metrics.Synchronizations
                };
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n[FATAL ERROR] Session initialization or execution failed: {ex.Message}");
                Console.WriteLine(ex.ToString());
                Console.ResetColor();

                report.TestResults.Add(new TestResultItem
                {
                    Name = "Fatal_Execution_Exception",
                    Category = "System",
                    Outcome = TestOutcome.Fail,
                    Message = ex.Message,
                    Details = ex.ToString()
                });
            }
            finally
            {
                session?.Dispose();
            }

            sw.Stop();
            report.TotalDurationMs = sw.Elapsed.TotalMilliseconds;

            // Summary tallies
            report.TotalTests = report.TestResults.Count;
            report.Passed = report.TestResults.FindAll(t => t.Outcome == TestOutcome.Pass).Count;
            report.Failed = report.TestResults.FindAll(t => t.Outcome == TestOutcome.Fail).Count;
            report.Skipped = report.TestResults.FindAll(t => t.Outcome == TestOutcome.Skipped).Count;

            Console.WriteLine("================================================================================");
            Console.WriteLine($" Summary: {report.Passed} Passed, {report.Failed} Failed, {report.Skipped} Skipped (Total: {report.TotalTests}) in {report.TotalDurationMs:F1} ms");
            Console.WriteLine("================================================================================");

            // Serialize JSON report
            try
            {
                var jsonOptions = new JsonSerializerOptions
                {
                    WriteIndented = true
                };

                string jsonContent = JsonSerializer.Serialize(report, jsonOptions);

                // Write to file
                string? dir = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                File.WriteAllText(outputPath, jsonContent);
                Console.WriteLine($"\n[INFO] Machine-readable JSON results saved to: {Path.GetFullPath(outputPath)}");
            }
            catch (Exception writeEx)
            {
                Console.WriteLine($"[WARN] Could not write output file '{outputPath}': {writeEx.Message}");
            }

            return report.Failed == 0 ? 0 : 1;
        }
    }
}
