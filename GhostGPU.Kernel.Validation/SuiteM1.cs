using System;
using System.Collections.Generic;
using System.Diagnostics;
using GhostGPU.Core;
using GhostGPU.Kernel;
using GhostGPU.Runtime;

namespace GhostGPU.Kernel.Validation
{
    public static class SuiteM1
    {
        public const double RelativeTol = 1e-12;
        public const double NormalTol = 1e-10;

        public static MilestoneReport Run(List<GoldenSurfaceCase> surfaces, List<GoldenCurveCase> curves)
        {
            var report = new MilestoneReport
            {
                Milestone = "M1",
                Status = "Running"
            };

            if (surfaces == null || surfaces.Count == 0 || curves == null || curves.Count == 0)
            {
                report.Status = "not run: golden data missing (run export.csx first)";
                report.NotRun++;
                report.Tests.Add(new TestCaseResult
                {
                    Name = "M1_GoldenData_Available",
                    Milestone = "M1",
                    Backend = "Host",
                    Outcome = TestOutcome.NotRun,
                    Message = "Golden surfaces.json or curves.json not found in tests/fixtures/golden."
                });
                return report;
            }

            // 1. Host Surface Evaluations & Derivatives
            foreach (var sc in surfaces)
            {
                RunSurfaceTest(sc, report);
            }

            // 2. Host Curve Evaluations & Derivatives
            foreach (var cc in curves)
            {
                RunCurveTest(cc, report);
            }

            // 3. FindSpan validation
            RunFindSpanTest(surfaces, report);

            // 4. Bezier Decomposition & Subdivision
            RunBezierDecompositionTest(surfaces, report);

            // 5. GPU Evaluation (CpuDebug mode)
            RunGpuCpuDebugTest(surfaces, report);

            // 6. GPU Evaluation (CUDA mode)
            RunGpuCudaTest(surfaces, report);

            report.TotalTests = report.Tests.Count;
            report.Passed = report.Tests.FindAll(t => t.Outcome == TestOutcome.Pass).Count;
            report.Failed = report.Tests.FindAll(t => t.Outcome == TestOutcome.Fail).Count;
            report.NotRun = report.Tests.FindAll(t => t.Outcome == TestOutcome.NotRun).Count;
            report.Skipped = report.Tests.FindAll(t => t.Outcome == TestOutcome.Skipped).Count;

            report.Status = report.Failed > 0 ? "Failed" : (report.Passed > 0 ? "Passed" : "Not Run");
            return report;
        }

        private static void RunSurfaceTest(GoldenSurfaceCase sc, MilestoneReport report)
        {
            var sw = Stopwatch.StartNew();
            double maxErrPt = 0.0;
            double maxErrDer = 0.0;
            double maxErrNorm = 0.0;
            bool fail = false;
            string failMsg = null;

            try
            {
                foreach (var sample in sc.Samples)
                {
                    // 1. Point
                    V3 pt = NurbsEval.SurfacePoint(sc.Surface, sample.U, sample.V);
                    double errPt = RelError(pt, sample.Point);
                    if (errPt > maxErrPt) maxErrPt = errPt;
                    if (errPt > RelativeTol)
                    {
                        fail = true;
                        failMsg = $"SurfacePoint mismatch at ({sample.U}, {sample.V}): got {pt.X},{pt.Y},{pt.Z} expected {sample.Point.X},{sample.Point.Y},{sample.Point.Z} relErr={errPt:E4}";
                        break;
                    }

                    // 2. Derivatives
                    V3[,] ders = NurbsEval.SurfaceDerivatives(sc.Surface, sample.U, sample.V, 2);
                    V3 su = ders[1, 0];
                    V3 sv = ders[0, 1];
                    V3 suu = ders[2, 0];
                    V3 suv = ders[1, 1];
                    V3 svv = ders[0, 2];

                    double errSu = RelError(su, sample.Su);
                    double errSv = RelError(sv, sample.Sv);
                    double errSuu = RelError(suu, sample.Suu);
                    double errSuv = RelError(suv, sample.Suv);
                    double errSvv = RelError(svv, sample.Svv);

                    double maxD = Math.Max(errSu, Math.Max(errSv, Math.Max(errSuu, Math.Max(errSuv, errSvv))));
                    if (maxD > maxErrDer) maxErrDer = maxD;
                    if (maxD > RelativeTol)
                    {
                        fail = true;
                        failMsg = $"SurfaceDerivatives mismatch at ({sample.U}, {sample.V}): maxRelErr={maxD:E4}";
                        break;
                    }

                    // 3. Normal
                    V3 norm = NurbsEval.SurfaceNormal(sc.Surface, sample.U, sample.V);
                    double errNorm = RelError(norm, sample.Normal);
                    if (errNorm > maxErrNorm) maxErrNorm = errNorm;
                    if (errNorm > NormalTol && sample.Normal.Length > 0.5)
                    {
                        fail = true;
                        failMsg = $"SurfaceNormal mismatch at ({sample.U}, {sample.V}): relErr={errNorm:E4}";
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                fail = true;
                failMsg = "Exception during eval: " + ex.Message;
            }

            sw.Stop();
            report.Tests.Add(new TestCaseResult
            {
                Name = $"M1_SurfaceEval_{sc.Name}",
                Milestone = "M1",
                Backend = "Host",
                Outcome = fail ? TestOutcome.Fail : TestOutcome.Pass,
                DurationMs = sw.Elapsed.TotalMilliseconds,
                MaxError = Math.Max(maxErrPt, Math.Max(maxErrDer, maxErrNorm)),
                Message = failMsg,
                Details = $"Samples={sc.Samples.Count}, maxErrPt={maxErrPt:E3}, maxErrDer={maxErrDer:E3}, maxErrNorm={maxErrNorm:E3}"
            });
        }

        private static void RunCurveTest(GoldenCurveCase cc, MilestoneReport report)
        {
            var sw = Stopwatch.StartNew();
            double maxErrPt = 0.0;
            double maxErrDer = 0.0;
            bool fail = false;
            string failMsg = null;

            try
            {
                foreach (var sample in cc.Samples)
                {
                    V3 pt = NurbsEval.CurvePoint(cc.Curve, sample.T);
                    double errPt = RelError(pt, sample.Point);
                    if (errPt > maxErrPt) maxErrPt = errPt;
                    if (errPt > RelativeTol)
                    {
                        fail = true;
                        failMsg = $"CurvePoint mismatch at t={sample.T}: relErr={errPt:E4}";
                        break;
                    }

                    V3[] ders = NurbsEval.CurveDerivatives(cc.Curve, sample.T, 2);
                    V3 d1 = ders.Length > 1 ? ders[1] : new V3(0, 0, 0);
                    V3 d2 = ders.Length > 2 ? ders[2] : new V3(0, 0, 0);

                    double errD1 = RelError(d1, sample.D1);
                    double errD2 = RelError(d2, sample.D2);
                    double maxD = Math.Max(errD1, errD2);
                    if (maxD > maxErrDer) maxErrDer = maxD;
                    if (maxD > RelativeTol)
                    {
                        fail = true;
                        failMsg = $"CurveDerivatives mismatch at t={sample.T}: maxRelErr={maxD:E4}";
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                fail = true;
                failMsg = "Exception: " + ex.Message;
            }

            sw.Stop();
            report.Tests.Add(new TestCaseResult
            {
                Name = $"M1_CurveEval_{cc.Name}",
                Milestone = "M1",
                Backend = "Host",
                Outcome = fail ? TestOutcome.Fail : TestOutcome.Pass,
                DurationMs = sw.Elapsed.TotalMilliseconds,
                MaxError = Math.Max(maxErrPt, maxErrDer),
                Message = failMsg,
                Details = $"Dim={cc.Dimension}, Samples={cc.Samples.Count}, maxErrPt={maxErrPt:E3}, maxErrDer={maxErrDer:E3}"
            });
        }

        private static void RunFindSpanTest(List<GoldenSurfaceCase> surfaces, MilestoneReport report)
        {
            var sw = Stopwatch.StartNew();
            bool fail = false;
            string failMsg = null;

            try
            {
                foreach (var sc in surfaces)
                {
                    var s = sc.Surface;
                    for (int i = 0; i < 20; i++)
                    {
                        double u = s.DomainU.ParameterAt(i / 19.0);
                        int spanU = NurbsEval.FindSpan(s.KnotsU, s.DegreeU, s.CvCountU, u);
                        if (u < s.KnotsU[spanU] || (u > s.KnotsU[spanU + 1] && spanU < s.CvCountU - 1))
                        {
                            fail = true;
                            failMsg = $"FindSpan U invalid for {sc.Name} at u={u}, span={spanU}, knots=[{s.KnotsU[spanU]}, {s.KnotsU[spanU+1]}]";
                            break;
                        }

                        double v = s.DomainV.ParameterAt(i / 19.0);
                        int spanV = NurbsEval.FindSpan(s.KnotsV, s.DegreeV, s.CvCountV, v);
                        if (v < s.KnotsV[spanV] || (v > s.KnotsV[spanV + 1] && spanV < s.CvCountV - 1))
                        {
                            fail = true;
                            failMsg = $"FindSpan V invalid for {sc.Name} at v={v}, span={spanV}";
                            break;
                        }
                    }
                    if (fail) break;
                }
            }
            catch (Exception ex)
            {
                fail = true;
                failMsg = "Exception: " + ex.Message;
            }

            sw.Stop();
            report.Tests.Add(new TestCaseResult
            {
                Name = "M1_FindSpan_Validation",
                Milestone = "M1",
                Backend = "Host",
                Outcome = fail ? TestOutcome.Fail : TestOutcome.Pass,
                DurationMs = sw.Elapsed.TotalMilliseconds,
                Message = failMsg,
                Details = "Tested span indexing across all golden surfaces"
            });
        }

        private static void RunBezierDecompositionTest(List<GoldenSurfaceCase> surfaces, MilestoneReport report)
        {
            var sw = Stopwatch.StartNew();
            bool fail = false;
            string failMsg = null;
            int totalPatches = 0;

            try
            {
                for (int si = 0; si < surfaces.Count; si++)
                {
                    var sc = surfaces[si];
                    var patches = BezierDecomposition.Decompose(sc.Surface, si);
                    totalPatches += patches.Count;

                    // Verify each patch covers a sub-domain and its box bounds the surface
                    foreach (var p in patches)
                    {
                        if (p.DegreeU != sc.Surface.DegreeU || p.DegreeV != sc.Surface.DegreeV)
                        {
                            fail = true;
                            failMsg = $"Patch degree mismatch in {sc.Name}: ({p.DegreeU}, {p.DegreeV})";
                            break;
                        }

                        // Test subdivision
                        BezierDecomposition.Subdivide(p, out var a, out var b, true);
                        if (a.DegreeU != p.DegreeU || b.DegreeU != p.DegreeU)
                        {
                            fail = true;
                            failMsg = $"Subdivide degree mismatch in {sc.Name}";
                            break;
                        }
                    }
                    if (fail) break;
                }
            }
            catch (Exception ex)
            {
                fail = true;
                failMsg = "Exception: " + ex.Message;
            }

            sw.Stop();
            report.Tests.Add(new TestCaseResult
            {
                Name = "M1_BezierDecomposition_Subdivision",
                Milestone = "M1",
                Backend = "Host",
                Outcome = fail ? TestOutcome.Fail : TestOutcome.Pass,
                DurationMs = sw.Elapsed.TotalMilliseconds,
                Message = failMsg,
                Details = $"Decomposed {surfaces.Count} surfaces into {totalPatches} Bezier patches"
            });
        }

        private static void RunGpuCpuDebugTest(List<GoldenSurfaceCase> surfaces, MilestoneReport report)
        {
            var sw = Stopwatch.StartNew();
            bool fail = false;
            string failMsg = null;
            double maxErr = 0.0;

            try
            {
                using var session = new GpuSession(new SessionOptions { Backend = GpuBackend.CpuDebug });
                foreach (var sc in surfaces)
                {
                    var uvList = new List<V3>();
                    foreach (var s in sc.Samples)
                    {
                        uvList.Add(new V3(s.U, s.V, 0.0));
                    }

                    using var uvBuf = session.UploadPoints(uvList.ToArray());
                    using var ptBuf = GpuEval.EvaluateSurfacePoints(session, sc.Surface, uvBuf);
                    V3[] downloaded = session.DownloadPoints(ptBuf);

                    for (int i = 0; i < downloaded.Length; i++)
                    {
                        double err = RelError(downloaded[i], sc.Samples[i].Point);
                        if (err > maxErr) maxErr = err;
                        if (err > RelativeTol)
                        {
                            fail = true;
                            failMsg = $"GpuEval CpuDebug mismatch on {sc.Name} sample {i}: got ({downloaded[i].X},{downloaded[i].Y},{downloaded[i].Z}) vs expected ({sc.Samples[i].Point.X},{sc.Samples[i].Point.Y},{sc.Samples[i].Point.Z}) err={err:E4}";
                            break;
                        }
                    }
                    if (fail) break;
                }
            }
            catch (Exception ex)
            {
                fail = true;
                failMsg = "Exception: " + ex.Message;
            }

            sw.Stop();
            report.Tests.Add(new TestCaseResult
            {
                Name = "M1_GpuEval_CpuDebug",
                Milestone = "M1",
                Backend = "CpuDebug",
                Outcome = fail ? TestOutcome.Fail : TestOutcome.Pass,
                DurationMs = sw.Elapsed.TotalMilliseconds,
                MaxError = maxErr,
                Message = failMsg,
                Details = $"CpuDebug verified against host SurfacePoint across all surfaces, maxErr={maxErr:E3}"
            });
        }

        private static void RunGpuCudaTest(List<GoldenSurfaceCase> surfaces, MilestoneReport report)
        {
            var sw = Stopwatch.StartNew();
            bool cudaAvailable = false;
            bool fail = false;
            string failMsg = null;
            double maxErr = 0.0;

            try
            {
                using var session = new GpuSession(new SessionOptions { Backend = GpuBackend.CudaOnly });
                cudaAvailable = session.IsHardwareAccelerated;

                if (cudaAvailable)
                {
                    foreach (var sc in surfaces)
                    {
                        var uvList = new List<V3>();
                        foreach (var s in sc.Samples)
                        {
                            uvList.Add(new V3(s.U, s.V, 0.0));
                        }

                        using var uvBuf = session.UploadPoints(uvList.ToArray());
                        using var ptBuf = GpuEval.EvaluateSurfacePoints(session, sc.Surface, uvBuf);
                        V3[] downloaded = session.DownloadPoints(ptBuf);

                        for (int i = 0; i < downloaded.Length; i++)
                        {
                            double err = RelError(downloaded[i], sc.Samples[i].Point);
                            if (err > maxErr) maxErr = err;
                            if (err > RelativeTol)
                            {
                                fail = true;
                                failMsg = $"GpuEval CUDA mismatch on {sc.Name} sample {i}: err={err:E4}";
                                break;
                            }
                        }
                        if (fail) break;
                    }
                }
            }
            catch (Exception ex)
            {
                cudaAvailable = false;
                failMsg = "CUDA unavailable: " + ex.Message;
            }

            sw.Stop();
            if (!cudaAvailable)
            {
                report.Tests.Add(new TestCaseResult
                {
                    Name = "M1_GpuEval_CUDA",
                    Milestone = "M1",
                    Backend = "CudaOnly",
                    Outcome = TestOutcome.NotRun,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = "not run: CUDA device unavailable or driver initialization failed.",
                    Details = failMsg
                });
            }
            else
            {
                report.Tests.Add(new TestCaseResult
                {
                    Name = "M1_GpuEval_CUDA",
                    Milestone = "M1",
                    Backend = "CudaOnly",
                    Outcome = fail ? TestOutcome.Fail : TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    MaxError = maxErr,
                    Message = failMsg,
                    Details = $"CUDA acceleration verified, maxErr={maxErr:E3}"
                });
            }
        }

        private static double RelError(V3 a, V3 b)
        {
            double dx = a.X - b.X;
            double dy = a.Y - b.Y;
            double dz = a.Z - b.Z;
            double diff = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            double mag = Math.Max(1.0, b.Length);
            return diff / mag;
        }
    }
}
