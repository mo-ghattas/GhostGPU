using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Rhino.Geometry;
using GhostGPU.Core;
using GhostGPU.Runtime;
using GhostGPU.Booleans;

namespace GhostGPU.Grasshopper.BrepPipeline
{
    /// <summary>
    /// Self-check test runner designed to run inside live Rhino (via RhinoCommon and Roslyn script runner)
    /// comparing GPU SUnion/SDiff/SInt vs native Brep.CreateBooleanX across 5 test suites:
    /// 1. Box / Box (overlapping)
    /// 2. Box / Sphere (curved surface)
    /// 3. Box / Cylinder (revolved surface with planar caps)
    /// 4. Multi-cutter (Box A vs multiple cutters [B1, B2])
    /// 5. Disjoint (Separated operands testing short-circuits)
    /// 6. Box / two overlapping Spheres (sequential cutters; 2nd classified against the 1st result)
    /// 7. Box / Cylinder through-hole along X with rotated seam (genus-1 result)
    /// Each case reports PASS-hybrid (hybrid path produced the result), PASS-fallback (correct only because
    /// a pair fell back to native Brep.CreateBooleanX) or FAIL. Only PASS-hybrid meets the hybrid criterion.
    /// With <see cref="BrepKernel.GhostGPU"/> the labels are PASS-kernel (GhostGPU.Kernel SSI/split/join produced every
    /// pair), PASS-fallback (some pair fell back to the RhinoCommon path or to native) or FAIL.
    /// </summary>
    public static class BrepBooleanSelfCheck
    {
        private static double ComputeVolume(IEnumerable<Brep> breps)
        {
            if (breps == null) return 0.0;
            double total = 0.0;
            foreach (var b in breps)
            {
                if (b == null) continue;
                var vmp = VolumeMassProperties.Compute(b);
                if (vmp != null) total += vmp.Volume;
            }
            return total;
        }

        public static string Run(GpuSession externalSession = null) => Run(externalSession, BrepKernel.RhinoCommon);

        public static string Run(GpuSession externalSession, BrepKernel kernel)
        {
            bool ownSession = false;
            GpuSession session = externalSession;
            if (session == null)
            {
                ownSession = true;
                var options = new SessionOptions { Backend = GpuBackend.CudaOnly };
                try
                {
                    session = new GpuSession(options);
                }
                catch
                {
                    options.Backend = GpuBackend.CpuDebug;
                    session = new GpuSession(options);
                }
            }

            var sb = new StringBuilder();
            sb.AppendLine("================================================================================");
            sb.AppendLine("GhostGPU.Grasshopper Brep Boolean Self-Check Suite");
            sb.AppendLine($"Device: {session.DeviceName} (Hardware Accelerated: {session.IsHardwareAccelerated})");
            sb.AppendLine($"Backend: {session.Options.Backend}");
            sb.AppendLine($"Kernel: {kernel}");
            sb.AppendLine("================================================================================");

            double tol = 0.001;
            string passLabel = kernel == BrepKernel.GhostGPU ? "PASS-kernel" : "PASS-hybrid";
            const double maxRelErr = 1e-6;
            int totalTests = 0;
            int hybridPassed = 0;
            int fallbackPassed = 0;

            var testSuites = new List<(string SuiteName, List<Brep> SetA, List<Brep> SetB)>();

            // Suite 1: Box / Box
            testSuites.Add((
                "1. Box / Box (Overlapping)",
                new List<Brep> { new Box(Plane.WorldXY, new Interval(0, 10), new Interval(0, 10), new Interval(0, 10)).ToBrep() },
                new List<Brep> { new Box(Plane.WorldXY, new Interval(5, 15), new Interval(5, 15), new Interval(5, 15)).ToBrep() }
            ));

            // Suite 2: Box / Sphere
            testSuites.Add((
                "2. Box / Sphere (Curved Surface)",
                new List<Brep> { new Box(Plane.WorldXY, new Interval(0, 10), new Interval(0, 10), new Interval(0, 10)).ToBrep() },
                new List<Brep> { new Sphere(new Point3d(5, 5, 10), 4.0).ToBrep() }
            ));

            // Suite 3: Box / Cylinder
            testSuites.Add((
                "3. Box / Cylinder (Revolved + Caps)",
                new List<Brep> { new Box(Plane.WorldXY, new Interval(0, 10), new Interval(0, 10), new Interval(0, 10)).ToBrep() },
                new List<Brep> { new Cylinder(new Circle(new Plane(new Point3d(5, 5, -2), Vector3d.ZAxis), 3.0), 14.0).ToBrep(true, true) }
            ));

            // Suite 4: Multi-Cutter
            testSuites.Add((
                "4. Multi-Cutter (Box vs Cutters [B1, B2])",
                new List<Brep> { new Box(Plane.WorldXY, new Interval(0, 20), new Interval(0, 20), new Interval(0, 20)).ToBrep() },
                new List<Brep>
                {
                    new Box(Plane.WorldXY, new Interval(5, 15), new Interval(-5, 25), new Interval(5, 15)).ToBrep(),
                    new Sphere(new Point3d(10, 10, 20), 5.0).ToBrep()
                }
            ));

            // Suite 5: Disjoint
            testSuites.Add((
                "5. Disjoint Operands",
                new List<Brep> { new Box(Plane.WorldXY, new Interval(0, 10), new Interval(0, 10), new Interval(0, 10)).ToBrep() },
                new List<Brep> { new Box(Plane.WorldXY, new Interval(30, 40), new Interval(30, 40), new Interval(30, 40)).ToBrep() }
            ));

            // Suite 6: Box / two overlapping spheres (kept off the box edges: tangency there is a separate degenerate case)
            testSuites.Add((
                "6. Multi-Cutter (Box vs overlapping Spheres [S1, S2])",
                new List<Brep> { new Box(Plane.WorldXY, new Interval(0, 10), new Interval(0, 10), new Interval(0, 10)).ToBrep() },
                new List<Brep>
                {
                    new Sphere(new Point3d(3.5, 5, 10), 2.5).ToBrep(),
                    new Sphere(new Point3d(6.5, 5, 10), 2.5).ToBrep()
                }
            ));

            // Suite 7: Cylinder fully piercing the box along X, seam rotated off-axis (Difference is genus 1)
            var throughPlane = new Plane(new Point3d(-2, 5, 5), Vector3d.XAxis);
            throughPlane.Rotate(0.3, Vector3d.XAxis);
            testSuites.Add((
                "7. Box / Cylinder Through-Hole (genus 1, rotated seam)",
                new List<Brep> { new Box(Plane.WorldXY, new Interval(0, 10), new Interval(0, 10), new Interval(0, 10)).ToBrep() },
                new List<Brep> { new Cylinder(new Circle(throughPlane, 2.5), 14.0).ToBrep(true, true) }
            ));

            foreach (var suite in testSuites)
            {
                sb.AppendLine($"\n--- {suite.SuiteName} ---");
                double volA = ComputeVolume(suite.SetA);
                double volB = ComputeVolume(suite.SetB);
                sb.AppendLine($"Input A count={suite.SetA.Count}, Vol={volA:F4}; Input B count={suite.SetB.Count}, Vol={volB:F4}");

                foreach (string op in new[] { "Difference", "Union", "Intersection" })
                {
                    totalTests++;
                    List<Brep> gpuResults = null;
                    List<Brep> natResults = new List<Brep>();
                    var warnings = new List<string>();
                    Action<string> onWarn = msg => warnings.Add(msg);

                    var opWatch = System.Diagnostics.Stopwatch.StartNew();
                    string stageTimes = "";
                    switch (op)
                    {
                        case "Difference":
                            gpuResults = BrepBooleanPipeline.BooleanDifference(session, suite.SetA, suite.SetB, tol, onWarn, kernel);
                            stageTimes = $"{opWatch.Elapsed.TotalMilliseconds:F1} ms: {BrepBooleanPipeline.LastTimings}";
                            var currentDiff = new List<Brep>(suite.SetA);
                            foreach (var cutter in suite.SetB)
                            {
                                var nextDiff = new List<Brep>();
                                foreach (var b in currentDiff)
                                {
                                    var pieceDiff = Brep.CreateBooleanDifference(new[] { b }, new[] { cutter }, tol);
                                    if (pieceDiff == null || pieceDiff.Length == 0)
                                        nextDiff.Add(b);
                                    else
                                        nextDiff.AddRange(pieceDiff);
                                }
                                currentDiff = nextDiff;
                            }
                            natResults = currentDiff;
                            break;

                        case "Union":
                            var unionOperands = new List<Brep>(suite.SetA);
                            unionOperands.AddRange(suite.SetB);
                            opWatch.Restart();
                            gpuResults = BrepBooleanPipeline.BooleanUnion(session, unionOperands, tol, onWarn, kernel);
                            stageTimes = $"{opWatch.Elapsed.TotalMilliseconds:F1} ms: {BrepBooleanPipeline.LastTimings}";
                            var natUnion = Brep.CreateBooleanUnion(unionOperands, tol);
                            if (natUnion != null) natResults.AddRange(natUnion);
                            break;

                        case "Intersection":
                            opWatch.Restart();
                            gpuResults = BrepBooleanPipeline.BooleanIntersection(session, suite.SetA, suite.SetB, tol, onWarn, kernel);
                            stageTimes = $"{opWatch.Elapsed.TotalMilliseconds:F1} ms: {BrepBooleanPipeline.LastTimings}";
                            foreach (var b in suite.SetB)
                            {
                                var pieceInter = Brep.CreateBooleanIntersection(suite.SetA, new[] { b }, tol);
                                if (pieceInter != null) natResults.AddRange(pieceInter);
                            }
                            break;
                    }

                    double gpuVol = ComputeVolume(gpuResults);
                    double natVol = ComputeVolume(natResults);
                    int gpuCount = gpuResults?.Count ?? 0;
                    int natCount = natResults.Count;

                    bool gpuSolid = gpuResults != null && (gpuCount == 0 || gpuResults.All(p => p != null && p.IsValid && p.IsSolid));
                    bool natSolid = natCount == 0 || natResults.All(p => p != null && p.IsValid && p.IsSolid);
                    bool gpuValid = gpuResults != null && (gpuCount == 0 || gpuResults.All(p => p != null && p.IsValid));
                    bool natValid = natCount == 0 || natResults.All(p => p != null && p.IsValid);

                    double denom = Math.Max(Math.Abs(natVol), 1.0);
                    double relErr = Math.Abs(gpuVol - natVol) / denom;

                    bool pass = (relErr < maxRelErr) && gpuSolid && gpuValid && (gpuSolid == natSolid) && (gpuValid == natValid) && (gpuCount == natCount);
                    // Any pair that fell back to native means this result does not prove the hybrid path.
                    int fallbacks = warnings.Count(w => w.Contains("falling back to native"));
                    // Kernel mode: a pair re-run on the RhinoCommon path does not prove the kernel either.
                    int kernelFallbacks = warnings.Count(w => w.Contains("falling back to the RhinoCommon"));

                    string passStr;
                    if (!pass) passStr = "FAIL";
                    else if (fallbacks + kernelFallbacks > 0) { passStr = "PASS-fallback"; fallbackPassed++; }
                    else { passStr = passLabel; hybridPassed++; }

                    sb.AppendLine($"[{passStr,-13}] {op,-12} | GPU: Vol={gpuVol,12:F6} Solid={gpuSolid,-5} Valid={gpuValid,-5} (Count={gpuCount}) | Nat: Vol={natVol,12:F6} Solid={natSolid,-5} (Count={natCount}) | RelErr={relErr:E2} | NativeFallbacks={fallbacks} KernelFallbacks={kernelFallbacks}");
                    sb.AppendLine($"       Timings: {stageTimes}");
                    foreach (var w in warnings.Distinct())
                    {
                        sb.AppendLine($"       Notice: {w}");
                    }
                }
            }

            sb.AppendLine("================================================================================");
            int failed = totalTests - hybridPassed - fallbackPassed;
            sb.AppendLine($"Summary: {hybridPassed} / {totalTests} {passLabel} (criterion: rel vol err < {maxRelErr:E0}, IsValid, IsSolid, same count, no fallback).");
            sb.AppendLine($"         {fallbackPassed} PASS-fallback (correct only via a fallback path), {failed} FAIL.");
            sb.AppendLine("================================================================================");

            if (ownSession)
            {
                try { session.Dispose(); } catch { }
            }

            return sb.ToString();
        }
    }
}
