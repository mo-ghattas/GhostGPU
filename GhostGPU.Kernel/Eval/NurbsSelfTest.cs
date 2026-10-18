using System;
using System.Collections.Generic;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Kernel
{
    public static class NurbsSelfTest
    {
        public static (bool success, string summary) RunAll()
        {
            var logs = new List<string>();
            bool allPassed = true;

            // 1. Plane Test
            try
            {
                var plane = CreateTestPlane(10.0, 5.0);
                double maxPtErr = 0;
                double maxSuErr = 0;
                double maxSvErr = 0;
                double maxNormErr = 0;

                for (double u = 0; u <= 10.0; u += 1.0)
                {
                    for (double v = 0; v <= 5.0; v += 0.5)
                    {
                        var pt = NurbsEval.SurfacePoint(plane, u, v);
                        var expectedPt = new V3(u, v, 0.0);
                        maxPtErr = Math.Max(maxPtErr, (pt - expectedPt).Length);

                        var ders = NurbsEval.SurfaceDerivatives(plane, u, v, 2);
                        var su = ders[1, 0];
                        var sv = ders[0, 1];
                        maxSuErr = Math.Max(maxSuErr, (su - new V3(1, 0, 0)).Length);
                        maxSvErr = Math.Max(maxSvErr, (sv - new V3(0, 1, 0)).Length);

                        var n = NurbsEval.SurfaceNormal(plane, u, v);
                        maxNormErr = Math.Max(maxNormErr, (n - new V3(0, 0, 1)).Length);
                    }
                }

                logs.Add($"[Plane] maxPtErr={maxPtErr:E2}, maxSuErr={maxSuErr:E2}, maxSvErr={maxSvErr:E2}, maxNormErr={maxNormErr:E2}");
                if (maxPtErr > 1e-14 || maxSuErr > 1e-14 || maxSvErr > 1e-14 || maxNormErr > 1e-14)
                {
                    allPassed = false;
                    logs.Add("[Plane] FAILED tolerance check");
                }
            }
            catch (Exception ex)
            {
                allPassed = false;
                logs.Add($"[Plane] EXCEPTION: {ex.Message}");
            }

            // 2. Rational Cylinder Test
            try
            {
                double r = 4.0;
                double h = 10.0;
                var cyl = CreateTestCylinder(r, h);
                double maxRadiusErr = 0;
                double maxZErr = 0;
                double maxNormalErr = 0;

                for (double u = 0.0; u <= 1.0; u += 0.02)
                {
                    for (double v = 0.0; v <= h; v += 1.0)
                    {
                        var pt = NurbsEval.SurfacePoint(cyl, u, v);
                        double computedR = Math.Sqrt(pt.X * pt.X + pt.Y * pt.Y);
                        maxRadiusErr = Math.Max(maxRadiusErr, Math.Abs(computedR - r));
                        maxZErr = Math.Max(maxZErr, Math.Abs(pt.Z - v));

                        var n = NurbsEval.SurfaceNormal(cyl, u, v);
                        var expectedN = new V3(pt.X / computedR, pt.Y / computedR, 0.0);
                        maxNormalErr = Math.Max(maxNormalErr, (n - expectedN).Length);
                    }
                }

                logs.Add($"[Cylinder] maxRadiusErr={maxRadiusErr:E2}, maxZErr={maxZErr:E2}, maxNormalErr={maxNormalErr:E2}");
                if (maxRadiusErr > 1e-13 || maxZErr > 1e-13 || maxNormalErr > 1e-13)
                {
                    allPassed = false;
                    logs.Add("[Cylinder] FAILED tolerance check");
                }
            }
            catch (Exception ex)
            {
                allPassed = false;
                logs.Add($"[Cylinder] EXCEPTION: {ex.Message}");
            }

            // 3. Rational Sphere Test (including degenerate poles)
            try
            {
                double r = 5.0;
                var sphere = CreateTestSphere(r);
                double maxSphereRadErr = 0;
                double maxSphereNormErr = 0;

                // Check North Pole
                var ptNorth = NurbsEval.SurfacePoint(sphere, 0.5, 1.0);
                var normNorth = NurbsEval.SurfaceNormal(sphere, 0.5, 1.0);
                double northPtErr = (ptNorth - new V3(0, 0, r)).Length;
                double northNormErr = (normNorth - new V3(0, 0, 1)).Length;

                // Check South Pole
                var ptSouth = NurbsEval.SurfacePoint(sphere, 0.5, 0.0);
                var normSouth = NurbsEval.SurfaceNormal(sphere, 0.5, 0.0);
                double southPtErr = (ptSouth - new V3(0, 0, -r)).Length;
                double southNormErr = (normSouth - new V3(0, 0, -1)).Length;

                // Interior points
                for (double u = 0.0; u <= 1.0; u += 0.05)
                {
                    for (double v = 0.0; v <= 1.0; v += 0.05)
                    {
                        var pt = NurbsEval.SurfacePoint(sphere, u, v);
                        double rad = Math.Sqrt(pt.X * pt.X + pt.Y * pt.Y + pt.Z * pt.Z);
                        maxSphereRadErr = Math.Max(maxSphereRadErr, Math.Abs(rad - r));

                        var n = NurbsEval.SurfaceNormal(sphere, u, v);
                        if (v > 1e-4 && v < 1.0 - 1e-4)
                        {
                            var expectedN = pt * (1.0 / rad);
                            maxSphereNormErr = Math.Max(maxSphereNormErr, (n - expectedN).Length);
                        }
                    }
                }

                logs.Add($"[Sphere] maxSphereRadErr={maxSphereRadErr:E2}, maxSphereNormErr={maxSphereNormErr:E2}, northNormErr={northNormErr:E2}, southNormErr={southNormErr:E2}");
                if (maxSphereRadErr > 1e-13 || maxSphereNormErr > 1e-13 || northNormErr > 1e-13 || southNormErr > 1e-13)
                {
                    allPassed = false;
                    logs.Add("[Sphere] FAILED tolerance check");
                }
            }
            catch (Exception ex)
            {
                allPassed = false;
                logs.Add($"[Sphere] EXCEPTION: {ex.Message}");
            }

            // 4. Bezier Decomposition & Subdivision Test
            try
            {
                double r = 3.0;
                var cyl = CreateTestCylinder(r, 6.0);
                var patches = BezierDecomposition.Decompose(cyl, 42);

                logs.Add($"[Decompose] Cyl patches count={patches.Count}");
                double maxDecompErr = 0;
                double maxSubdivErr = 0;

                foreach (var patch in patches)
                {
                    // Verify patch points match surface points
                    double uMid = 0.5 * (patch.U.T0 + patch.U.T1);
                    double vMid = 0.5 * (patch.V.T0 + patch.V.T1);
                    var srfPt = NurbsEval.SurfacePoint(cyl, uMid, vMid);

                    // Evaluate patch at midpoint (t=0.5, s=0.5)
                    var patchPt = EvaluateBezierPatchAt(patch, 0.5, 0.5);
                    maxDecompErr = Math.Max(maxDecompErr, (srfPt - patchPt).Length);

                    // Verify bounding box contains midpoint
                    if (patchPt.X < patch.BoxMin.X - 1e-9 || patchPt.X > patch.BoxMax.X + 1e-9 ||
                        patchPt.Y < patch.BoxMin.Y - 1e-9 || patchPt.Y > patch.BoxMax.Y + 1e-9 ||
                        patchPt.Z < patch.BoxMin.Z - 1e-9 || patchPt.Z > patch.BoxMax.Z + 1e-9)
                    {
                        allPassed = false;
                        logs.Add("[Decompose] Patch bounding box did not contain patch midpoint!");
                    }

                    // Test subdivision
                    BezierDecomposition.Subdivide(patch, out var pa, out var pb, true);
                    var paPt = EvaluateBezierPatchAt(pa, 0.5, 0.5); // (u=0.25, v=0.5)
                    var expectedPaPt = NurbsEval.SurfacePoint(cyl, patch.U.ParameterAt(0.25), vMid);
                    maxSubdivErr = Math.Max(maxSubdivErr, (paPt - expectedPaPt).Length);
                }

                logs.Add($"[Decompose] maxDecompErr={maxDecompErr:E2}, maxSubdivErr={maxSubdivErr:E2}");
                if (maxDecompErr > 1e-13 || maxSubdivErr > 1e-13)
                {
                    allPassed = false;
                    logs.Add("[Decompose] FAILED tolerance check");
                }
            }
            catch (Exception ex)
            {
                allPassed = false;
                logs.Add($"[Decompose] EXCEPTION: {ex.Message}");
            }

            // 5. GPU vs Host Agreement on CUDA
            try
            {
                using var session = new GpuSession(new SessionOptions { Backend = GpuBackend.CudaOnly });
                double r = 4.0;
                var cyl = CreateTestCylinder(r, 8.0);

                int sampleCount = 2000;
                var uvPoints = new V3[sampleCount];
                var hostPts = new V3[sampleCount];

                var rnd = new Random(12345);
                for (int i = 0; i < sampleCount; i++)
                {
                    double u = rnd.NextDouble();
                    double v = rnd.NextDouble() * 8.0;
                    uvPoints[i] = new V3(u, v, 0.0);
                    hostPts[i] = NurbsEval.SurfacePoint(cyl, u, v);
                }

                using var uvBuf = session.UploadPoints(uvPoints, ElementType.V3D, Residency.Device);
                using var gpuPtsBuf = GpuEval.EvaluateSurfacePoints(session, cyl, uvBuf);
                var gpuPts = session.DownloadPoints(gpuPtsBuf);

                double maxGpuRelErr = 0;
                double maxGpuAbsErr = 0;
                for (int i = 0; i < sampleCount; i++)
                {
                    double absErr = (gpuPts[i] - hostPts[i]).Length;
                    double refLen = hostPts[i].Length;
                    double relErr = refLen > 1e-12 ? absErr / refLen : absErr;

                    maxGpuAbsErr = Math.Max(maxGpuAbsErr, absErr);
                    maxGpuRelErr = Math.Max(maxGpuRelErr, relErr);
                }

                logs.Add($"[GPU Eval] sampleCount={sampleCount}, backend={session.DeviceName}, maxGpuAbsErr={maxGpuAbsErr:E2}, maxGpuRelErr={maxGpuRelErr:E2}");
                if (maxGpuRelErr > 1e-12)
                {
                    allPassed = false;
                    logs.Add("[GPU Eval] FAILED 1e-12 relative tolerance check");
                }

                // PatchBoxes test on GPU
                var patches = BezierDecomposition.Decompose(cyl, 1);
                using var gpuBoxesBuf = GpuEval.PatchBoxes(session, patches);
                var gpuBoxes = session.DownloadPoints(gpuBoxesBuf);

                double maxBoxErr = 0;
                for (int i = 0; i < patches.Count; i++)
                {
                    var hostMin = patches[i].BoxMin;
                    var hostMax = patches[i].BoxMax;
                    var gpuMin = gpuBoxes[i * 2];
                    var gpuMax = gpuBoxes[i * 2 + 1];

                    maxBoxErr = Math.Max(maxBoxErr, (gpuMin - hostMin).Length);
                    maxBoxErr = Math.Max(maxBoxErr, (gpuMax - hostMax).Length);
                }

                logs.Add($"[GPU Boxes] patchCount={patches.Count}, maxBoxErr={maxBoxErr:E2}");
                if (maxBoxErr > 1e-12)
                {
                    allPassed = false;
                    logs.Add("[GPU Boxes] FAILED 1e-12 tolerance check");
                }

                // 6. Curve Derivatives Test (3D and 2D)
                var circle2D = CreateTestCircleCurve(5.0, is2D: true);
                var circle3D = CreateTestCircleCurve(5.0, is2D: false);
                double maxCrvPtErr = 0;
                double maxCrvD1Err = 0;
                double maxCrvD2Err = 0;
                double max2dZErr = 0;

                for (double t = 0.0; t <= 1.0; t += 0.02)
                {
                    // Test 2D curve
                    var pt2D = NurbsEval.CurvePoint(circle2D, t);
                    max2dZErr = Math.Max(max2dZErr, Math.Abs(pt2D.Z));
                    double rad2D = Math.Sqrt(pt2D.X * pt2D.X + pt2D.Y * pt2D.Y);
                    maxCrvPtErr = Math.Max(maxCrvPtErr, Math.Abs(rad2D - 5.0));

                    var ders2D = NurbsEval.CurveDerivatives(circle2D, t, 2);
                    max2dZErr = Math.Max(max2dZErr, Math.Abs(ders2D[0].Z));
                    max2dZErr = Math.Max(max2dZErr, Math.Abs(ders2D[1].Z));
                    max2dZErr = Math.Max(max2dZErr, Math.Abs(ders2D[2].Z));

                    // Tangent must be perpendicular to radius: pt . d1 == 0
                    double dot = ders2D[0].X * ders2D[1].X + ders2D[0].Y * ders2D[1].Y;
                    maxCrvD1Err = Math.Max(maxCrvD1Err, Math.Abs(dot));

                    // Test 3D curve
                    var ders3D = NurbsEval.CurveDerivatives(circle3D, t, 2);
                    double rad3D = Math.Sqrt(ders3D[0].X * ders3D[0].X + ders3D[0].Y * ders3D[0].Y);
                    maxCrvPtErr = Math.Max(maxCrvPtErr, Math.Abs(rad3D - 5.0));
                    double dot3D = ders3D[0].X * ders3D[1].X + ders3D[0].Y * ders3D[1].Y;
                    maxCrvD2Err = Math.Max(maxCrvD2Err, Math.Abs(dot3D));
                }

                logs.Add($"[Curve] maxCrvPtErr={maxCrvPtErr:E2}, maxCrvD1Err={maxCrvD1Err:E2}, max2dZErr={max2dZErr:E2}");
                if (maxCrvPtErr > 1e-13 || maxCrvD1Err > 1e-12 || max2dZErr > 0)
                {
                    allPassed = false;
                    logs.Add("[Curve] FAILED tolerance check");
                }
            }
            catch (Exception ex)
            {
                allPassed = false;
                logs.Add($"[GPU Eval] EXCEPTION: {ex}");
            }

            string summary = string.Join(Environment.NewLine, logs);
            return (allPassed, summary);
        }

        private static V3 EvaluateBezierPatchAt(BezierPatch p, double s, double t)
        {
            // Bernstein basis evaluation in [0, 1] x [0, 1]
            int pu = p.DegreeU;
            int pv = p.DegreeV;
            int strideV = pv + 1;

            double[] bu = Bernstein(pu, s);
            double[] bv = Bernstein(pv, t);

            double x = 0, y = 0, z = 0, w = 0;
            for (int i = 0; i <= pu; i++)
            {
                double bi = bu[i];
                for (int j = 0; j <= pv; j++)
                {
                    double weight = bi * bv[j];
                    int idx = (i * strideV + j) * 4;
                    x += weight * p.Cvs[idx];
                    y += weight * p.Cvs[idx + 1];
                    z += weight * p.Cvs[idx + 2];
                    w += weight * p.Cvs[idx + 3];
                }
            }

            if (w != 0.0)
            {
                double invW = 1.0 / w;
                x *= invW;
                y *= invW;
                z *= invW;
            }

            return new V3(x, y, z);
        }

        private static double[] Bernstein(int p, double t)
        {
            double[] b = new double[p + 1];
            b[0] = 1.0;
            double t1 = 1.0 - t;
            for (int j = 1; j <= p; j++)
            {
                double saved = 0.0;
                for (int k = 0; k < j; k++)
                {
                    double temp = b[k];
                    b[k] = saved + t1 * temp;
                    saved = t * temp;
                }
                b[j] = saved;
            }
            return b;
        }

        private static NurbsSurface CreateTestPlane(double width, double height)
        {
            double[] knotsU = new double[] { 0, 0, width, width };
            double[] knotsV = new double[] { 0, 0, height, height };
            double[] cvs = new double[2 * 2 * 4]
            {
                0, 0, 0, 1,
                0, height, 0, 1,
                width, 0, 0, 1,
                width, height, 0, 1
            };
            return new NurbsSurface(1, 1, 2, 2, false, knotsU, knotsV, cvs);
        }

        private static NurbsSurface CreateTestCylinder(double radius, double height)
        {
            // 4 quadratic rational spans for 360 circle
            double w = Math.Sqrt(2.0) / 2.0;
            double r = radius;

            double[] knotsU = new double[] { 0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1 };
            double[] knotsV = new double[] { 0, 0, height, height };

            // 9 CVs along U, 2 CVs along V
            // CV at (i, j): i along U (0..8), j along V (0..1)
            // Cvs array: (i * 2 + j) * 4
            double[,] circleCvs = new double[9, 4]
            {
                { r, 0, 0, 1 },
                { r * w, r * w, 0, w },
                { 0, r, 0, 1 },
                { -r * w, r * w, 0, w },
                { -r, 0, 0, 1 },
                { -r * w, -r * w, 0, w },
                { 0, -r, 0, 1 },
                { r * w, -r * w, 0, w },
                { r, 0, 0, 1 }
            };

            double[] cvs = new double[9 * 2 * 4];
            for (int i = 0; i < 9; i++)
            {
                double cx = circleCvs[i, 0];
                double cy = circleCvs[i, 1];
                double cw = circleCvs[i, 3];

                // j = 0: z = 0
                int idx0 = (i * 2 + 0) * 4;
                cvs[idx0] = cx;
                cvs[idx0 + 1] = cy;
                cvs[idx0 + 2] = 0.0;
                cvs[idx0 + 3] = cw;

                // j = 1: z = height * cw (since CV is homogeneous: z*w = height*cw)
                int idx1 = (i * 2 + 1) * 4;
                cvs[idx1] = cx;
                cvs[idx1 + 1] = cy;
                cvs[idx1 + 2] = height * cw;
                cvs[idx1 + 3] = cw;
            }

            return new NurbsSurface(2, 1, 9, 2, true, knotsU, knotsV, cvs);
        }

        private static NurbsSurface CreateTestSphere(double radius)
        {
            // Semicircle in V from -R to +R (degree 2, 2 spans, 5 CVs)
            // Revolved 360 deg in U (degree 2, 4 spans, 9 CVs)
            double w = Math.Sqrt(2.0) / 2.0;
            double r = radius;

            double[] knotsU = new double[] { 0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1 };
            double[] knotsV = new double[] { 0, 0, 0, 0.5, 0.5, 1, 1, 1 };

            // Semicircle in XZ plane:
            // v=0: (0, 0, -r)
            // v=0.5: (r, 0, 0)
            // v=1.0: (0, 0, r)
            double[,] semiCvs = new double[5, 4]
            {
                { 0, 0, -r, 1 },
                { r * w, 0, -r * w, w },
                { r, 0, 0, 1 },
                { r * w, 0, r * w, w },
                { 0, 0, r, 1 }
            };

            // Revolve circle unit weights along U
            double[,] unitCircle = new double[9, 4]
            {
                { 1, 0, 0, 1 },
                { w, w, 0, w },
                { 0, 1, 0, 1 },
                { -w, w, 0, w },
                { -1, 0, 0, 1 },
                { -w, -w, 0, w },
                { 0, -1, 0, 1 },
                { w, -w, 0, w },
                { 1, 0, 0, 1 }
            };

            double[] cvs = new double[9 * 5 * 4];
            for (int i = 0; i < 9; i++)
            {
                double cosU = unitCircle[i, 0] / unitCircle[i, 3];
                double sinU = unitCircle[i, 1] / unitCircle[i, 3];
                double wu = unitCircle[i, 3];

                for (int j = 0; j < 5; j++)
                {
                    double rho = semiCvs[j, 0]; // x*wv
                    double z = semiCvs[j, 2];   // z*wv
                    double wv = semiCvs[j, 3];

                    // Combined weight
                    double wij = wu * wv;
                    // Cartesian coordinates: X = (rho/wv) * cosU, Y = (rho/wv) * sinU, Z = z/wv
                    // Homogeneous: X * wij = rho * cosU * wu = rho * unitCircle[i, 0]
                    double xHom = (rho / wv) * cosU * wij;
                    double yHom = (rho / wv) * sinU * wij;
                    double zHom = (z / wv) * wij;

                    int idx = (i * 5 + j) * 4;
                    cvs[idx] = xHom;
                    cvs[idx + 1] = yHom;
                    cvs[idx + 2] = zHom;
                    cvs[idx + 3] = wij;
                }
            }

            return new NurbsSurface(2, 2, 9, 5, true, knotsU, knotsV, cvs);
        }

        private static NurbsCurve CreateTestCircleCurve(double radius, bool is2D)
        {
            double w = Math.Sqrt(2.0) / 2.0;
            double r = radius;

            double[] knots = new double[] { 0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1 };
            double[,] circleCvs = new double[9, 4]
            {
                { r, 0, 0, 1 },
                { r * w, r * w, 0, w },
                { 0, r, 0, 1 },
                { -r * w, r * w, 0, w },
                { -r, 0, 0, 1 },
                { -r * w, -r * w, 0, w },
                { 0, -r, 0, 1 },
                { r * w, -r * w, 0, w },
                { r, 0, 0, 1 }
            };

            double[] cvs = new double[9 * 4];
            for (int i = 0; i < 9; i++)
            {
                cvs[i * 4] = circleCvs[i, 0];
                cvs[i * 4 + 1] = circleCvs[i, 1];
                cvs[i * 4 + 2] = circleCvs[i, 2];
                cvs[i * 4 + 3] = circleCvs[i, 3];
            }

            return new NurbsCurve(is2D ? 2 : 3, 2, true, knots, cvs);
        }
    }
}
