using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using GhostGPU.Booleans;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Booleans.Validation
{
    public sealed class TestCaseResult
    {
        public string Name { get; set; }
        public string Backend { get; set; }
        public bool Passed { get; set; }
        public double ComputedValue { get; set; }
        public double ExpectedValue { get; set; }
        public double Error { get; set; }
        public double Tolerance { get; set; }
        public bool IsClosed { get; set; }
        public int TriangleCount { get; set; }
        public long DurationMs { get; set; }
        public string Details { get; set; }
    }

    public sealed class SuiteReport
    {
        public string Backend { get; set; }
        public bool IsHardwareAccelerated { get; set; }
        public string DeviceName { get; set; }
        public int TotalTests { get; set; }
        public int PassedTests { get; set; }
        public int FailedTests { get; set; }
        public long KernelDispatches { get; set; }
        public long HostDispatches { get; set; }
        public string Error { get; set; }
        public List<TestCaseResult> Results { get; set; } = new List<TestCaseResult>();
    }

    public static class Program
    {
        // Box-box results are exact polyhedra: volumes must match to 1e-9.
        private const double ExactTol = 1e-9;
        // Sphere booleans vs the exact volume of the same tessellated meshes (convex clipping oracle).
        private const double SphereMeshTol = 1e-9;
        // Tessellated (32x33 UV) vs smooth sphere-box volume; the full sphere's tessellation deficit is ~0.9%.
        private const double SphereTessRelTol = 1.5e-2;
        // Fast winding number (BVH, order-2 far field, theta = 0.1) vs an independent brute solid-angle sum.
        private const double WindingBvhTol = 1e-6;
        private const double BooleanTolerance = 1e-4;

        public static int Main(string[] args)
        {
            Console.WriteLine("GhostGPU.Booleans validation (CpuDebug + CUDA)");

            string outPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "booleans_validation_results.json");
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--diag" && i + 1 < args.Length) { Diag.Run(args[i + 1], args.Length > i + 2 && args[i + 2] == "cuda" ? GpuBackend.CudaOnly : GpuBackend.CpuDebug); return 0; }
                if (args[i] == "--output")
                {
                    if (i + 1 >= args.Length) { Console.Error.WriteLine("--output requires a path"); return 2; }
                    outPath = args[++i];
                }
            }

            var reports = new List<SuiteReport>();
            bool anyFailed = false;
            foreach (var backend in new[] { GpuBackend.CpuDebug, GpuBackend.CudaOnly })
            {
                if (Environment.GetEnvironmentVariable("GG_BACKEND") is string only && only.Length > 0 && only != backend.ToString()) continue;
                Console.WriteLine($"\n=== Backend: {backend} ===");
                SuiteReport report;
                try
                {
                    report = RunSuite(backend);
                }
                catch (Exception ex)
                {
                    report = new SuiteReport { Backend = backend.ToString(), Error = ex.ToString(), FailedTests = 1 };
                    Console.WriteLine($"[FAIL] backend {backend} threw: {ex.Message}");
                }
                reports.Add(report);
                PrintReport(report);
                if (report.FailedTests > 0 || report.Error != null || report.TotalTests == 0)
                    anyFailed = true;
            }

            string json = JsonSerializer.Serialize(reports, new JsonSerializerOptions { WriteIndented = true });
            string dir = Path.GetDirectoryName(Path.GetFullPath(outPath));
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(outPath, json);
            Console.WriteLine($"\nResults saved to: {outPath}");
            Console.WriteLine(anyFailed ? "Validation: FAILED" : "Validation: PASSED");
            return anyFailed ? 1 : 0;
        }

        public static SuiteReport RunSuite(GpuBackend backend)
        {
            var options = new SessionOptions { Backend = backend, MemoryLimitBytes = 1024L * 1024 * 1024 };
            using var session = new GpuSession(options);
            var report = new SuiteReport
            {
                Backend = backend.ToString(),
                IsHardwareAccelerated = session.IsHardwareAccelerated,
                DeviceName = session.DeviceName
            };

            Guard(report, "WindingNumbers_Box_Brute", () => TestWindingBox(session, report));
            Guard(report, "WindingNumbers_Sphere2048_BVH", () => TestWindingSphereBvh(session, report));
            Guard(report, "Broadphase_RandomBoxes", () => TestBroadphase(session, report));

            var unit = MeshFactory.CreateBox(new V3(-1, -1, -1), new V3(1, 1, 1));

            // Transversal overlap: B = [0,2]x[-0.5,1.5]^2, overlap 1 x 1.5 x 1.5 = 2.25.
            var shifted = MeshFactory.CreateBox(new V3(0, -0.5, -0.5), new V3(2, 1.5, 1.5));
            OpTest(session, report, "OverlapBoxes_Union", new[] { unit }, new[] { shifted }, BooleanOperation.Union, 13.75, ExactTol);
            OpTest(session, report, "OverlapBoxes_Difference", new[] { unit }, new[] { shifted }, BooleanOperation.Difference, 5.75, ExactTol);
            OpTest(session, report, "OverlapBoxes_Intersection", new[] { unit }, new[] { shifted }, BooleanOperation.Intersection, 2.25, ExactTol);

            var left = MeshFactory.CreateBox(new V3(-3, -1, -1), new V3(-1, 1, 1));
            var right = MeshFactory.CreateBox(new V3(1, -1, -1), new V3(3, 1, 1));
            OpTest(session, report, "DisjointBoxes_Union", new[] { left }, new[] { right }, BooleanOperation.Union, 16.0, ExactTol);
            OpTest(session, report, "DisjointBoxes_Difference", new[] { left }, new[] { right }, BooleanOperation.Difference, 8.0, ExactTol);
            OpTest(session, report, "DisjointBoxes_Intersection", new[] { left }, new[] { right }, BooleanOperation.Intersection, 0.0, ExactTol);

            var outer = MeshFactory.CreateBox(new V3(-2, -2, -2), new V3(2, 2, 2));
            OpTest(session, report, "NestedBoxes_Union", new[] { outer }, new[] { unit }, BooleanOperation.Union, 64.0, ExactTol);
            OpTest(session, report, "NestedBoxes_Difference", new[] { outer }, new[] { unit }, BooleanOperation.Difference, 56.0, ExactTol);
            OpTest(session, report, "NestedBoxes_Intersection", new[] { outer }, new[] { unit }, BooleanOperation.Intersection, 8.0, ExactTol);

            // Coincident faces: B = [0,2]x[-1,1]^2 shares the y/z face planes with A.
            var flush = MeshFactory.CreateBox(new V3(0, -1, -1), new V3(2, 1, 1));
            OpTest(session, report, "CoincidentBoxes_Union", new[] { unit }, new[] { flush }, BooleanOperation.Union, 12.0, ExactTol, expectCoincidentWarning: true);
            OpTest(session, report, "CoincidentBoxes_Difference", new[] { unit }, new[] { flush }, BooleanOperation.Difference, 4.0, ExactTol, expectCoincidentWarning: true);
            OpTest(session, report, "CoincidentBoxes_Intersection", new[] { unit }, new[] { flush }, BooleanOperation.Intersection, 4.0, ExactTol, expectCoincidentWarning: true);

            // List semantics: overlapping meshes inside each list. |A| = 8 + 3 - 1 = 10, |B| = 6 + 3.375 - 1.125 = 8.25,
            // A n B = [2.5,3]x[-0.5,0.5]^2 = 0.5.
            var a1 = unit;
            var a2 = MeshFactory.CreateBox(new V3(0, -0.5, -0.5), new V3(3, 0.5, 0.5));
            var b1 = MeshFactory.CreateBox(new V3(2.5, -1, -1), new V3(4, 1, 1));
            var b2 = MeshFactory.CreateBox(new V3(3.5, -0.75, -0.75), new V3(5, 0.75, 0.75));
            OpTest(session, report, "ListBoxes_Union", new[] { a1, a2 }, new[] { b1, b2 }, BooleanOperation.Union, 17.75, ExactTol);
            OpTest(session, report, "ListBoxes_Difference", new[] { a1, a2 }, new[] { b1, b2 }, BooleanOperation.Difference, 9.5, ExactTol);
            OpTest(session, report, "ListBoxes_Intersection", new[] { a1, a2 }, new[] { b1, b2 }, BooleanOperation.Intersection, 0.5, ExactTol);

            // Sphere cases (32x33 UV sphere = 2048 triangles, so classification against it uses the BVH kernel).
            SphereCases(session, report, "SphereCentred_r1.2", new V3(0, 0, 0), 1.2);
            SphereCases(session, report, "SphereOffCentre_x0.7_r0.6", new V3(0.7, 0, 0), 0.6);

            Guard(report, "EmptyAndNullInputs", () => TestEmpty(session, report));
            Guard(report, "DenseCases", () => DenseCases.Run(session, report));

            report.TotalTests = report.Results.Count;
            report.PassedTests = report.Results.FindAll(r => r.Passed).Count;
            report.FailedTests = report.TotalTests - report.PassedTests;
            var metrics = session.Metrics;
            report.KernelDispatches = metrics.KernelDispatches;
            report.HostDispatches = metrics.HostDispatches;
            return report;
        }

        internal static void Guard(SuiteReport report, string name, Action test)
        {
            try { test(); }
            catch (Exception ex)
            {
                report.Results.Add(new TestCaseResult { Name = name, Backend = report.Backend, Passed = false, Details = "threw: " + ex });
            }
        }

        private static void OpTest(GpuSession s, SuiteReport report, string name, TriMesh[] a, TriMesh[] b,
                                   BooleanOperation op, double expected, double tol, bool expectCoincidentWarning = false)
        {
            Guard(report, name, () =>
            {
                var sw = Stopwatch.StartNew();
                var r1 = MeshBoolean.Compute(s, a, b, op, BooleanTolerance);
                sw.Stop();
                var r2 = MeshBoolean.Compute(s, a, b, op, BooleanTolerance);
                double vol = MeshFactory.SignedVolume(r1.Mesh);
                double vol2 = MeshFactory.SignedVolume(r2.Mesh);
                bool deterministic = r1.Mesh.TriangleCount == r2.Mesh.TriangleCount && vol.Equals(vol2);
                double err = Math.Abs(vol - expected);
                int coincidentWarnings = r1.Warnings.Length == 0 ? 0 : Array.FindAll(r1.Warnings, w => w.Contains("coincident")).Length;
                bool passed = err < tol && deterministic &&
                              (expected == 0.0 ? r1.Mesh.TriangleCount == 0 : r1.IsClosed) &&
                              (!expectCoincidentWarning || coincidentWarnings == 1);
                report.Results.Add(new TestCaseResult
                {
                    Name = name, Backend = report.Backend, Passed = passed, ComputedValue = vol, ExpectedValue = expected,
                    Error = err, Tolerance = tol, IsClosed = r1.IsClosed, TriangleCount = r1.Mesh.TriangleCount,
                    DurationMs = sw.ElapsedMilliseconds,
                    Details = $"deterministic={deterministic} warnings=[{string.Join(" | ", r1.Warnings)}]"
                });
            });
        }

        /// <summary>
        /// Box [-1,1]^3 against a tessellated sphere. Expected values come from the input meshes: I = |box n sphere mesh|
        /// by exact convex clipping (Oracle.ClipConvexVolume), U = |A| + |B| - I, D = |A| - I, so the comparison
        /// measures algorithm error only. One analytic sanity check compares that oracle with the smooth sphere-box
        /// volume (sphere minus spherical caps) within the stated tessellation tolerance.
        /// </summary>
        private static void SphereCases(GpuSession s, SuiteReport report, string name, V3 centre, double r)
        {
            var box = MeshFactory.CreateBox(new V3(-1, -1, -1), new V3(1, 1, 1));
            var sphere = MeshFactory.CreateSphere(centre, r, 32, 33);
            double vBox = MeshFactory.SignedVolume(box);
            double vSphere = MeshFactory.SignedVolume(sphere);
            double expInt = Oracle.ClipConvexVolume(sphere, Oracle.BoxPlanes(new V3(-1, -1, -1), new V3(1, 1, 1)));
            if (!(expInt >= 0 && expInt <= Math.Min(vBox, vSphere) + ExactTol))
                throw new InvalidOperationException($"{name}: clipped oracle volume {expInt} is outside [0, min(|A|, |B|)].");

            Guard(report, $"{name}_AnalyticSanity", () =>
            {
                double smooth = Oracle.SmoothBoxSphereIntersection(1.0, centre, r);
                double rel = Math.Abs(expInt - smooth) / smooth;
                report.Results.Add(new TestCaseResult
                {
                    Name = $"{name}_AnalyticSanity", Backend = report.Backend, Passed = rel < SphereTessRelTol,
                    ComputedValue = expInt, ExpectedValue = smooth, Error = rel, Tolerance = SphereTessRelTol, IsClosed = true,
                    TriangleCount = sphere.TriangleCount,
                    Details = $"tessellated oracle vs smooth sphere-minus-caps; full-sphere tessellation deficit {(1 - vSphere / (4.0 / 3.0 * Math.PI * r * r * r)):E2}"
                });
            });

            var vols = new Dictionary<BooleanOperation, double>();
            foreach (var (op, expected) in new[] { (BooleanOperation.Union, vBox + vSphere - expInt), (BooleanOperation.Difference, vBox - expInt), (BooleanOperation.Intersection, expInt) })
            {
                string caseName = $"{name}_{op}";
                Guard(report, caseName, () =>
                {
                    var sw = Stopwatch.StartNew();
                    var r1 = MeshBoolean.Compute(s, new[] { box }, new[] { sphere }, op, BooleanTolerance);
                    sw.Stop();
                    var r2 = MeshBoolean.Compute(s, new[] { box }, new[] { sphere }, op, BooleanTolerance);
                    double vol = MeshFactory.SignedVolume(r1.Mesh);
                    vols[op] = vol;
                    bool deterministic = r1.Mesh.TriangleCount == r2.Mesh.TriangleCount && vol.Equals(MeshFactory.SignedVolume(r2.Mesh));
                    double err = Math.Abs(vol - expected);
                    report.Results.Add(new TestCaseResult
                    {
                        Name = caseName, Backend = report.Backend, Passed = err < SphereMeshTol && r1.IsClosed && deterministic,
                        ComputedValue = vol, ExpectedValue = expected, Error = err, Tolerance = SphereMeshTol, IsClosed = r1.IsClosed,
                        TriangleCount = r1.Mesh.TriangleCount, DurationMs = sw.ElapsedMilliseconds,
                        Details = $"abs error vs tessellated oracle; deterministic={deterministic} warnings=[{string.Join(" | ", r1.Warnings)}]"
                    });
                });
            }

            if (vols.Count == 3)
            {
                double idErr = Math.Max(Math.Abs(vols[BooleanOperation.Union] + vols[BooleanOperation.Intersection] - (vBox + vSphere)),
                                        Math.Abs(vols[BooleanOperation.Difference] - (vBox - vols[BooleanOperation.Intersection])));
                report.Results.Add(new TestCaseResult
                {
                    Name = $"{name}_VolumeIdentities", Backend = report.Backend, Passed = idErr < ExactTol,
                    ComputedValue = idErr, ExpectedValue = 0, Error = idErr, Tolerance = ExactTol, IsClosed = true,
                    Details = "max(|U + I - |A| - |B||, |D - (|A| - I)|)"
                });
            }
        }

        private static void TestWindingBox(GpuSession s, SuiteReport report)
        {
            var box = MeshFactory.CreateBox(new V3(-1, -1, -1), new V3(1, 1, 1));
            var rng = new Random(7);
            var q = new List<V3> { new V3(0, 0, 0), new V3(0.999, 0.2, -0.3), new V3(1.001, 0.2, -0.3), new V3(2, 0, 0) };
            for (int i = 0; i < 200; i++) q.Add(new V3(rng.NextDouble() * 4 - 2, rng.NextDouble() * 4 - 2, rng.NextDouble() * 4 - 2));
            CompareWinding(s, report, "WindingNumbers_Box_Brute", box, q.ToArray(), 1e-9);
        }

        private static void TestWindingSphereBvh(GpuSession s, SuiteReport report)
        {
            var sphere = MeshFactory.CreateSphere(new V3(0.1, -0.2, 0.05), 1.0, 32, 33);
            var rng = new Random(11);
            var q = new List<V3>();
            for (int i = 0; i < 300; i++) q.Add(new V3(rng.NextDouble() * 3 - 1.5, rng.NextDouble() * 3 - 1.5, rng.NextDouble() * 3 - 1.5));
            // Near-surface queries: face centroids offset by +/-1e-3 along the face normal.
            for (int i = 0; i < 300; i++)
            {
                int t = rng.Next(sphere.TriangleCount);
                V3 a = sphere.Vertices[sphere.Triangles[3 * t]], b = sphere.Vertices[sphere.Triangles[3 * t + 1]], c = sphere.Vertices[sphere.Triangles[3 * t + 2]];
                V3 n = MeshFactory.Cross(b - a, c - a);
                if (n.Length == 0) continue;
                V3 cen = (a + b + c) * (1.0 / 3.0);
                q.Add(cen + n * ((i % 2 == 0 ? 1e-3 : -1e-3) / n.Length));
            }
            CompareWinding(s, report, "WindingNumbers_Sphere2048_BVH", sphere, q.ToArray(), WindingBvhTol);
        }

        /// <summary>Compares GPU winding numbers to an independent brute solid-angle sum and ray-parity inside test.</summary>
        private static void CompareWinding(GpuSession s, SuiteReport report, string name, TriMesh mesh, V3[] queries, double tol)
        {
            var sw = Stopwatch.StartNew();
            double[] w;
            using (var gm = GpuTriMesh.Upload(s, mesh, ElementType.V3D))
            using (var pts = s.UploadPoints(queries, ElementType.V3D, Residency.Device))
            using (var wb = GpuGeometry.WindingNumbers(s, pts, gm))
                w = s.DownloadScalars(wb);
            sw.Stop();

            double maxErr = 0;
            int parityMismatch = 0;
            for (int i = 0; i < queries.Length; i++)
            {
                maxErr = Math.Max(maxErr, Math.Abs(w[i] - Oracle.SolidAngleWinding(mesh, queries[i])));
                if ((Math.Abs(w[i]) >= 0.5) != Oracle.RayParityInside(mesh, queries[i])) parityMismatch++;
            }
            report.Results.Add(new TestCaseResult
            {
                Name = name, Backend = report.Backend, Passed = maxErr < tol && parityMismatch == 0,
                ComputedValue = maxErr, ExpectedValue = 0, Error = maxErr, Tolerance = tol, IsClosed = true,
                TriangleCount = mesh.TriangleCount, DurationMs = sw.ElapsedMilliseconds,
                Details = $"{queries.Length} queries, max |W - brute| = {maxErr:E2}, ray-parity mismatches = {parityMismatch}"
            });
        }

        private static void TestBroadphase(GpuSession s, SuiteReport report)
        {
            var rng = new Random(3);
            V3[] RandomBoxes(int n, out V3[] max)
            {
                var min = new V3[n];
                max = new V3[n];
                for (int i = 0; i < n; i++)
                {
                    var p = new V3(rng.NextDouble() * 10, rng.NextDouble() * 10, rng.NextDouble() * 10);
                    min[i] = p;
                    max[i] = p + new V3(rng.NextDouble(), rng.NextDouble(), rng.NextDouble());
                }
                return min;
            }
            var minA = RandomBoxes(400, out var maxA);
            var minB = RandomBoxes(300, out var maxB);
            const double tol = 1e-3;
            var expected = new List<int>();
            for (int i = 0; i < minA.Length; i++)
                for (int j = 0; j < minB.Length; j++)
                    if (minA[i].X <= maxB[j].X + tol && maxA[i].X >= minB[j].X - tol &&
                        minA[i].Y <= maxB[j].Y + tol && maxA[i].Y >= minB[j].Y - tol &&
                        minA[i].Z <= maxB[j].Z + tol && maxA[i].Z >= minB[j].Z - tol) { expected.Add(i); expected.Add(j); }

            var sw = Stopwatch.StartNew();
            int[] pairs;
            using (var a0 = s.UploadPoints(minA, ElementType.V3D, Residency.Device))
            using (var a1 = s.UploadPoints(maxA, ElementType.V3D, Residency.Device))
            using (var b0 = s.UploadPoints(minB, ElementType.V3D, Residency.Device))
            using (var b1 = s.UploadPoints(maxB, ElementType.V3D, Residency.Device))
            using (var pb = GpuGeometry.OverlappingBoxPairs(s, a0, a1, b0, b1, tol))
                pairs = s.DownloadIndices(pb);
            sw.Stop();

            bool same = pairs.Length == expected.Count;
            for (int i = 0; same && i < pairs.Length; i++) same = pairs[i] == expected[i];
            report.Results.Add(new TestCaseResult
            {
                Name = "Broadphase_RandomBoxes", Backend = report.Backend, Passed = same && expected.Count > 0,
                ComputedValue = pairs.Length / 2, ExpectedValue = expected.Count / 2, IsClosed = true, DurationMs = sw.ElapsedMilliseconds,
                Details = "exact pair list vs host brute oracle"
            });
        }

        private static void TestEmpty(GpuSession s, SuiteReport report)
        {
            var box = MeshFactory.CreateBox(new V3(-1, -1, -1), new V3(1, 1, 1));
            var empty = new TriMesh();
            var r1 = MeshBoolean.Compute(s, new[] { empty }, new[] { box }, BooleanOperation.Union, BooleanTolerance);
            var r2 = MeshBoolean.Compute(s, new[] { box }, new TriMesh[] { null }, BooleanOperation.Difference, BooleanTolerance);
            var r3 = MeshBoolean.Compute(s, new[] { empty }, new[] { empty }, BooleanOperation.Intersection, BooleanTolerance);
            var r4 = MeshBoolean.Compute(s, new[] { box }, Array.Empty<TriMesh>(), BooleanOperation.Intersection, BooleanTolerance);
            double v1 = MeshFactory.SignedVolume(r1.Mesh), v2 = MeshFactory.SignedVolume(r2.Mesh);
            bool passed = Math.Abs(v1 - 8) < ExactTol && r1.IsClosed && Math.Abs(v2 - 8) < ExactTol && r2.IsClosed &&
                          r3.Mesh.TriangleCount == 0 && r4.Mesh.TriangleCount == 0 &&
                          Array.Exists(r2.Warnings, w => w.Contains("null")) && Array.Exists(r1.Warnings, w => w.Contains("no triangles"));
            report.Results.Add(new TestCaseResult
            {
                Name = "EmptyAndNullInputs", Backend = report.Backend, Passed = passed, ComputedValue = v1, ExpectedValue = 8,
                Error = Math.Abs(v1 - 8), Tolerance = ExactTol, IsClosed = r1.IsClosed, TriangleCount = r1.Mesh.TriangleCount,
                Details = $"empty u box={v1}, box - null={v2}, empty n empty tris={r3.Mesh.TriangleCount}, box n [] tris={r4.Mesh.TriangleCount}"
            });
        }

        private static void PrintReport(SuiteReport report)
        {
            Console.WriteLine($"Backend: {report.Backend} | Hardware: {report.IsHardwareAccelerated} | Device: {report.DeviceName} | " +
                              $"Passed {report.PassedTests}/{report.TotalTests} | kernel dispatches {report.KernelDispatches}, host dispatches {report.HostDispatches}");
            if (report.Error != null) Console.WriteLine(report.Error);
            Console.WriteLine($"{"",-6} {"case",-40} {"value",18} {"expected",18} {"error",10} {"closed",6} {"tris",6} {"ms",6}");
            foreach (var r in report.Results)
            {
                Console.WriteLine($"{(r.Passed ? "[PASS]" : "[FAIL]"),-6} {r.Name,-40} {r.ComputedValue,18:R} {r.ExpectedValue,18:R} {r.Error,10:E2} {r.IsClosed,6} {r.TriangleCount,6} {r.DurationMs,6}");
                if (!r.Passed) Console.WriteLine("       " + r.Details);
            }
        }
    }

    /// <summary>Independent oracles, written separately from the library kernels.</summary>
    public static class Oracle
    {
        public static double SolidAngleWinding(TriMesh m, V3 p)
        {
            double total = 0;
            for (int t = 0; t < m.TriangleCount; t++)
            {
                V3 a = m.Vertices[m.Triangles[3 * t]] - p, b = m.Vertices[m.Triangles[3 * t + 1]] - p, c = m.Vertices[m.Triangles[3 * t + 2]] - p;
                double la = a.Length, lb = b.Length, lc = c.Length;
                if (la == 0 || lb == 0 || lc == 0) continue;
                double triple = V3.Dot(a, MeshFactory.Cross(b, c));
                double denom = la * lb * lc + V3.Dot(a, b) * lc + V3.Dot(b, c) * la + V3.Dot(c, a) * lb;
                total += 2.0 * Math.Atan2(triple, denom);
            }
            return total / (4.0 * Math.PI);
        }

        /// <summary>Largest distance of any vertex in front of any face plane (0 for a convex closed mesh).</summary>
        public static double ConvexityViolation(TriMesh m)
        {
            double worst = 0;
            for (int t = 0; t < m.TriangleCount; t++)
            {
                V3 a = m.Vertices[m.Triangles[3 * t]], b = m.Vertices[m.Triangles[3 * t + 1]], c = m.Vertices[m.Triangles[3 * t + 2]];
                V3 n = MeshFactory.Cross(b - a, c - a);
                double len = n.Length;
                if (len == 0) continue;
                foreach (var v in m.Vertices) worst = Math.Max(worst, V3.Dot(v - a, n) / len);
            }
            return worst;
        }

        /// <summary>Half-spaces n.x &lt;= d of an axis-aligned box.</summary>
        public static (V3 N, double D)[] BoxPlanes(V3 min, V3 max) => new[]
        {
            (new V3(1, 0, 0), max.X), (new V3(-1, 0, 0), -min.X), (new V3(0, 1, 0), max.Y),
            (new V3(0, -1, 0), -min.Y), (new V3(0, 0, 1), max.Z), (new V3(0, 0, -1), -min.Z)
        };

        /// <summary>
        /// Exact volume of (closed convex mesh) n (half-spaces): each plane clips every face polygon (Sutherland-Hodgman)
        /// and closes the cut with the convex cap polygon of the cut points; volume by the divergence theorem.
        /// </summary>
        public static double ClipConvexVolume(TriMesh m, (V3 N, double D)[] planes)
        {
            var faces = new List<List<V3>>();
            for (int t = 0; t < m.TriangleCount; t++)
                faces.Add(new List<V3> { m.Vertices[m.Triangles[3 * t]], m.Vertices[m.Triangles[3 * t + 1]], m.Vertices[m.Triangles[3 * t + 2]] });

            foreach (var (n, d) in planes)
            {
                var next = new List<List<V3>>();
                var cut = new List<V3>();
                foreach (var f in faces)
                {
                    var o = new List<V3>();
                    for (int i = 0; i < f.Count; i++)
                    {
                        V3 p = f[i], q = f[(i + 1) % f.Count];
                        double sp = V3.Dot(n, p) - d, sq = V3.Dot(n, q) - d;
                        if (sp <= 0) o.Add(p);
                        if (sp == 0) cut.Add(p);
                        if ((sp < 0 && sq > 0) || (sp > 0 && sq < 0))
                        {
                            V3 x = p + (q - p) * (sp / (sp - sq));
                            o.Add(x);
                            cut.Add(x);
                        }
                    }
                    if (o.Count >= 3) next.Add(o);
                }
                var cap = ConvexCap(cut, n);
                if (cap.Count >= 3) next.Add(cap);
                faces = next;
            }

            double vol = 0;
            foreach (var f in faces)
                for (int i = 1; i + 1 < f.Count; i++)
                    vol += V3.Dot(f[0], MeshFactory.Cross(f[i], f[i + 1]));
            return vol / 6.0;
        }

        // Cut points of a convex solid on one plane, de-duplicated and ordered CCW about the outward normal n.
        private static List<V3> ConvexCap(List<V3> pts, V3 n)
        {
            var uniq = new List<V3>();
            foreach (var p in pts)
                if (!uniq.Exists(u => (u - p).Length < 1e-12)) uniq.Add(p);
            if (uniq.Count < 3) return uniq;
            V3 c = default;
            foreach (var p in uniq) c = c + p;
            c = c * (1.0 / uniq.Count);
            V3 u0 = Math.Abs(n.X) < 0.9 ? new V3(1, 0, 0) : new V3(0, 1, 0);
            V3 ux = MeshFactory.Cross(u0, n);
            ux = ux * (1.0 / ux.Length);
            V3 uy = MeshFactory.Cross(n, ux);
            uniq.Sort((a, b) => Math.Atan2(V3.Dot(a - c, uy), V3.Dot(a - c, ux)).CompareTo(Math.Atan2(V3.Dot(b - c, uy), V3.Dot(b - c, ux))));
            return uniq;
        }

        /// <summary>
        /// Smooth volume of the cube [-h,h]^3 n sphere(c, r) = sphere - sum of the spherical caps beyond each face plane,
        /// cap height = r - (distance from c to the plane). Throws unless the formula applies: c inside the cube and every
        /// cap's base disk inside its face (then caps are disjoint and lie outside the cube).
        /// </summary>
        public static double SmoothBoxSphereIntersection(double h, V3 c, double r)
        {
            double[] cc = { c.X, c.Y, c.Z };
            foreach (double x in cc)
                if (Math.Abs(x) >= h) throw new InvalidOperationException("Sphere centre must lie inside the box.");
            double vol = 4.0 / 3.0 * Math.PI * r * r * r;
            for (int axis = 0; axis < 3; axis++)
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    double dist = h - sign * cc[axis];
                    double capH = r - dist;
                    if (capH <= 0) continue;
                    double baseR = Math.Sqrt(r * r - dist * dist);
                    for (int other = 0; other < 3; other++)
                        if (other != axis && h - Math.Abs(cc[other]) < baseR)
                            throw new InvalidOperationException("Spherical caps reach a box edge; cap formula does not apply.");
                    vol -= Math.PI * capH * capH * (3 * r - capH) / 3.0;
                }
            if (!(vol > 0)) throw new InvalidOperationException($"Non-positive analytic intersection volume {vol}.");
            return vol;
        }

        /// <summary>Ray-parity inside test (Moller-Trumbore) along a fixed generic direction.</summary>
        public static bool RayParityInside(TriMesh m, V3 p)
        {
            var dir = new V3(0.5773502691896258, 0.5345224838248488, 0.6172133998483676);
            int hits = 0;
            for (int t = 0; t < m.TriangleCount; t++)
            {
                V3 a = m.Vertices[m.Triangles[3 * t]], b = m.Vertices[m.Triangles[3 * t + 1]], c = m.Vertices[m.Triangles[3 * t + 2]];
                V3 e1 = b - a, e2 = c - a;
                V3 h = MeshFactory.Cross(dir, e2);
                double det = V3.Dot(e1, h);
                if (Math.Abs(det) < 1e-300) continue;
                double inv = 1.0 / det;
                V3 sv = p - a;
                double u = V3.Dot(sv, h) * inv;
                if (u < 0 || u > 1) continue;
                V3 qv = MeshFactory.Cross(sv, e1);
                double v = V3.Dot(dir, qv) * inv;
                if (v < 0 || u + v > 1) continue;
                if (V3.Dot(e2, qv) * inv > 0) hits++;
            }
            return (hits & 1) == 1;
        }
    }

    public static class MeshFactory
    {
        public static TriMesh CreateBox(V3 min, V3 max)
        {
            V3[] v =
            {
                new V3(min.X, min.Y, min.Z), new V3(max.X, min.Y, min.Z), new V3(max.X, max.Y, min.Z), new V3(min.X, max.Y, min.Z),
                new V3(min.X, min.Y, max.Z), new V3(max.X, min.Y, max.Z), new V3(max.X, max.Y, max.Z), new V3(min.X, max.Y, max.Z)
            };
            int[] t =
            {
                0, 2, 1, 0, 3, 2,  4, 5, 6, 4, 6, 7,  0, 1, 5, 0, 5, 4,
                1, 2, 6, 1, 6, 5,  2, 3, 7, 2, 7, 6,  3, 0, 4, 3, 4, 7
            };
            return new TriMesh(v, t);
        }

        /// <summary>Closed, outward UV sphere with shared seam and pole vertices: 2*lon*(lat-1) triangles.</summary>
        public static TriMesh CreateSphere(V3 c, double r, int lon, int lat)
        {
            var verts = new List<V3> { new V3(c.X, c.Y, c.Z - r) };
            for (int i = 1; i < lat; i++)
            {
                double phi = Math.PI * (-0.5 + (double)i / lat);
                for (int j = 0; j < lon; j++)
                {
                    double th = 2 * Math.PI * j / lon;
                    verts.Add(new V3(c.X + r * Math.Cos(phi) * Math.Cos(th), c.Y + r * Math.Cos(phi) * Math.Sin(th), c.Z + r * Math.Sin(phi)));
                }
            }
            verts.Add(new V3(c.X, c.Y, c.Z + r));
            int top = verts.Count - 1;
            int Ring(int i, int j) => 1 + (i - 1) * lon + (j % lon);

            var tris = new List<int>();
            for (int j = 0; j < lon; j++) tris.AddRange(new[] { 0, Ring(1, j + 1), Ring(1, j) });
            for (int i = 1; i < lat - 1; i++)
                for (int j = 0; j < lon; j++)
                {
                    tris.AddRange(new[] { Ring(i, j), Ring(i, j + 1), Ring(i + 1, j + 1) });
                    tris.AddRange(new[] { Ring(i, j), Ring(i + 1, j + 1), Ring(i + 1, j) });
                }
            for (int j = 0; j < lon; j++) tris.AddRange(new[] { top, Ring(lat - 1, j), Ring(lat - 1, j + 1) });
            return new TriMesh(verts.ToArray(), tris.ToArray());
        }

        /// <summary>Closed outward box whose faces are nx x ny x nz grids of split quads (welded, like a meshed Brep box).</summary>
        public static TriMesh CreateGridBox(V3 min, V3 max, int nx, int ny, int nz)
        {
            var verts = new List<V3>();
            var index = new Dictionary<(double, double, double), int>();
            var tris = new List<int>();
            int Vert(double x, double y, double z)
            {
                if (!index.TryGetValue((x, y, z), out int id)) { id = verts.Count; verts.Add(new V3(x, y, z)); index[(x, y, z)] = id; }
                return id;
            }
            double L(double a, double b, int i, int n) => i == n ? b : a + (b - a) * i / n;
            // face(axis, side): grid over the two other axes, wound so the normal points outward.
            void Face(int axis, bool high)
            {
                int u = (axis + 1) % 3, w = (axis + 2) % 3;
                double[] lo = { min.X, min.Y, min.Z }, hi = { max.X, max.Y, max.Z };
                int[] cnt = { nx, ny, nz };
                for (int i = 0; i < cnt[u]; i++)
                    for (int j = 0; j < cnt[w]; j++)
                    {
                        int P(int di, int dj)
                        {
                            var c = new double[3];
                            c[axis] = high ? hi[axis] : lo[axis];
                            c[u] = L(lo[u], hi[u], i + di, cnt[u]);
                            c[w] = L(lo[w], hi[w], j + dj, cnt[w]);
                            return Vert(c[0], c[1], c[2]);
                        }
                        int a = P(0, 0), b = P(1, 0), cc = P(1, 1), d = P(0, 1);
                        if (high) tris.AddRange(new[] { a, b, cc, a, cc, d });
                        else tris.AddRange(new[] { a, cc, b, a, d, cc });
                    }
            }
            for (int axis = 0; axis < 3; axis++) { Face(axis, false); Face(axis, true); }
            return new TriMesh(verts.ToArray(), tris.ToArray());
        }

        /// <summary>Closed outward capped cylinder along +Z: seg side quads split by a diagonal, fan caps.</summary>
        public static TriMesh CreateCylinder(V3 baseCentre, double r, double h, int seg)
        {
            var verts = new List<V3> { baseCentre, new V3(baseCentre.X, baseCentre.Y, baseCentre.Z + h) };
            for (int k = 0; k < seg; k++)
            {
                double th = 2 * Math.PI * k / seg;
                double x = baseCentre.X + r * Math.Cos(th), y = baseCentre.Y + r * Math.Sin(th);
                verts.Add(new V3(x, y, baseCentre.Z));
                verts.Add(new V3(x, y, baseCentre.Z + h));
            }
            var tris = new List<int>();
            for (int k = 0; k < seg; k++)
            {
                int b0 = 2 + 2 * k, t0 = b0 + 1, b1 = 2 + 2 * ((k + 1) % seg), t1 = b1 + 1;
                tris.AddRange(new[] { b0, b1, t1, b0, t1, t0 });
                tris.AddRange(new[] { 0, b1, b0 });
                tris.AddRange(new[] { 1, t0, t1 });
            }
            return new TriMesh(verts.ToArray(), tris.ToArray());
        }

        /// <summary>Signed volume by the divergence theorem; negative for inside-out meshes.</summary>
        public static double SignedVolume(TriMesh m)
        {
            if (m?.Triangles == null) return 0;
            double sum = 0;
            for (int t = 0; t < m.TriangleCount; t++)
                sum += V3.Dot(m.Vertices[m.Triangles[3 * t]], Cross(m.Vertices[m.Triangles[3 * t + 1]], m.Vertices[m.Triangles[3 * t + 2]]));
            return sum / 6.0;
        }

        public static V3 Cross(V3 a, V3 b) => new V3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    }
}
