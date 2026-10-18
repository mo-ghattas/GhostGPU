using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Booleans.Validation
{
    /// <summary>
    /// Dense and near-degenerate cases: the meshes exported from the live benchmark .gh, procedural slab-minus-grid
    /// rebuilds, a near-tangent sphere and coplanar overlapping subdivided boxes. Every case runs all three operations
    /// twice and requires: closed output, bit-identical repeat runs, native piece counts, no warnings other than the
    /// documented coincident-face note, volume vs an independent oracle (1e-9 relative) and the volume identities
    /// V(A) = V(A-B) + V(AnB), V(AuB) = V(A) + V(B) - V(AnB) (1e-9 relative).
    /// </summary>
    public static class DenseCases
    {
        private const double RelTol = 1e-9;

        public static void Run(GpuSession s, SuiteReport report)
        {
            foreach (string path in Directory.GetFiles(Fixture.Dir, "*.json"))
            {
                var f = Fixture.Load(path);
                SlabCase(s, report, "Fixture_" + f.Name, f.A, f.B, 0.01);
            }

            foreach (var (m, res) in new[] { (6, 24), (6, 32) })
            {
                var slab = MeshFactory.CreateGridBox(new V3(0, 0, -5), new V3(100, 100, 5), res, res, 4);
                SlabCase(s, report, $"Grid{m}_Res{res}_Cylinders", new[] { slab }, Grid(m, (x, y) => MeshFactory.CreateCylinder(new V3(x, y, -10), 30.0 / m, 20, res)), 0.01);
                SlabCase(s, report, $"Grid{m}_Res{res}_Domes", new[] { slab }, Grid(m, (x, y) => MeshFactory.CreateSphere(new V3(x, y, 3), 40.0 / m, res, res)), 0.01);
            }
            {
                // Benchmark M=4 / res 16 intersection (GPU 40177 vs native 38335 there): domes pierce both slab faces.
                var slab = MeshFactory.CreateGridBox(new V3(0, 0, -5), new V3(100, 100, 5), 16, 16, 4);
                SlabCase(s, report, "Grid4_Res16_Domes", new[] { slab }, Grid(4, (x, y) => MeshFactory.CreateSphere(new V3(x, y, 3), 10, 16, 16)), 0.01);
            }

            // Near-tangent: the sphere's south pole pierces the box top by 1e-7.
            {
                var box = MeshFactory.CreateGridBox(new V3(-1, -1, -1), new V3(1, 1, 1), 3, 3, 3);
                var sphere = MeshFactory.CreateSphere(new V3(0.123, -0.071, 1 + 0.5 - 1e-7), 0.5, 24, 17);
                var expInt = Oracle.ClipConvexVolume(sphere, Oracle.BoxPlanes(new V3(-1, -1, -1), new V3(1, 1, 1)));
                Case(s, report, "NearTangent_Pole1e-7", new[] { box }, new[] { sphere }, 1e-4, expInt, 1);
            }

            // Coplanar overlap: top and bottom faces coplanar; misaligned grids (crossing edges, vertices on edges) and
            // aligned grids (collinear overlapping edges, coincident vertices).
            {
                var a = MeshFactory.CreateGridBox(new V3(0, 0, 0), new V3(2, 2, 2), 7, 7, 3);
                var b = MeshFactory.CreateGridBox(new V3(1, 0.3, 0), new V3(3, 1.7, 2), 5, 4, 2);
                Case(s, report, "CoplanarOverlap_Misaligned", new[] { a }, new[] { b }, 1e-4, 2.8, 1, expectCoincident: true);
                var a2 = MeshFactory.CreateGridBox(new V3(0, 0, 0), new V3(2, 2, 2), 8, 8, 8);
                var b2 = MeshFactory.CreateGridBox(new V3(1, 0.5, 0), new V3(3, 1.5, 2), 8, 4, 8);
                Case(s, report, "CoplanarOverlap_Aligned", new[] { a2 }, new[] { b2 }, 1e-4, 2.0, 1, expectCoincident: true);
            }
        }

        private static TriMesh[] Grid(int m, Func<double, double, TriMesh> make)
        {
            var list = new List<TriMesh>();
            double cell = 100.0 / m;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    list.Add(make(cell / 2 + i * cell, cell / 2 + j * cell));
            return list.ToArray();
        }

        /// <summary>A = one axis-aligned box mesh, B = pairwise-disjoint convex meshes: exact oracle by convex clipping.</summary>
        private static void SlabCase(GpuSession s, SuiteReport report, string name, TriMesh[] a, TriMesh[] b, double tol)
        {
            V3 mn = a[0].Vertices[0], mx = mn;
            foreach (var v in a[0].Vertices) { mn = V3.Min(mn, v); mx = V3.Max(mx, v); }
            V3 size = mx - mn;
            if (a.Length != 1 || Math.Abs(MeshFactory.SignedVolume(a[0]) - size.X * size.Y * size.Z) > 1e-9 * size.X * size.Y * size.Z)
                throw new InvalidOperationException($"{name}: A is not a single axis-aligned box; the clipping oracle does not apply.");
            double inter = 0;
            int touching = 0;
            foreach (var m in b)
            {
                double v = Oracle.ClipConvexVolume(m, Oracle.BoxPlanes(mn, mx));
                inter += v;
                if (v > 0) touching++;
            }
            Case(s, report, name, a, b, tol, inter, touching);
        }

        /// <summary>Runs all three operations; expInt = oracle volume of A n B; intPieces = native piece count of A n B.</summary>
        private static void Case(GpuSession s, SuiteReport report, string name, TriMesh[] a, TriMesh[] b, double tol,
                                 double expInt, int intPieces, bool expectCoincident = false)
        {
            string only = Environment.GetEnvironmentVariable("GG_CASE");
            if (!string.IsNullOrEmpty(only) && !name.Contains(only)) return;
            double va = 0, vb = 0;
            foreach (var m in a) va += MeshFactory.SignedVolume(m);
            foreach (var m in b) vb += MeshFactory.SignedVolume(m); // B meshes are disjoint in every case
            var expected = new Dictionary<BooleanOperation, double>
            {
                [BooleanOperation.Union] = va + vb - expInt,
                [BooleanOperation.Difference] = va - expInt,
                [BooleanOperation.Intersection] = expInt
            };
            var pieces = new Dictionary<BooleanOperation, int>
            {
                [BooleanOperation.Union] = 1,
                [BooleanOperation.Difference] = 1,
                [BooleanOperation.Intersection] = intPieces
            };
            var vols = new Dictionary<BooleanOperation, double>();
            foreach (var op in new[] { BooleanOperation.Difference, BooleanOperation.Intersection, BooleanOperation.Union })
            {
                string caseName = $"{name}_{op}";
                Program.Guard(report, caseName, () =>
                {
                    var sw = Stopwatch.StartNew();
                    var r1 = MeshBoolean.Compute(s, a, b, op, tol);
                    sw.Stop();
                    string profile = MeshBoolean.LastProfile;
                    var r2 = MeshBoolean.Compute(s, a, b, op, tol);
                    double vol = MeshFactory.SignedVolume(r1.Mesh);
                    vols[op] = vol;
                    bool deterministic = SameMesh(r1.Mesh, r2.Mesh) && r1.Pieces.Length == r2.Pieces.Length;
                    var bad = Array.FindAll(r1.Warnings, w => !w.Contains("coincident"));
                    double err = Math.Abs(vol - expected[op]) / Math.Max(va, 1e-300);
                    bool piecesOk = r1.Pieces.Length == pieces[op];
                    bool piecesClosed = Array.TrueForAll(r1.Pieces, MeshBoolean.CheckClosedness);
                    bool passed = r1.IsClosed && piecesClosed && deterministic && bad.Length == 0 && err < RelTol && piecesOk;
                    report.Results.Add(new TestCaseResult
                    {
                        Name = caseName, Backend = report.Backend, Passed = passed, ComputedValue = vol, ExpectedValue = expected[op],
                        Error = err, Tolerance = RelTol, IsClosed = r1.IsClosed, TriangleCount = r1.Mesh.TriangleCount,
                        DurationMs = sw.ElapsedMilliseconds,
                        Details = $"rel error vs oracle; pieces={r1.Pieces.Length} (expected {pieces[op]}, all closed={piecesClosed}) " +
                                  $"deterministic={deterministic} profile=[{profile}] warnings=[{string.Join(" | ", r1.Warnings)}]" +
                                  (expectCoincident ? " (coincident faces expected)" : "")
                    });
                });
            }
            if (vols.Count == 3)
            {
                double id1 = Math.Abs(va - vols[BooleanOperation.Difference] - vols[BooleanOperation.Intersection]);
                double id2 = Math.Abs(vols[BooleanOperation.Union] - (va + vb - vols[BooleanOperation.Intersection]));
                double rel = Math.Max(id1, id2) / va;
                report.Results.Add(new TestCaseResult
                {
                    Name = $"{name}_VolumeIdentities", Backend = report.Backend, Passed = rel < RelTol, ComputedValue = rel,
                    ExpectedValue = 0, Error = rel, Tolerance = RelTol, IsClosed = true,
                    Details = "max(|V(A) - V(A-B) - V(AnB)|, |V(AuB) - V(A) - V(B) + V(AnB)|) / V(A)"
                });
            }
        }

        private static bool SameMesh(TriMesh x, TriMesh y)
        {
            if (x.Vertices.Length != y.Vertices.Length || x.Triangles.Length != y.Triangles.Length) return false;
            for (int i = 0; i < x.Vertices.Length; i++)
                if (x.Vertices[i].X != y.Vertices[i].X || x.Vertices[i].Y != y.Vertices[i].Y || x.Vertices[i].Z != y.Vertices[i].Z) return false;
            for (int i = 0; i < x.Triangles.Length; i++)
                if (x.Triangles[i] != y.Triangles[i]) return false;
            return true;
        }
    }
}
