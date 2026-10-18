using System;
using System.Collections.Generic;
using System.Diagnostics;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Booleans.Validation
{
    /// <summary>--diag fixture.json [cuda]: cold + warm timing with the stage profile, float-safety check and the exact oracle.</summary>
    public static class Diag
    {
        public static void Run(string path, GpuBackend backend, double tol = 0.01)
        {
            var f = Fixture.Load(path);
            Console.WriteLine($"{f.Name}: op={f.Op} |A|={f.A.Length} |B|={f.B.Length} native={f.NativeVolume}");
            using (var s = new GpuSession(new SessionOptions { Backend = backend, MemoryLimitBytes = 1L << 30 }))
            {
                var sw = Stopwatch.StartNew();
                var r = MeshBoolean.Compute(s, f.A, f.B, f.Op, tol);
                Console.WriteLine($"Compute {sw.ElapsedMilliseconds} ms, vol={MeshFactory.SignedVolume(r.Mesh):R} closed={r.IsClosed} tris={r.Mesh.TriangleCount}");
                Console.WriteLine("  profile: " + MeshBoolean.LastProfile);
                {
                    // Float (Rhino single-precision vertex) collisions and shortest edge of the result.
                    var seen = new Dictionary<(float, float, float), int>();
                    int collide = 0;
                    foreach (var v in r.Mesh.Vertices) { var k = ((float)v.X, (float)v.Y, (float)v.Z); if (seen.ContainsKey(k)) collide++; else seen[k] = 1; }
                    double minEdge = double.MaxValue;
                    for (int t = 0; t < r.Mesh.Triangles.Length; t += 3)
                        for (int e = 0; e < 3; e++) minEdge = Math.Min(minEdge, (r.Mesh.Vertices[r.Mesh.Triangles[t + e]] - r.Mesh.Vertices[r.Mesh.Triangles[t + (e + 1) % 3]]).Length);
                    // Rhino-like: float vertices, topology vertices welded by position, collapsed faces culled.
                    var fid = new Dictionary<(float, float, float), int>();
                    var map = new int[r.Mesh.Vertices.Length];
                    for (int i = 0; i < map.Length; i++)
                    {
                        var v = r.Mesh.Vertices[i]; var k = ((float)v.X, (float)v.Y, (float)v.Z);
                        if (!fid.TryGetValue(k, out int id)) fid[k] = id = fid.Count;
                        map[i] = id;
                    }
                    var ft = new List<int>();
                    for (int t = 0; t < r.Mesh.Triangles.Length; t += 3)
                    {
                        int a = map[r.Mesh.Triangles[t]], b = map[r.Mesh.Triangles[t + 1]], c = map[r.Mesh.Triangles[t + 2]];
                        if (a != b && b != c && c != a) { ft.Add(a); ft.Add(b); ft.Add(c); }
                    }
                    bool floatClosed = MeshBoolean.CheckClosedness(new TriMesh(new V3[fid.Count], ft.ToArray()));
                    Console.WriteLine($"  pieces={r.Pieces.Length} floatCollisions={collide} minEdge={minEdge:E2} closedAfterFloatWeld={floatClosed} culled={(r.Mesh.Triangles.Length - ft.Count) / 3}");
                }
                sw.Restart(); MeshBoolean.Compute(s, f.A, f.B, f.Op, tol);
                Console.WriteLine($"  warm run {sw.ElapsedMilliseconds} ms profile: " + MeshBoolean.LastProfile);
                foreach (var w in r.Warnings) Console.WriteLine("  warn: " + w);
            }

            OracleReport(f);
        }

        public static void OracleReport(Fixture f)
        {
            double va = 0; foreach (var m in f.A) va += MeshFactory.SignedVolume(m);
            double inter = 0, vb = 0, maxViol = 0;
            V3 mn = f.A[0].Vertices[0], mx = mn;
            foreach (var v in f.A[0].Vertices) { mn = V3.Min(mn, v); mx = V3.Max(mx, v); }
            foreach (var m in f.B)
            {
                vb += MeshFactory.SignedVolume(m);
                inter += Oracle.ClipConvexVolume(m, Oracle.BoxPlanes(mn, mx));
                maxViol = Math.Max(maxViol, Oracle.ConvexityViolation(m));
            }
            Console.WriteLine($"oracle: |A|={va:R} sum|B|={vb:R} A^B={inter:R} A-B={va - inter:R} AuB={va + vb - inter:R} maxConvexViolation={maxViol:E2}");
        }

    }
}
