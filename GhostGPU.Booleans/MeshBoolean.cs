using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Booleans
{
    public sealed class MeshBooleanResult
    {
        public TriMesh Mesh;
        public string[] Warnings;
        public bool IsClosed;
        /// <summary>Connected pieces of <see cref="Mesh"/> (native Mesh.CreateBoolean* returns one mesh per piece).</summary>
        public TriMesh[] Pieces = Array.Empty<TriMesh>();

        public MeshBooleanResult()
        {
            Mesh = new TriMesh();
            Warnings = Array.Empty<string>();
        }

        public MeshBooleanResult(TriMesh mesh, string[] warnings, bool isClosed)
        {
            Mesh = mesh ?? new TriMesh();
            Warnings = warnings ?? Array.Empty<string>();
            IsClosed = isClosed;
        }
    }

    public static class MeshBoolean
    {
        /// <summary>
        /// Solid mesh boolean with native GH list semantics: the A list is one solid (the union of its meshes),
        /// the B list another; Union = A u B, Difference = A - B, Intersection = A n B. Overlaps inside a list are
        /// resolved by classifying every face against every other operand (no concatenation shortcut).
        /// Pipeline (GPU = ILGPU kernel, host = counted host dispatch):
        /// 1. GPU broadphase of triangle AABBs per overlapping operand pair.
        /// 2. Host exact tri-tri narrowphase (Parallel.For): every intersection point gets a symbolic key and one global
        ///    vertex id (see Intersector), so triangles sharing an edge share its cut points.
        /// 3. Host per-triangle constrained re-triangulation with exact predicates (Parallel.For, see Arrangement).
        /// 4. Patches: sub-triangles connected across non-cut edges; one representative per patch.
        /// 5. Winding numbers (GPU kernel, host below HostWorkThreshold) of two samples per patch (centroid +/- delta*normal) against each operand.
        /// 6. Regularized keep/flip rule per patch (reduces to the rule table; also resolves coincident faces).
        /// 7. Compaction (vertices already shared), safety-net T-junction repair, closedness, connected pieces.
        /// </summary>
        internal static string LastProfile = "";
        public static MeshBooleanResult Compute(
            GpuSession s,
            IReadOnlyList<TriMesh> a,
            IReadOnlyList<TriMesh> b,
            BooleanOperation op,
            double tolerance)
        {
            if (s == null)
                throw new ArgumentNullException(nameof(s));

            var warnings = new List<string>();
            var operands = new List<TriMesh>();
            var sides = new List<int>(); // 0 = A, 1 = B
            Collect(a, 0, "A", operands, sides, warnings);
            Collect(b, 1, "B", operands, sides, warnings);
            if (operands.Count == 0)
            {
                warnings.Add("All input meshes are null or empty.");
                return new MeshBooleanResult(new TriMesh(), warnings.ToArray(), false);
            }

            int n = operands.Count;
            var bmin = new V3[n];
            var bmax = new V3[n];
            for (int i = 0; i < n; i++)
                GetBounds(operands[i], out bmin[i], out bmax[i]);
            V3 gmin = bmin[0], gmax = bmax[0];
            for (int i = 1; i < n; i++) { gmin = V3.Min(gmin, bmin[i]); gmax = V3.Max(gmax, bmax[i]); }
            double diag = (gmax - gmin).Length;
            if (!(diag > 0))
            {
                warnings.Add("All input vertices coincide; result is empty.");
                return new MeshBooleanResult(new TriMesh(), warnings.ToArray(), false);
            }

            // One geometric tolerance for snapping, de-duplication, welding and coincidence: the caller's tolerance,
            // bounded relative to the model size (1e-12 .. 1e-3 of the bounding-box diagonal).
            double tol = tolerance > 0 ? tolerance : 1e-9 * diag;
            tol = Math.Max(1e-12 * diag, Math.Min(1e-3 * diag, tol));

            var gpu = new GpuTriMesh[n];
            try
            {
                for (int i = 0; i < n; i++)
                    gpu[i] = GpuTriMesh.Upload(s, operands[i], ElementType.V3D);
                return Run(s, operands, sides, gpu, bmin, bmax, op, tol, diag, warnings);
            }
            finally
            {
                for (int i = 0; i < n; i++)
                    gpu[i]?.Dispose();
            }
        }

        private static void Collect(IReadOnlyList<TriMesh> list, int side, string name, List<TriMesh> operands, List<int> sides, List<string> warnings)
        {
            if (list == null)
                return;
            for (int i = 0; i < list.Count; i++)
            {
                var m = list[i];
                if (m == null)
                {
                    warnings.Add($"{name}[{i}] is null and was ignored.");
                    continue;
                }
                if (m.TriangleCount == 0 || m.Vertices == null || m.Vertices.Length == 0)
                {
                    warnings.Add($"{name}[{i}] has no triangles and was ignored.");
                    continue;
                }
                if (m.Triangles.Length % 3 != 0)
                    throw new ArgumentException($"{name}[{i}]: triangle index count is not a multiple of 3.");
                foreach (int idx in m.Triangles)
                    if ((uint)idx >= (uint)m.Vertices.Length)
                        throw new ArgumentException($"{name}[{i}]: vertex index {idx} out of range.");
                foreach (var v in m.Vertices)
                    if (!v.Finite)
                        throw new ArgumentException($"{name}[{i}]: non-finite vertex.");
                operands.Add(m);
                sides.Add(side);
            }
        }

        private static MeshBooleanResult Run(
            GpuSession s, List<TriMesh> operands, List<int> sides, GpuTriMesh[] gpu, V3[] bmin, V3[] bmax,
            BooleanOperation op, double tol, double diag, List<string> warnings)
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var prof = new System.Text.StringBuilder();
            void Mark(string stage) { prof.Append(stage).Append('=').Append(clock.ElapsedMilliseconds).Append("ms "); clock.Restart(); }

            int n = operands.Count;
            int[] triOffset = new int[n + 1];
            for (int i = 0; i < n; i++)
                triOffset[i + 1] = triOffset[i] + operands[i].TriangleCount;
            // Points closer than mergeEps become one vertex (before re-triangulation, so consistently in every triangle).
            // Kept at roundoff scale: a larger radius can collapse a whole tiny intersection loop (near-tangent contact).
            double mergeEps = Math.Min(tol, 1e-9 * diag);
            var ix = new Intersector(operands, triOffset, mergeEps);
            Mark("topology");

            // Stages 1-2: GPU broadphase per overlapping operand pair, host exact narrowphase (Parallel.For per pair list).
            for (int k = 0; k < n; k++)
            {
                for (int l = k + 1; l < n; l++)
                {
                    if (!BoxesOverlap(bmin[k], bmax[k], bmin[l], bmax[l], tol))
                        continue;
                    int[] pairs;
                    using (var pairsBuf = GpuGeometry.OverlappingBoxPairs(s, gpu[k].MinAABB, gpu[k].MaxAABB, gpu[l].MinAABB, gpu[l].MaxAABB, tol))
                        pairs = s.DownloadIndices(pairsBuf);
                    ix.AddPairs(pairs, triOffset[k], triOffset[l]);
                    s.CountHostDispatch();
                }
            }
            Mark("intersect");

            // Stage 3: shared triple points, global epsilon merge, per-triangle exact re-triangulation (Parallel.For).
            ix.SplitCrossings(diag);
            ix.Merge(mergeEps);
            ix.Triangulate();
            s.CountHostDispatch();
            if (ix.Degenerate > 0)
                warnings.Add($"{ix.Degenerate} near-degenerate configuration(s) were snapped during re-triangulation.");
            if (ix.DroppedTriangles > 0)
                warnings.Add($"{ix.DroppedTriangles} degenerate input triangle(s) carrying cut points were dropped.");
            Mark("triangulate");

            // Stage 4: patches = sub-triangles of one operand connected across non-cut edges. Every patch lies entirely
            // inside or outside every other operand, so one representative (its largest sub-triangle) classifies it.
            var pts = ix.Pts;
            var tri = ix.SubTris;
            var subOwner = ix.SubOwner;
            int subCount = subOwner.Count;
            var patchOf = new int[subCount];
            for (int i = 0; i < subCount; i++) patchOf[i] = i;
            int FindPatch(int i) { while (patchOf[i] != i) { patchOf[i] = patchOf[patchOf[i]]; i = patchOf[i]; } return i; }
            var firstOnEdge = new Dictionary<(long, int), int>(KeyComparer.Instance);
            for (int i = 0; i < subCount; i++)
            {
                for (int e = 0; e < 3; e++)
                {
                    long key = Intersector.UndirectedKey(tri[3 * i + e], tri[3 * i + (e + 1) % 3]);
                    if (ix.CutEdges.Contains(key)) continue;
                    if (!firstOnEdge.TryGetValue((key, subOwner[i]), out int other)) { firstOnEdge[(key, subOwner[i])] = i; continue; }
                    int ra = FindPatch(i), rb = FindPatch(other);
                    if (ra != rb) { if (ra < rb) patchOf[rb] = ra; else patchOf[ra] = rb; }
                }
            }
            var patchRep = new Dictionary<int, int>();
            var patchArea = new Dictionary<int, double>();
            for (int i = 0; i < subCount; i++)
            {
                int r = FindPatch(i);
                V3 p0 = pts[tri[3 * i]], p1 = pts[tri[3 * i + 1]], p2 = pts[tri[3 * i + 2]];
                double area = Cross(p1 - p0, p2 - p0).Length;
                if (!patchArea.TryGetValue(r, out double best) || area > best) { patchArea[r] = area; patchRep[r] = i; }
            }
            var patches = new List<int>(patchRep.Keys);
            patches.Sort();
            int patchCount = patches.Count;
            var repTri = new int[patchCount];
            var plus = new V3[patchCount];
            var minus = new V3[patchCount];
            var patchIndex = new Dictionary<int, int>();
            for (int q = 0; q < patchCount; q++)
            {
                patchIndex[patches[q]] = q;
                int i = repTri[q] = patchRep[patches[q]];
                V3 p0 = pts[tri[3 * i]], p1 = pts[tri[3 * i + 1]], p2 = pts[tri[3 * i + 2]];
                V3 c = new V3((p0.X + p1.X + p2.X) / 3.0, (p0.Y + p1.Y + p2.Y) / 3.0, (p0.Z + p1.Z + p2.Z) / 3.0);
                V3 nrm = Cross(p1 - p0, p2 - p0);
                double nLen = nrm.Length;
                double maxEdge = Math.Max(Math.Max((p1 - p0).Length, (p2 - p1).Length), (p0 - p2).Length);
                // Offset delta: at most the tolerance (surfaces closer than tol count as coincident), at most a tenth
                // of the representative's smallest altitude, and never below roundoff level.
                double delta = nLen > 0 ? Math.Max(1e-10 * diag, Math.Min(tol, 0.1 * nLen / maxEdge)) : 0;
                V3 off = nLen > 0 ? nrm * (delta / nLen) : default;
                plus[q] = c + off;
                minus[q] = c - off;
            }
            Mark("patches");

            // Stage 5: winding numbers of every patch sample against every other operand (bounding-box culled). One sample
            // pair per patch keeps this small, so below HostWorkThreshold it runs on the host (runtime convention).
            int hostThreshold = s.Options.HostWorkThreshold;
            var inPlus = new bool[n][];
            var inMinus = new bool[n][];
            int ambiguous = 0;
            for (int m = 0; m < n; m++)
            {
                inPlus[m] = new bool[patchCount];
                inMinus[m] = new bool[patchCount];
                var idx = new List<int>();
                var samples = new List<V3>();
                for (int q = 0; q < patchCount; q++)
                {
                    if (subOwner[repTri[q]] == m) continue;
                    if (!PointInBox(plus[q], bmin[m], bmax[m], tol) && !PointInBox(minus[q], bmin[m], bmax[m], tol)) continue;
                    idx.Add(q);
                    samples.Add(plus[q]);
                    samples.Add(minus[q]);
                }
                if (idx.Count == 0) continue;

                double[] w;
                if (samples.Count < hostThreshold)
                    w = GpuGeometry.HostWindingNumbers(s, samples.ToArray(), gpu[m]);
                else
                    using (var ptsBuf = s.UploadPoints(samples.ToArray(), ElementType.V3D, Residency.Device))
                    using (var wBuf = GpuGeometry.WindingNumbers(s, ptsBuf, gpu[m]))
                        w = s.DownloadScalars(wBuf);

                for (int q = 0; q < idx.Count; q++)
                {
                    double wp = Math.Abs(w[2 * q]), wm = Math.Abs(w[2 * q + 1]);
                    if (Math.Abs(wp - 0.5) < 0.25 || Math.Abs(wm - 0.5) < 0.25) ambiguous++;
                    inPlus[m][idx[q]] = wp >= 0.5;
                    inMinus[m][idx[q]] = wm >= 0.5;
                }
            }
            if (ambiguous > 0)
                warnings.Add($"{ambiguous} sample(s) had winding numbers near 0.5 (|W| in 0.25..0.75): an input mesh is probably open or self-intersecting.");
            Mark("winding");

            // Stage 6: regularized keep/flip per patch. R(x) = op(inside A-list(x), inside B-list(x)) evaluated just behind
            // (-) and just in front of (+) the patch. Keep if R(-) && !R(+); keep flipped if !R(-) && R(+).
            // For non-coincident faces this is exactly the rule table (Difference keeps B-inside-A flipped).
            // Coincident patches (an operand's inside differs across the face) are kept from the lowest-index operand only.
            var decision = new int[patchCount]; // 0 drop, 1 keep, -1 keep flipped
            int coincident = 0;
            for (int q = 0; q < patchCount; q++)
            {
                int own = subOwner[repTri[q]];
                bool aMinus = false, aPlus = false, bMinus = false, bPlus = false, coincidentLower = false, isCoincident = false;
                for (int m = 0; m < n; m++)
                {
                    bool pm = m != own && inPlus[m][q];
                    bool mm = m == own || inMinus[m][q];
                    if (m != own && pm != mm)
                    {
                        isCoincident = true;
                        if (m < own) coincidentLower = true;
                    }
                    if (sides[m] == 0) { aMinus |= mm; aPlus |= pm; }
                    else { bMinus |= mm; bPlus |= pm; }
                }
                if (isCoincident) coincident++;
                if (coincidentLower) continue;
                bool rMinus = Apply(op, aMinus, bMinus);
                bool rPlus = Apply(op, aPlus, bPlus);
                decision[q] = rMinus == rPlus ? 0 : rMinus ? 1 : -1;
            }
            if (coincident > 0)
                warnings.Add($"{coincident} patch(es) lie on coincident faces (within tolerance); resolved by the regularized rule.");

            var kept = new List<int>();
            for (int i = 0; i < subCount; i++)
            {
                int d = decision[patchIndex[FindPatch(i)]];
                if (d == 0) continue;
                kept.Add(tri[3 * i]);
                if (d > 0) { kept.Add(tri[3 * i + 1]); kept.Add(tri[3 * i + 2]); }
                else { kept.Add(tri[3 * i + 2]); kept.Add(tri[3 * i + 1]); }
            }
            Mark("classify");

            // Stage 7: compact the shared vertex table (no weld needed: neighbours already share ids). A T-junction repair
            // runs only as a safety net if some edge is still open; then split into connected pieces like native.
            var remap = new Dictionary<int, int>();
            var verts = new List<V3>();
            var tris = new List<int>(kept.Count);
            foreach (int id in kept)
            {
                if (!remap.TryGetValue(id, out int r)) { r = verts.Count; remap[id] = r; verts.Add(pts[id]); }
                tris.Add(r);
            }
            var result = new TriMesh(verts.ToArray(), tris.ToArray());
            bool closed = result.TriangleCount > 0 && CheckClosedness(result);
            if (result.TriangleCount > 0 && !closed)
            {
                tris = RepairTJunctions(verts, tris, tol);
                result = new TriMesh(verts.ToArray(), tris.ToArray());
                closed = CheckClosedness(result);
            }
            s.CountHostDispatch();
            if (result.TriangleCount > 0 && !closed)
                warnings.Add("Result mesh is not closed.");
            var pieces = SplitPieces(result);
            Mark("output");
            LastProfile = prof.ToString();
            return new MeshBooleanResult(result, warnings.ToArray(), closed) { Pieces = pieces };
        }

        /// <summary>Connected components over shared edges, each compacted, ordered by first triangle (native SplitDisjointPieces).</summary>
        private static TriMesh[] SplitPieces(TriMesh mesh)
        {
            int tc = mesh.TriangleCount;
            if (tc == 0) return Array.Empty<TriMesh>();
            var parent = new int[tc];
            for (int i = 0; i < tc; i++) parent[i] = i;
            int Find(int i) { while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; }
            var firstOnEdge = new Dictionary<long, int>(KeyComparer.Instance);
            for (int t = 0; t < tc; t++)
                for (int e = 0; e < 3; e++)
                {
                    long key = Intersector.UndirectedKey(mesh.Triangles[3 * t + e], mesh.Triangles[3 * t + (e + 1) % 3]);
                    if (!firstOnEdge.TryGetValue(key, out int o)) { firstOnEdge[key] = t; continue; }
                    int ra = Find(t), rb = Find(o);
                    if (ra != rb) { if (ra < rb) parent[rb] = ra; else parent[ra] = rb; }
                }
            var pieceOf = new Dictionary<int, int>();
            var pv = new List<List<V3>>();
            var pt = new List<List<int>>();
            var vmap = new List<Dictionary<int, int>>();
            for (int t = 0; t < tc; t++)
            {
                int r = Find(t);
                if (!pieceOf.TryGetValue(r, out int p))
                {
                    p = pv.Count;
                    pieceOf[r] = p;
                    pv.Add(new List<V3>()); pt.Add(new List<int>()); vmap.Add(new Dictionary<int, int>());
                }
                for (int c = 0; c < 3; c++)
                {
                    int v = mesh.Triangles[3 * t + c];
                    if (!vmap[p].TryGetValue(v, out int lv)) { lv = pv[p].Count; vmap[p][v] = lv; pv[p].Add(mesh.Vertices[v]); }
                    pt[p].Add(lv);
                }
            }
            var pieces = new TriMesh[pv.Count];
            for (int p = 0; p < pieces.Length; p++) pieces[p] = new TriMesh(pv[p].ToArray(), pt[p].ToArray());
            return pieces;
        }
        private static bool Apply(BooleanOperation op, bool inA, bool inB)
        {
            switch (op)
            {
                case BooleanOperation.Union: return inA || inB;
                case BooleanOperation.Intersection: return inA && inB;
                case BooleanOperation.Difference: return inA && !inB;
                default: throw new ArgumentOutOfRangeException(nameof(op), op, null);
            }
        }

        private static bool BoxesOverlap(V3 minA, V3 maxA, V3 minB, V3 maxB, double tol) =>
            minA.X <= maxB.X + tol && maxA.X >= minB.X - tol &&
            minA.Y <= maxB.Y + tol && maxA.Y >= minB.Y - tol &&
            minA.Z <= maxB.Z + tol && maxA.Z >= minB.Z - tol;

        private static bool PointInBox(V3 p, V3 min, V3 max, double tol) => BoxesOverlap(p, p, min, max, tol);

        /// <summary>
        /// Worklist T-junction repair. Only open edges (a directed edge u->v not matched by v->u) can be T-junctions, and
        /// only vertices of open edges can be the T: such an edge is split at the nearest open vertex lying on it (within
        /// tol). Children inherit the open flag on the two halves of the split edge; the new interior edge is closed, so
        /// matched edges are never split (a nearly-flat legitimate vertex close to a matched edge is left alone) and the
        /// pass terminates: every split strictly shortens a piece of an originally open edge.
        /// </summary>
        private static List<int> RepairTJunctions(List<V3> verts, List<int> tris, double tol)
        {
            var edgeCounts = new Dictionary<long, int>(KeyComparer.Instance);
            for (int t = 0; t < tris.Count; t += 3)
                for (int e = 0; e < 3; e++)
                {
                    long k = EdgeKey(tris[t + e], tris[t + (e + 1) % 3]);
                    edgeCounts.TryGetValue(k, out int c);
                    edgeCounts[k] = c + 1;
                }
            bool IsOpen(int u, int v)
            {
                edgeCounts.TryGetValue(EdgeKey(u, v), out int c);
                edgeCounts.TryGetValue(EdgeKey(v, u), out int o);
                return c != o;
            }

            var openVerts = new List<int>();
            var isOpenVert = new bool[verts.Count];
            foreach (var kvp in edgeCounts)
            {
                int u = (int)(kvp.Key >> 32), v = (int)(kvp.Key & 0xFFFFFFFF);
                if (!IsOpen(u, v)) continue;
                if (!isOpenVert[u]) { isOpenVert[u] = true; openVerts.Add(u); }
                if (!isOpenVert[v]) { isOpenVert[v] = true; openVerts.Add(v); }
            }
            if (openVerts.Count == 0)
                return tris;
            openVerts.Sort();

            // Open vertices sorted along each axis; an edge searches the slab of its thinnest axis.
            var sorted = new int[3][];
            var keys = new double[3][];
            for (int axis = 0; axis < 3; axis++)
            {
                sorted[axis] = openVerts.ToArray();
                keys[axis] = new double[openVerts.Count];
                for (int i = 0; i < openVerts.Count; i++) keys[axis][i] = Coord(verts[openVerts[i]], axis);
                Array.Sort(keys[axis], sorted[axis]);
            }

            var output = new List<int>(tris.Count);
            var work = new Stack<(int, int, int, int)>(); // corners + open-edge mask (bit e = edge corner e -> e+1)
            for (int t = tris.Count - 3; t >= 0; t -= 3)
            {
                int i0 = tris[t], i1 = tris[t + 1], i2 = tris[t + 2];
                int mask = (IsOpen(i0, i1) ? 1 : 0) | (IsOpen(i1, i2) ? 2 : 0) | (IsOpen(i2, i0) ? 4 : 0);
                work.Push((i0, i1, i2, mask));
            }

            while (work.Count > 0)
            {
                var (i0, i1, i2, mask) = work.Pop();
                int split = -1, e;
                for (e = 0; e < 3; e++)
                {
                    if ((mask & (1 << e)) == 0) continue;
                    int u = e == 0 ? i0 : e == 1 ? i1 : i2;
                    int v = e == 0 ? i1 : e == 1 ? i2 : i0;
                    split = FindOnEdge(verts, sorted, keys, u, v, tol);
                    if (split >= 0) break;
                }
                if (split < 0)
                {
                    output.Add(i0); output.Add(i1); output.Add(i2);
                    continue;
                }
                // Rotate so the split edge is (a,b), opposite corner c; children (k,b,c) and (a,k,c).
                int a = e == 0 ? i0 : e == 1 ? i1 : i2;
                int b = e == 0 ? i1 : e == 1 ? i2 : i0;
                int c2 = e == 0 ? i2 : e == 1 ? i0 : i1;
                bool openBC = (mask & (1 << ((e + 1) % 3))) != 0;
                bool openCA = (mask & (1 << ((e + 2) % 3))) != 0;
                work.Push((split, b, c2, 1 | (openBC ? 2 : 0)));
                work.Push((a, split, c2, 1 | (openCA ? 4 : 0)));
            }
            return output;
        }

        private static long EdgeKey(int u, int v) => ((long)u << 32) | (uint)v;

        private static int FindOnEdge(List<V3> verts, int[][] sorted, double[][] keys, int u, int v, double tol)
        {
            V3 pu = verts[u], pv = verts[v];
            V3 e = pv - pu;
            double lenSq = V3.Dot(e, e);
            double len = Math.Sqrt(lenSq);
            if (len <= 2 * tol) return -1;

            int axis = 0;
            double best = double.MaxValue;
            for (int a = 0; a < 3; a++)
            {
                double ext = Math.Abs(Coord(e, a));
                if (ext < best) { best = ext; axis = a; }
            }
            double lo = Math.Min(Coord(pu, axis), Coord(pv, axis)) - tol;
            double hi = Math.Max(Coord(pu, axis), Coord(pv, axis)) + tol;
            int start = LowerBound(keys[axis], lo);

            int found = -1;
            double foundT = double.MaxValue;
            for (int s = start; s < keys[axis].Length && keys[axis][s] <= hi; s++)
            {
                int k = sorted[axis][s];
                if (k == u || k == v) continue;
                V3 d = verts[k] - pu;
                double along = V3.Dot(d, e) / len;
                if (along <= tol || along >= len - tol) continue;
                V3 perp = d - e * (along / len);
                if (V3.Dot(perp, perp) > tol * tol) continue;
                if (along < foundT || (along == foundT && k < found)) { foundT = along; found = k; }
            }
            return found;
        }

        private static int LowerBound(double[] arr, double value)
        {
            int lo = 0, hi = arr.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (arr[mid] < value) lo = mid + 1; else hi = mid;
            }
            return lo;
        }

        private static double Coord(V3 v, int axis) => axis == 0 ? v.X : axis == 1 ? v.Y : v.Z;

        /// <summary>True if every directed edge (u,v) is matched by the same number of opposite edges (v,u).</summary>
        public static bool CheckClosedness(TriMesh mesh)
        {
            if (mesh == null || mesh.Triangles == null || mesh.Triangles.Length == 0)
                return false;

            var edgeCounts = new Dictionary<long, int>(KeyComparer.Instance);
            long Key(int u, int v) => ((long)u << 32) | (uint)v;
            for (int t = 0; t < mesh.Triangles.Length; t += 3)
            {
                for (int e = 0; e < 3; e++)
                {
                    long k = Key(mesh.Triangles[t + e], mesh.Triangles[t + (e + 1) % 3]);
                    edgeCounts.TryGetValue(k, out int c);
                    edgeCounts[k] = c + 1;
                }
            }
            foreach (var kvp in edgeCounts)
            {
                int u = (int)(kvp.Key >> 32), v = (int)(kvp.Key & 0xFFFFFFFF);
                if (!edgeCounts.TryGetValue(Key(v, u), out int opp) || opp != kvp.Value)
                    return false;
            }
            return true;
        }

        private static void GetBounds(TriMesh mesh, out V3 min, out V3 max)
        {
            min = new V3(double.MaxValue, double.MaxValue, double.MaxValue);
            max = new V3(double.MinValue, double.MinValue, double.MinValue);
            foreach (var v in mesh.Vertices)
            {
                min = V3.Min(min, v);
                max = V3.Max(max, v);
            }
        }

        private static V3 Cross(V3 a, V3 b) => new V3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    }
}
