using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    /// <summary>
    /// Exact Brep edge sewing and joining kernel (M4 API).
    /// Corresponds to RhinoCommon Brep.JoinBreps(pieces, tol).
    /// </summary>
    public static class BrepJoin
    {
        private sealed class WorkingVertex
        {
            public V3 Point;
            public double Tolerance;
            public List<int> Edges = new List<int>();
        }

        private sealed class WorkingEdge
        {
            public int Curve3 = -1;
            public Interval Domain;
            public int V0 = -1;
            public int V1 = -1;
            public List<int> Trims = new List<int>();
            public double Tolerance;
        }

        private sealed class WorkingTrim
        {
            public int Curve2 = -1;
            public Interval Domain;
            public int Edge = -1;
            public int Loop = -1;
            public bool Reversed3d;
            public TrimType Type;
            public double TolU;
            public double TolV;
        }

        private sealed class WorkingLoop
        {
            public int Face = -1;
            public LoopType Type;
            public List<int> Trims = new List<int>();
        }

        private sealed class WorkingFace
        {
            public int Surface = -1;
            public bool Reversed;
            public List<int> Loops = new List<int>();
        }

        private sealed class UnionFind
        {
            private readonly int[] _parent;

            public UnionFind(int count)
            {
                _parent = new int[count];
                for (int i = 0; i < count; i++) _parent[i] = i;
            }

            public int Find(int i)
            {
                int root = i;
                while (root != _parent[root])
                    root = _parent[root];
                int curr = i;
                while (curr != root)
                {
                    int next = _parent[curr];
                    _parent[curr] = root;
                    curr = next;
                }
                return root;
            }

            public bool Union(int i, int j)
            {
                int rootI = Find(i);
                int rootJ = Find(j);
                if (rootI == rootJ) return false;
                _parent[rootJ] = rootI;
                return true;
            }
        }

        /// <summary>
        /// Joins an array of Breps into connected manifold components within the given tolerance.
        /// </summary>
        public static List<KBrep> Join(IReadOnlyList<KBrep> pieces, double tolerance)
        {
            if (pieces == null || pieces.Count == 0)
                return new List<KBrep>();

            double tol = tolerance > 0 ? tolerance : 0.001;

            // Stage 1: Topological Aggregation into working buffers
            var surfaces = new List<NurbsSurface>();
            var curves3 = new List<NurbsCurve>();
            var curves2 = new List<NurbsCurve>();
            var vertices = new List<WorkingVertex>();
            var edges = new List<WorkingEdge>();
            var trims = new List<WorkingTrim>();
            var loops = new List<WorkingLoop>();
            var faces = new List<WorkingFace>();

            foreach (var piece in pieces)
            {
                if (piece == null || piece.Faces.Count == 0) continue;

                int srfOffset = surfaces.Count;
                int c3Offset = curves3.Count;
                int c2Offset = curves2.Count;
                int vOffset = vertices.Count;
                int eOffset = edges.Count;
                int tOffset = trims.Count;
                int lOffset = loops.Count;
                int fOffset = faces.Count;

                surfaces.AddRange(piece.Surfaces);
                curves3.AddRange(piece.Curves3);
                curves2.AddRange(piece.Curves2);

                for (int vi = 0; vi < piece.Vertices.Count; vi++)
                {
                    var pv = piece.Vertices[vi];
                    var wv = new WorkingVertex
                    {
                        Point = pv.Point,
                        Tolerance = pv.Tolerance
                    };
                    if (pv.Edges != null)
                    {
                        foreach (var e in pv.Edges) wv.Edges.Add(e + eOffset);
                    }
                    vertices.Add(wv);
                }

                for (int ei = 0; ei < piece.Edges.Count; ei++)
                {
                    var pe = piece.Edges[ei];
                    var we = new WorkingEdge
                    {
                        Curve3 = pe.Curve3 >= 0 ? pe.Curve3 + c3Offset : -1,
                        Domain = pe.Domain,
                        V0 = pe.V0 >= 0 ? pe.V0 + vOffset : -1,
                        V1 = pe.V1 >= 0 ? pe.V1 + vOffset : -1,
                        Tolerance = pe.Tolerance
                    };
                    if (pe.Trims != null)
                    {
                        foreach (var t in pe.Trims) we.Trims.Add(t + tOffset);
                    }
                    edges.Add(we);
                }

                for (int ti = 0; ti < piece.Trims.Count; ti++)
                {
                    var pt = piece.Trims[ti];
                    var wt = new WorkingTrim
                    {
                        Curve2 = pt.Curve2 >= 0 ? pt.Curve2 + c2Offset : -1,
                        Domain = pt.Domain,
                        Edge = pt.Edge >= 0 ? pt.Edge + eOffset : -1,
                        Loop = pt.Loop >= 0 ? pt.Loop + lOffset : -1,
                        Reversed3d = pt.Reversed3d,
                        Type = pt.Type,
                        TolU = pt.TolU,
                        TolV = pt.TolV
                    };
                    trims.Add(wt);
                }

                for (int li = 0; li < piece.Loops.Count; li++)
                {
                    var pl = piece.Loops[li];
                    var wl = new WorkingLoop
                    {
                        Face = pl.Face >= 0 ? pl.Face + fOffset : -1,
                        Type = pl.Type
                    };
                    if (pl.Trims != null)
                    {
                        foreach (var t in pl.Trims) wl.Trims.Add(t + tOffset);
                    }
                    loops.Add(wl);
                }

                for (int fi = 0; fi < piece.Faces.Count; fi++)
                {
                    var pf = piece.Faces[fi];
                    var wf = new WorkingFace
                    {
                        Surface = pf.Surface >= 0 ? pf.Surface + srfOffset : -1,
                        Reversed = pf.Reversed
                    };
                    if (pf.Loops != null)
                    {
                        foreach (var l in pf.Loops) wf.Loops.Add(l + lOffset);
                    }
                    faces.Add(wf);
                }
            }

            if (faces.Count == 0)
                return new List<KBrep>();

            // Stage 2: Vertex-Vertex Clustering (TL_Join_VVData)
            ClusterVertices(vertices, edges, tol);

            // Stage 3: Vertex-Edge T-Junction Resolution (TL_Join_VEData)
            ResolveTJunctions(vertices, edges, trims, loops, curves3, tol);

            // Stage 4: Naked Edge Matching & Sewing (TL_BrepJoin::MatchEdges)
            MatchAndSewEdges(vertices, edges, trims, loops, faces, surfaces, curves3, curves2, tol);

            // Stage 5: Face Orientation Reconciliation
            ReconcileFaceOrientations(vertices, edges, trims, loops, faces, surfaces, curves2);

            // Stage 6: Split into Connected Components and Package Output Breps
            return PackageConnectedComponents(vertices, edges, trims, loops, faces, surfaces, curves3, curves2);
        }

        #region Stage 2: Vertex Clustering

        private static void ClusterVertices(List<WorkingVertex> vertices, List<WorkingEdge> edges, double tol)
        {
            int count = vertices.Count;
            if (count <= 1) return;

            var uf = new UnionFind(count);
            double cellSize = tol > 1e-12 ? tol : 0.001;
            var grid = new Dictionary<long, List<int>>();

            for (int i = 0; i < count; i++)
            {
                var pt = vertices[i].Point;
                long cx = (long)Math.Floor(pt.X / cellSize);
                long cy = (long)Math.Floor(pt.Y / cellSize);
                long cz = (long)Math.Floor(pt.Z / cellSize);

                // Check 27 neighboring cells
                for (long dx = -1; dx <= 1; dx++)
                {
                    for (long dy = -1; dy <= 1; dy++)
                    {
                        for (long dz = -1; dz <= 1; dz++)
                        {
                            long key = HashGridKey(cx + dx, cy + dy, cz + dz);
                            if (grid.TryGetValue(key, out var bucket))
                            {
                                foreach (var j in bucket)
                                {
                                    double dist = (pt - vertices[j].Point).Length;
                                    if (dist <= tol)
                                    {
                                        uf.Union(i, j);
                                    }
                                }
                            }
                        }
                    }
                }

                long selfKey = HashGridKey(cx, cy, cz);
                if (!grid.TryGetValue(selfKey, out var selfBucket))
                {
                    selfBucket = new List<int>();
                    grid[selfKey] = selfBucket;
                }
                selfBucket.Add(i);
            }

            // Compute cluster centroids
            var clusterVertices = new Dictionary<int, List<int>>();
            for (int i = 0; i < count; i++)
            {
                int root = uf.Find(i);
                if (!clusterVertices.TryGetValue(root, out var list))
                {
                    list = new List<int>();
                    clusterVertices[root] = list;
                }
                list.Add(i);
            }

            foreach (var kvp in clusterVertices)
            {
                int root = kvp.Key;
                var list = kvp.Value;
                if (list.Count > 1)
                {
                    double sumX = 0, sumY = 0, sumZ = 0;
                    double maxTol = 0;
                    foreach (var idx in list)
                    {
                        sumX += vertices[idx].Point.X;
                        sumY += vertices[idx].Point.Y;
                        sumZ += vertices[idx].Point.Z;
                        if (vertices[idx].Tolerance > maxTol) maxTol = vertices[idx].Tolerance;
                    }
                    double inv = 1.0 / list.Count;
                    vertices[root].Point = new V3(sumX * inv, sumY * inv, sumZ * inv);
                    vertices[root].Tolerance = Math.Max(maxTol, tol);
                }
            }

            // Remap edge endpoints to cluster roots
            for (int e = 0; e < edges.Count; e++)
            {
                if (edges[e].V0 >= 0) edges[e].V0 = uf.Find(edges[e].V0);
                if (edges[e].V1 >= 0) edges[e].V1 = uf.Find(edges[e].V1);
            }

            // Rebuild vertex Edges lists
            for (int i = 0; i < vertices.Count; i++) vertices[i].Edges.Clear();
            for (int e = 0; e < edges.Count; e++)
            {
                if (edges[e].V0 >= 0 && edges[e].V0 < vertices.Count)
                    vertices[edges[e].V0].Edges.Add(e);
                if (edges[e].V1 >= 0 && edges[e].V1 < vertices.Count && edges[e].V1 != edges[e].V0)
                    vertices[edges[e].V1].Edges.Add(e);
            }
        }

        private static long HashGridKey(long x, long y, long z)
        {
            unchecked
            {
                long h = 17;
                h = h * 31 + x;
                h = h * 31 + y;
                h = h * 31 + z;
                return h;
            }
        }

        #endregion

        #region Stage 3: T-Junction Resolution

        private static void ResolveTJunctions(
            List<WorkingVertex> vertices,
            List<WorkingEdge> edges,
            List<WorkingTrim> trims,
            List<WorkingLoop> loops,
            List<NurbsCurve> curves3,
            double tol)
        {
            bool splitHappened = true;
            int iteration = 0;
            const int maxIterations = 200;

            while (splitHappened && iteration++ < maxIterations)
            {
                splitHappened = false;

                // Identify naked edges
                var nakedEdges = new List<int>();
                for (int e = 0; e < edges.Count; e++)
                {
                    if (edges[e].Trims.Count == 1 && edges[e].V0 >= 0 && edges[e].V1 >= 0 && edges[e].V0 != edges[e].V1)
                        nakedEdges.Add(e);
                }

                // Identify naked vertices
                var nakedVertexSet = new HashSet<int>();
                foreach (var e in nakedEdges)
                {
                    nakedVertexSet.Add(edges[e].V0);
                    nakedVertexSet.Add(edges[e].V1);
                }
                var nakedVertices = new List<int>(nakedVertexSet);

                foreach (var eIdx in nakedEdges)
                {
                    var edge = edges[eIdx];
                    if (edge.Trims.Count != 1) continue;

                    var crv = curves3[edge.Curve3];
                    var dom = edge.Domain;
                    var p0 = NurbsEval.CurvePoint(crv, dom.T0);
                    var p1 = NurbsEval.CurvePoint(crv, dom.T1);

                    var splits = new List<(double t, int vIdx)>();

                    foreach (var vIdx in nakedVertices)
                    {
                        if (vIdx == edge.V0 || vIdx == edge.V1) continue;
                        var vPt = vertices[vIdx].Point;

                        // Fast check: distance to endpoints must exceed tol
                        if ((vPt - p0).Length <= tol || (vPt - p1).Length <= tol)
                            continue;

                        if (ClosestPointOnCurve(crv, dom, vPt, out double bestT, out double bestDist))
                        {
                            if (bestDist <= tol)
                            {
                                double paramTol = 1e-6 * dom.Length;
                                if (bestT > dom.T0 + paramTol && bestT < dom.T1 - paramTol)
                                {
                                    splits.Add((bestT, vIdx));
                                }
                            }
                        }
                    }

                    if (splits.Count > 0)
                    {
                        splits.Sort((a, b) => a.t.CompareTo(b.t));

                        // Deduplicate close split points
                        var uniqueSplits = new List<(double t, int vIdx)>();
                        foreach (var sp in splits)
                        {
                            if (uniqueSplits.Count == 0 || Math.Abs(sp.t - uniqueSplits[uniqueSplits.Count - 1].t) > 1e-6 * dom.Length)
                                uniqueSplits.Add(sp);
                        }

                        if (uniqueSplits.Count > 0)
                        {
                            SplitNakedEdge(eIdx, uniqueSplits, vertices, edges, trims, loops);
                            splitHappened = true;
                            break; // restart scan with updated topology
                        }
                    }
                }
            }
        }

        private static void SplitNakedEdge(
            int eIdx,
            List<(double t, int vIdx)> splits,
            List<WorkingVertex> vertices,
            List<WorkingEdge> edges,
            List<WorkingTrim> trims,
            List<WorkingLoop> loops)
        {
            var edge = edges[eIdx];
            int trimIdx = edge.Trims[0];
            var trim = trims[trimIdx];
            var loop = loops[trim.Loop];

            double t0 = edge.Domain.T0;
            double t1 = edge.Domain.T1;
            int v0 = edge.V0;
            int v1 = edge.V1;
            double s0 = trim.Domain.T0;
            double s1 = trim.Domain.T1;
            bool rev3d = trim.Reversed3d;

            int m = splits.Count;
            // Build sub-intervals along the edge in forward curve direction
            var subEdgeIndices = new List<int>(m + 1);

            // Sub-edge 0 modifies eIdx in place
            edge.Domain = new Interval(t0, splits[0].t);
            edge.V0 = v0;
            edge.V1 = splits[0].vIdx;
            edge.Trims.Clear();
            subEdgeIndices.Add(eIdx);

            for (int j = 1; j <= m; j++)
            {
                double subT0 = splits[j - 1].t;
                double subT1 = (j == m) ? t1 : splits[j].t;
                int subV0 = splits[j - 1].vIdx;
                int subV1 = (j == m) ? v1 : splits[j].vIdx;

                var newEdge = new WorkingEdge
                {
                    Curve3 = edge.Curve3,
                    Domain = new Interval(subT0, subT1),
                    V0 = subV0,
                    V1 = subV1,
                    Tolerance = edge.Tolerance
                };
                int newEdgeIdx = edges.Count;
                edges.Add(newEdge);
                subEdgeIndices.Add(newEdgeIdx);
            }

            // Create sub-trims
            var subTrimIndices = new List<int>(m + 1);

            if (!rev3d)
            {
                // Trim runs in forward edge direction: s matches t
                double totalT = t1 - t0;
                double totalS = s1 - s0;

                for (int j = 0; j <= m; j++)
                {
                    double subT0 = (j == 0) ? t0 : splits[j - 1].t;
                    double subT1 = (j == m) ? t1 : splits[j].t;
                    double subS0 = s0 + ((subT0 - t0) / totalT) * totalS;
                    double subS1 = s0 + ((subT1 - t0) / totalT) * totalS;

                    int targetEdge = subEdgeIndices[j];

                    if (j == 0)
                    {
                        trim.Domain = new Interval(subS0, subS1);
                        trim.Edge = targetEdge;
                        edges[targetEdge].Trims.Add(trimIdx);
                        subTrimIndices.Add(trimIdx);
                    }
                    else
                    {
                        var newTrim = new WorkingTrim
                        {
                            Curve2 = trim.Curve2,
                            Domain = new Interval(subS0, subS1),
                            Edge = targetEdge,
                            Loop = trim.Loop,
                            Reversed3d = false,
                            Type = TrimType.Boundary,
                            TolU = trim.TolU,
                            TolV = trim.TolV
                        };
                        int newTrimIdx = trims.Count;
                        trims.Add(newTrim);
                        edges[targetEdge].Trims.Add(newTrimIdx);
                        subTrimIndices.Add(newTrimIdx);
                    }
                }
            }
            else
            {
                // Trim runs in reverse edge direction: s0 is at t1, s1 is at t0
                double totalT = t1 - t0;
                double totalS = s1 - s0;

                // As s increases from s0 to s1, t decreases from t1 to t0.
                // Sub-edges in reverse order: m, m-1, ..., 0
                for (int j = 0; j <= m; j++)
                {
                    int edgeIdxInReverse = m - j;
                    double subT0 = (edgeIdxInReverse == 0) ? t0 : splits[edgeIdxInReverse - 1].t;
                    double subT1 = (edgeIdxInReverse == m) ? t1 : splits[edgeIdxInReverse].t;

                    // s at subT1 (which is earlier along trim than subT0)
                    double subS0 = s0 + ((t1 - subT1) / totalT) * totalS;
                    double subS1 = s0 + ((t1 - subT0) / totalT) * totalS;

                    int targetEdge = subEdgeIndices[edgeIdxInReverse];

                    if (j == 0)
                    {
                        trim.Domain = new Interval(subS0, subS1);
                        trim.Edge = targetEdge;
                        edges[targetEdge].Trims.Add(trimIdx);
                        subTrimIndices.Add(trimIdx);
                    }
                    else
                    {
                        var newTrim = new WorkingTrim
                        {
                            Curve2 = trim.Curve2,
                            Domain = new Interval(subS0, subS1),
                            Edge = targetEdge,
                            Loop = trim.Loop,
                            Reversed3d = true,
                            Type = TrimType.Boundary,
                            TolU = trim.TolU,
                            TolV = trim.TolV
                        };
                        int newTrimIdx = trims.Count;
                        trims.Add(newTrim);
                        edges[targetEdge].Trims.Add(newTrimIdx);
                        subTrimIndices.Add(newTrimIdx);
                    }
                }
            }

            // Replace trim in loop's trims list
            int posInLoop = loop.Trims.IndexOf(trimIdx);
            if (posInLoop >= 0)
            {
                loop.Trims.RemoveAt(posInLoop);
                loop.Trims.InsertRange(posInLoop, subTrimIndices);
            }
        }

        #endregion

        #region Stage 4: Edge Matching & Sewing

        private sealed class MatchCandidate
        {
            public int E1;
            public int E2;
            public double Deviation;
            public bool IsParallel;
        }

        private static void MatchAndSewEdges(
            List<WorkingVertex> vertices,
            List<WorkingEdge> edges,
            List<WorkingTrim> trims,
            List<WorkingLoop> loops,
            List<WorkingFace> faces,
            List<NurbsSurface> surfaces,
            List<NurbsCurve> curves3,
            List<NurbsCurve> curves2,
            double tol)
        {
            // Collect naked edges. Closed edges (V0 == V1, e.g. a sphere/plane SSI circle) are included: they mate with
            // another closed edge on the same vertex; their relative direction is decided by tangents below.
            var nakedEdges = new List<int>();
            for (int e = 0; e < edges.Count; e++)
            {
                if (edges[e].Trims.Count == 1 && edges[e].V0 >= 0 && edges[e].V1 >= 0)
                    nakedEdges.Add(e);
            }

            var candidates = new List<MatchCandidate>();

            for (int i = 0; i < nakedEdges.Count; i++)
            {
                int e1 = nakedEdges[i];
                var edge1 = edges[e1];
                var crv1 = curves3[edge1.Curve3];

                for (int j = i + 1; j < nakedEdges.Count; j++)
                {
                    int e2 = nakedEdges[j];
                    var edge2 = edges[e2];

                    bool closed1 = edge1.V0 == edge1.V1, closed2 = edge2.V0 == edge2.V1;
                    if (closed1 != closed2) continue;
                    bool isParallel = (edge1.V0 == edge2.V0 && edge1.V1 == edge2.V1);
                    bool isAntiParallel = (edge1.V0 == edge2.V1 && edge1.V1 == edge2.V0);

                    if (!isParallel && !isAntiParallel)
                        continue;

                    var crv2 = curves3[edge2.Curve3];
                    // closed pair: both flags are true, so the direction comes from the tangents
                    if (closed1) isParallel = TangentsAgree(crv1, edge1.Domain, crv2, edge2.Domain);

                    // Sample Hausdorff deviation
                    double dev = MeasureEdgeDeviation(crv1, edge1.Domain, crv2, edge2.Domain, 10);
                    if (dev <= tol)
                    {
                        candidates.Add(new MatchCandidate
                        {
                            E1 = e1,
                            E2 = e2,
                            Deviation = dev,
                            IsParallel = isParallel
                        });
                    }
                }
            }

            // Sort candidates by deviation ascending (best match first)
            candidates.Sort((a, b) => a.Deviation.CompareTo(b.Deviation));

            var matedSet = new HashSet<int>();

            foreach (var cand in candidates)
            {
                if (matedSet.Contains(cand.E1) || matedSet.Contains(cand.E2))
                    continue;

                int e1 = cand.E1;
                int e2 = cand.E2;
                var edge1 = edges[e1];
                var edge2 = edges[e2];

                if (edge1.Trims.Count != 1 || edge2.Trims.Count != 1)
                    continue;

                int t1 = edge1.Trims[0];
                int t2 = edge2.Trims[0];

                matedSet.Add(e1);
                matedSet.Add(e2);

                // Mate edge2 into edge1
                edge1.Trims.Add(t2);
                trims[t2].Edge = e1;
                trims[t1].Type = TrimType.Mated;
                trims[t2].Type = TrimType.Mated;
                edge1.Tolerance = Math.Max(edge1.Tolerance, Math.Max(edge2.Tolerance, cand.Deviation));

                // t1 keeps its direction relative to edge1; t2 was relative to edge2, so it toggles when edge2 runs
                // against edge1. (Recomputing from the trim start vertex fails on closed edges where V0 == V1.)
                if (!cand.IsParallel) trims[t2].Reversed3d = !trims[t2].Reversed3d;

                // Clear trims from edge2 (marking it inactive)
                edge2.Trims.Clear();
            }
        }

        // Closed edges sharing their start vertex: same direction when the tangents at a quarter of edge2 agree.
        private static bool TangentsAgree(NurbsCurve crv1, Interval dom1, NurbsCurve crv2, Interval dom2)
        {
            double t2 = dom2.ParameterAt(0.25);
            var d2 = NurbsEval.CurveDerivatives(crv2, t2, 1);
            ClosestPointOnCurve(crv1, dom1, d2[0], out double t1, out _);
            var d1 = NurbsEval.CurveDerivatives(crv1, t1, 1);
            return V3.Dot(d1[1], d2[1]) >= 0;
        }

        private static double MeasureEdgeDeviation(
            NurbsCurve crv1, Interval dom1,
            NurbsCurve crv2, Interval dom2,
            int sampleCount)
        {
            double maxDev = 0;

            for (int k = 0; k <= sampleCount; k++)
            {
                double s = k / (double)sampleCount;
                double t1 = dom1.ParameterAt(s);
                var p1 = NurbsEval.CurvePoint(crv1, t1);

                if (ClosestPointOnCurve(crv2, dom2, p1, out _, out double dist))
                {
                    if (dist > maxDev) maxDev = dist;
                }
            }

            for (int k = 0; k <= sampleCount; k++)
            {
                double s = k / (double)sampleCount;
                double t2 = dom2.ParameterAt(s);
                var p2 = NurbsEval.CurvePoint(crv2, t2);

                if (ClosestPointOnCurve(crv1, dom1, p2, out _, out double dist))
                {
                    if (dist > maxDev) maxDev = dist;
                }
            }

            return maxDev;
        }

        #endregion

        #region Stage 5: Face Orientation Reconciliation

        private static void ReconcileFaceOrientations(
            List<WorkingVertex> vertices,
            List<WorkingEdge> edges,
            List<WorkingTrim> trims,
            List<WorkingLoop> loops,
            List<WorkingFace> faces,
            List<NurbsSurface> surfaces,
            List<NurbsCurve> curves2)
        {
            int faceCount = faces.Count;
            if (faceCount <= 1) return;

            // Adjacency graph between faces via mated edges
            // Edge: (faceA, faceB, matedEdgeIndex)
            var adj = new List<(int otherFace, int edgeIdx)>[faceCount];
            for (int f = 0; f < faceCount; f++) adj[f] = new List<(int, int)>();

            for (int e = 0; e < edges.Count; e++)
            {
                if (edges[e].Trims.Count == 2)
                {
                    int t0 = edges[e].Trims[0];
                    int t1 = edges[e].Trims[1];
                    int f0 = loops[trims[t0].Loop].Face;
                    int f1 = loops[trims[t1].Loop].Face;
                    if (f0 != f1 && f0 >= 0 && f1 >= 0)
                    {
                        adj[f0].Add((f1, e));
                        adj[f1].Add((f0, e));
                    }
                }
            }

            var visited = new bool[faceCount];
            var faceFlipped = new bool[faceCount];

            for (int root = 0; root < faceCount; root++)
            {
                if (visited[root]) continue;

                var queue = new Queue<int>();
                queue.Enqueue(root);
                visited[root] = true;

                while (queue.Count > 0)
                {
                    int curr = queue.Dequeue();

                    foreach (var (next, edgeIdx) in adj[curr])
                    {
                        var edge = edges[edgeIdx];
                        int tCurr = -1;
                        int tNext = -1;

                        foreach (var t in edge.Trims)
                        {
                            int f = loops[trims[t].Loop].Face;
                            if (f == curr) tCurr = t;
                            else if (f == next) tNext = t;
                        }

                        if (tCurr < 0 || tNext < 0) continue;

                        // Effective direction of a trim along its edge on the oriented face is
                        // Reversed3d XOR face.Reversed (ON_BrepFace::m_bRev flips the face normal, the UV loop stays CCW).
                        bool dirCurr = trims[tCurr].Reversed3d ^ faces[curr].Reversed ^ faceFlipped[curr];
                        // Desired direction of tNext along edge is opposite to dirCurr
                        bool desiredDirNext = !dirCurr;

                        if (!visited[next])
                        {
                            visited[next] = true;
                            faceFlipped[next] = (trims[tNext].Reversed3d ^ faces[next].Reversed) != desiredDirNext;
                            queue.Enqueue(next);
                        }
                    }
                }
            }

            // Apply flips like ON_Brep::Flip / RhinoCommon Brep.Flip: toggle m_bRev only. Reversing the loop's trim
            // order without reversing the 2D curves breaks loop continuity (brep.m_L[i] loop is not valid).
            for (int f = 0; f < faceCount; f++)
            {
                if (faceFlipped[f]) faces[f].Reversed = !faces[f].Reversed;
            }
        }

        #endregion

        #region Stage 6: Connected Components & Packaging

        private static List<KBrep> PackageConnectedComponents(
            List<WorkingVertex> vertices,
            List<WorkingEdge> edges,
            List<WorkingTrim> trims,
            List<WorkingLoop> loops,
            List<WorkingFace> faces,
            List<NurbsSurface> surfaces,
            List<NurbsCurve> curves3,
            List<NurbsCurve> curves2)
        {
            int faceCount = faces.Count;
            var uf = new UnionFind(faceCount);

            // Faces sharing a mated edge belong to the same connected component
            for (int e = 0; e < edges.Count; e++)
            {
                if (edges[e].Trims.Count == 2)
                {
                    int f0 = loops[trims[edges[e].Trims[0]].Loop].Face;
                    int f1 = loops[trims[edges[e].Trims[1]].Loop].Face;
                    if (f0 >= 0 && f1 >= 0)
                    {
                        uf.Union(f0, f1);
                    }
                }
            }

            // Group faces by component root
            var components = new Dictionary<int, List<int>>();
            for (int f = 0; f < faceCount; f++)
            {
                int root = uf.Find(f);
                if (!components.TryGetValue(root, out var list))
                {
                    list = new List<int>();
                    components[root] = list;
                }
                list.Add(f);
            }

            var result = new List<KBrep>();

            foreach (var kvp in components)
            {
                var compFaces = kvp.Value;
                var brep = CompactBrepComponent(compFaces, vertices, edges, trims, loops, faces, surfaces, curves3, curves2);
                result.Add(brep);
            }

            return result;
        }

        private static KBrep CompactBrepComponent(
            List<int> compFaces,
            List<WorkingVertex> vertices,
            List<WorkingEdge> edges,
            List<WorkingTrim> trims,
            List<WorkingLoop> loops,
            List<WorkingFace> faces,
            List<NurbsSurface> surfaces,
            List<NurbsCurve> curves3,
            List<NurbsCurve> curves2)
        {
            var brep = new KBrep();

            var surfaceMap = new Dictionary<int, int>();
            var curve3Map = new Dictionary<int, int>();
            var curve2Map = new Dictionary<int, int>();
            var vertexMap = new Dictionary<int, int>();
            var edgeMap = new Dictionary<int, int>();
            var trimMap = new Dictionary<int, int>();
            var loopMap = new Dictionary<int, int>();
            var faceMap = new Dictionary<int, int>();

            // Collect all loops in this component
            var compLoops = new List<int>();
            foreach (var fIdx in compFaces)
            {
                faceMap[fIdx] = brep.Faces.Count;
                brep.Faces.Add(null); // placeholder
                foreach (var lIdx in faces[fIdx].Loops)
                {
                    if (!loopMap.ContainsKey(lIdx))
                    {
                        loopMap[lIdx] = compLoops.Count;
                        compLoops.Add(lIdx);
                    }
                }
            }

            // Collect all trims in this component
            var compTrims = new List<int>();
            foreach (var lIdx in compLoops)
            {
                foreach (var tIdx in loops[lIdx].Trims)
                {
                    if (!trimMap.ContainsKey(tIdx))
                    {
                        trimMap[tIdx] = compTrims.Count;
                        compTrims.Add(tIdx);
                    }
                }
            }

            // Collect all edges in this component
            var compEdges = new List<int>();
            foreach (var tIdx in compTrims)
            {
                int eIdx = trims[tIdx].Edge;
                if (eIdx >= 0 && !edgeMap.ContainsKey(eIdx))
                {
                    edgeMap[eIdx] = compEdges.Count;
                    compEdges.Add(eIdx);
                }
            }

            // Collect all vertices in this component
            var compVertices = new List<int>();
            foreach (var eIdx in compEdges)
            {
                int v0 = edges[eIdx].V0;
                int v1 = edges[eIdx].V1;
                if (v0 >= 0 && !vertexMap.ContainsKey(v0))
                {
                    vertexMap[v0] = compVertices.Count;
                    compVertices.Add(v0);
                }
                if (v1 >= 0 && !vertexMap.ContainsKey(v1))
                {
                    vertexMap[v1] = compVertices.Count;
                    compVertices.Add(v1);
                }
            }

            // Copy surfaces
            foreach (var fIdx in compFaces)
            {
                int sIdx = faces[fIdx].Surface;
                if (sIdx >= 0 && !surfaceMap.ContainsKey(sIdx))
                {
                    surfaceMap[sIdx] = brep.Surfaces.Count;
                    brep.Surfaces.Add(surfaces[sIdx]);
                }
            }

            // Copy curves3
            foreach (var eIdx in compEdges)
            {
                int c3Idx = edges[eIdx].Curve3;
                if (c3Idx >= 0 && !curve3Map.ContainsKey(c3Idx))
                {
                    curve3Map[c3Idx] = brep.Curves3.Count;
                    brep.Curves3.Add(curves3[c3Idx]);
                }
            }

            // Copy curves2
            foreach (var tIdx in compTrims)
            {
                int c2Idx = trims[tIdx].Curve2;
                if (c2Idx >= 0 && !curve2Map.ContainsKey(c2Idx))
                {
                    curve2Map[c2Idx] = brep.Curves2.Count;
                    brep.Curves2.Add(curves2[c2Idx]);
                }
            }

            // Copy vertices
            foreach (var vIdx in compVertices)
            {
                var wv = vertices[vIdx];
                var kv = new KVertex
                {
                    Point = wv.Point,
                    Tolerance = wv.Tolerance
                };
                brep.Vertices.Add(kv);
            }

            // Copy edges
            foreach (var eIdx in compEdges)
            {
                var we = edges[eIdx];
                var ke = new KEdge
                {
                    Curve3 = we.Curve3 >= 0 && curve3Map.TryGetValue(we.Curve3, out int newC3) ? newC3 : -1,
                    Domain = we.Domain,
                    V0 = we.V0 >= 0 && vertexMap.TryGetValue(we.V0, out int newV0) ? newV0 : -1,
                    V1 = we.V1 >= 0 && vertexMap.TryGetValue(we.V1, out int newV1) ? newV1 : -1,
                    Tolerance = we.Tolerance
                };
                foreach (var t in we.Trims)
                {
                    if (trimMap.TryGetValue(t, out int newT)) ke.Trims.Add(newT);
                }
                brep.Edges.Add(ke);
            }

            // Copy trims
            foreach (var tIdx in compTrims)
            {
                var wt = trims[tIdx];
                var kt = new KTrim
                {
                    Curve2 = wt.Curve2 >= 0 && curve2Map.TryGetValue(wt.Curve2, out int newC2) ? newC2 : -1,
                    Domain = wt.Domain,
                    Edge = wt.Edge >= 0 && edgeMap.TryGetValue(wt.Edge, out int newE) ? newE : -1,
                    Loop = wt.Loop >= 0 && loopMap.TryGetValue(wt.Loop, out int newL) ? newL : -1,
                    Reversed3d = wt.Reversed3d,
                    Type = wt.Type,
                    TolU = wt.TolU,
                    TolV = wt.TolV
                };
                brep.Trims.Add(kt);
            }

            // Copy loops
            foreach (var lIdx in compLoops)
            {
                var wl = loops[lIdx];
                var kl = new KLoop
                {
                    Face = wl.Face >= 0 && faceMap.TryGetValue(wl.Face, out int newF) ? newF : -1,
                    Type = wl.Type
                };
                foreach (var t in wl.Trims)
                {
                    if (trimMap.TryGetValue(t, out int newT)) kl.Trims.Add(newT);
                }
                brep.Loops.Add(kl);
            }

            // Copy faces
            foreach (var fIdx in compFaces)
            {
                var wf = faces[fIdx];
                int newF = faceMap[fIdx];
                var kf = new KFace
                {
                    Surface = wf.Surface >= 0 && surfaceMap.TryGetValue(wf.Surface, out int newS) ? newS : -1,
                    Reversed = wf.Reversed
                };
                foreach (var l in wf.Loops)
                {
                    if (loopMap.TryGetValue(l, out int newL)) kf.Loops.Add(newL);
                }
                brep.Faces[newF] = kf;
            }

            // Back-populate vertex Edges
            for (int ei = 0; ei < brep.Edges.Count; ei++)
            {
                var e = brep.Edges[ei];
                if (e.V0 >= 0 && e.V0 < brep.Vertices.Count)
                    brep.Vertices[e.V0].Edges.Add(ei);
                if (e.V1 >= 0 && e.V1 < brep.Vertices.Count && e.V1 != e.V0)
                    brep.Vertices[e.V1].Edges.Add(ei);
            }

            return brep;
        }

        #endregion

        #region Geometry Helpers

        /// <summary>
        /// Finds the closest point on a NURBS curve segment to a given 3D point.
        /// </summary>
        public static bool ClosestPointOnCurve(
            NurbsCurve c,
            Interval domain,
            V3 pt,
            out double bestT,
            out double bestDist)
        {
            if (c == null)
            {
                bestT = 0;
                bestDist = double.PositiveInfinity;
                return false;
            }

            if (c.Degree == 1)
            {
                var p0 = NurbsEval.CurvePoint(c, domain.T0);
                var p1 = NurbsEval.CurvePoint(c, domain.T1);
                var v = p1 - p0;
                double l2 = v.X * v.X + v.Y * v.Y + v.Z * v.Z;

                if (l2 < 1e-24)
                {
                    bestT = domain.T0;
                    bestDist = (pt - p0).Length;
                    return true;
                }

                var w = pt - p0;
                double u = (w.X * v.X + w.Y * v.Y + w.Z * v.Z) / l2;
                u = Math.Max(0.0, Math.Min(1.0, u));
                bestT = domain.T0 + u * domain.Length;
                var proj = new V3(p0.X + v.X * u, p0.Y + v.Y * u, p0.Z + v.Z * u);
                bestDist = (pt - proj).Length;
                return true;
            }

            // For higher-degree curves: sample + Golden Section Search
            const int sampleCount = 32;
            int bestK = 0;
            double minDistSq = double.PositiveInfinity;
            double step = domain.Length / sampleCount;

            for (int k = 0; k <= sampleCount; k++)
            {
                double t = domain.T0 + k * step;
                var p = NurbsEval.CurvePoint(c, t);
                double d2 = DistSq(pt, p);
                if (d2 < minDistSq)
                {
                    minDistSq = d2;
                    bestK = k;
                }
            }

            double ta = Math.Max(domain.T0, domain.T0 + (bestK - 1) * step);
            double tb = Math.Min(domain.T1, domain.T0 + (bestK + 1) * step);

            const double goldenRatio = 0.618033988749895;
            for (int iter = 0; iter < 20; iter++)
            {
                double c1 = tb - goldenRatio * (tb - ta);
                double c2 = ta + goldenRatio * (tb - ta);

                var pC1 = NurbsEval.CurvePoint(c, c1);
                var pC2 = NurbsEval.CurvePoint(c, c2);

                if (DistSq(pt, pC1) < DistSq(pt, pC2))
                    tb = c2;
                else
                    ta = c1;
            }

            bestT = 0.5 * (ta + tb);
            bestDist = (pt - NurbsEval.CurvePoint(c, bestT)).Length;
            return true;
        }

        private static double DistSq(V3 a, V3 b)
        {
            double dx = a.X - b.X;
            double dy = a.Y - b.Y;
            double dz = a.Z - b.Z;
            return dx * dx + dy * dy + dz * dz;
        }

        /// <summary>
        /// Calculates the number of naked edges on a KBrep (edges with valence 1).
        /// </summary>
        public static int GetNakedEdgeCount(KBrep brep)
        {
            if (brep == null || brep.Edges == null) return 0;
            int count = 0;
            foreach (var e in brep.Edges)
            {
                if (e.Trims.Count == 1) count++;
            }
            return count;
        }

        /// <summary>
        /// Determines whether a KBrep represents a closed solid (no naked edges and at least one face).
        /// </summary>
        public static bool IsSolid(KBrep brep)
        {
            if (brep == null || brep.Faces.Count < 1) return false;
            return GetNakedEdgeCount(brep) == 0;
        }

        #endregion
    }
}
