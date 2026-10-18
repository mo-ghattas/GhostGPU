using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Rhino;
using Rhino.Geometry;
using Rhino.Geometry.Intersect;
using GhostGPU.Core;
using GhostGPU.Runtime;
using GhostGPU.Booleans;
using K = GhostGPU.Kernel;

namespace GhostGPU.Grasshopper.BrepPipeline
{
    /// <summary>
    /// Operand mesh cache that ensures each input Brep is tessellated and uploaded to GPU exactly once
    /// during multi-operand boolean evaluations, avoiding redundant GPU transfers.
    /// </summary>
    public sealed class OperandMeshCache : IDisposable
    {
        private readonly GpuSession _session;
        private readonly Dictionary<Brep, GpuTriMesh> _meshes = new Dictionary<Brep, GpuTriMesh>();

        public OperandMeshCache(GpuSession session)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
        }

        public GpuTriMesh GetOrUpload(Brep b)
        {
            if (b == null) return null;
            if (_meshes.TryGetValue(b, out var existing)) return existing;

            var triMesh = BrepBooleanPipeline.TessellateToTriMesh(b);
            if (triMesh == null) return null;

            var gpuMesh = GpuTriMesh.Upload(_session, triMesh, ElementType.V3D);
            _meshes[b] = gpuMesh;
            return gpuMesh;
        }

        public void Dispose()
        {
            foreach (var m in _meshes.Values)
            {
                try { m?.Dispose(); } catch { }
            }
            _meshes.Clear();
        }
    }

    /// <summary>Which exact kernel runs stages 2-3 (SSI + split) and 5 (join) of the hybrid pipeline.</summary>
    public enum BrepKernel
    {
        /// <summary>RhinoCommon Intersection.BrepBrep / Brep.Split / Brep.JoinBreps (default).</summary>
        RhinoCommon,
        /// <summary>GhostGPU.Kernel Ssi.Intersect / FaceSplit.Split / BrepJoin.Join; falls back to RhinoCommon per pair.</summary>
        GhostGPU
    }

    /// <summary>
    /// Hybrid analytic Brep boolean pipeline implementing the 5-stage algorithm from contract section 6:
    /// Stage 0: Prep (SplitKinkyFaces with clamped angle tolerance, SolidOrientation inward Flip).
    /// Stage 1: GPU Face-AABB broadphase via GpuGeometry.OverlappingBoxPairs with disjoint short-circuit.
    /// Stage 2-3: Exact surface-surface intersection and split on RhinoCommon.
    /// Stage 4: GPU winding number classification of interior sample points via GpuGeometry.WindingNumbers,
    ///          applying rule table (Inside = |W| >= 0.5, flipping B for Difference).
    /// Stage 5: B-Rep topological sewing via Brep.JoinBreps + IsValid/IsSolid check,
    ///          falling back with visible warning to native Brep.CreateBooleanX.
    /// With <see cref="BrepKernel.GhostGPU"/>, stages 2-3 are Ssi.Intersect + FaceSplit.Split on KBrep copies of the
    /// operands, stage 4 classifies kernel pieces by a UV-sampled interior point and exact Brep.IsPointInside, and
    /// stage 5 is BrepJoin.Join + KernelBrepAdapter.ToRhino; any kernel exception or invalid result re-runs that pair
    /// on the RhinoCommon path with a visible warning. Per-stage times: <see cref="LastTimings"/>.
    /// </summary>
    public static class BrepBooleanPipeline
    {
        /// <summary>Parses the component "Kernel" input: "RhinoCommon" or "GhostGPU" (case-insensitive, trimmed).</summary>
        public static bool TryParseKernel(string text, out BrepKernel kernel)
        {
            kernel = BrepKernel.RhinoCommon;
            string t = text?.Trim();
            if (string.Equals(t, nameof(BrepKernel.RhinoCommon), StringComparison.OrdinalIgnoreCase)) return true;
            if (string.Equals(t, nameof(BrepKernel.GhostGPU), StringComparison.OrdinalIgnoreCase)) { kernel = BrepKernel.GhostGPU; return true; }
            return false;
        }

        /// <summary>
        /// Prepares Brep input matching native EnsureCorrectBrep:
        /// Duplicates, splits kinky faces with clamped document angle tolerance, and flips if solid orientation is Inward.
        /// </summary>
        public static Brep EnsureCorrectBrep(Brep input, double angleTolerance)
        {
            if (input == null) return null;
            Brep brep = input.DuplicateBrep();
            double clampAngle = RhinoMath.Clamp(angleTolerance, RhinoMath.ToRadians(0.1), RhinoMath.ToRadians(2.0));
            brep.Faces.SplitKinkyFaces(clampAngle, compact: true);
            if (brep.SolidOrientation == BrepSolidOrientation.Inward)
            {
                brep.Flip();
            }
            return brep;
        }

        /// <summary>
        /// Tessellates a Brep into a TriMesh suitable for GpuTriMesh upload.
        /// </summary>
        public static TriMesh TessellateToTriMesh(Brep brep)
        {
            if (brep == null) return null;
            var mp = MeshingParameters.Default;
            var meshes = Rhino.Geometry.Mesh.CreateFromBrep(brep, mp);
            if (meshes == null || meshes.Length == 0)
            {
                mp = MeshingParameters.FastRenderMesh;
                meshes = Rhino.Geometry.Mesh.CreateFromBrep(brep, mp);
            }
            if (meshes == null || meshes.Length == 0) return null;

            var combined = new Rhino.Geometry.Mesh();
            foreach (var m in meshes)
            {
                if (m != null) combined.Append(m);
            }
            combined.Faces.ConvertQuadsToTriangles();
            combined.Compact();

            int vCount = combined.Vertices.Count;
            int fCount = combined.Faces.Count;
            var vertices = new V3[vCount];
            for (int i = 0; i < vCount; i++)
            {
                var pt = combined.Vertices[i];
                vertices[i] = new V3(pt.X, pt.Y, pt.Z);
            }

            var triangles = new int[fCount * 3];
            for (int i = 0; i < fCount; i++)
            {
                var f = combined.Faces[i];
                triangles[i * 3] = f.A;
                triangles[i * 3 + 1] = f.B;
                triangles[i * 3 + 2] = f.C;
            }

            return new TriMesh
            {
                Vertices = vertices,
                Triangles = triangles
            };
        }

        /// <summary>
        /// Picks a classification sample point on a split Brep piece, as FAR as possible from the face's
        /// edges (trims, seams, and the intersection curves that bound the piece).
        /// Why: the GPU winding number classifies against a TESSELLATED copy of the other operand. A point
        /// within chord-sag distance of the intersection curve can sit between the true surface and its
        /// inscribed mesh and get the wrong side (a cylinder/sphere cap disk sampled at r=2.989 for a
        /// r=3 cut returned W=0 instead of 1, which kept a wrong piece and broke JoinBreps).
        /// Candidates: centroids of a per-face render mesh plus a 15x15 parameter grid, filtered to
        /// IsPointOnFace == Interior; score = min distance to the face's adjacent edges; keep the max.
        /// Returns Point3d.Unset if no interior point exists.
        /// </summary>
        public static Point3d SampleInteriorPoint(Brep piece)
        {
            return SampleInteriorPoint(piece, out _);
        }

        /// <summary>As <see cref="SampleInteriorPoint(Brep)"/>, also returning the point's distance to the nearest face edge.</summary>
        public static Point3d SampleInteriorPoint(Brep piece, out double clearance)
        {
            clearance = -1.0;
            if (piece == null || piece.Faces.Count == 0) return Point3d.Unset;

            Point3d best = Point3d.Unset;
            // Limitation: early exit once clearance >= 10% of the piece diagonal; no explicit bound against the
            // classifier mesh's chord sag. Upgrade: compare against the tessellation's max edge-to-surface distance.
            double goodEnough = 0.1 * piece.GetBoundingBox(false).Diagonal.Length;

            foreach (var face in piece.Faces)
            {
                var edges = face.AdjacentEdges().Select(ei => piece.Edges[ei]).ToArray();
                var candidates = new List<Point2d>();

                Rhino.Geometry.Mesh[] meshes = null;
                try { meshes = Rhino.Geometry.Mesh.CreateFromBrep(face.DuplicateFace(false), MeshingParameters.FastRenderMesh); }
                catch { /* grid candidates below still apply */ }
                if (meshes != null)
                {
                    foreach (var m in meshes)
                    {
                        if (m == null) continue;
                        for (int fi = 0; fi < m.Faces.Count; fi++)
                        {
                            var f = m.Faces[fi];
                            Point3d centroid = ((Point3d)m.Vertices[f.A] + (Point3d)m.Vertices[f.B] + (Point3d)m.Vertices[f.C]) / 3.0;
                            if (face.ClosestPoint(centroid, out double u, out double v)) candidates.Add(new Point2d(u, v));
                        }
                    }
                }

                Interval du = face.Domain(0), dv = face.Domain(1);
                for (int i = 1; i < 16; i++)
                    for (int j = 1; j < 16; j++)
                        candidates.Add(new Point2d(du.ParameterAt(i / 16.0), dv.ParameterAt(j / 16.0)));

                foreach (var uv in candidates)
                {
                    if (face.IsPointOnFace(uv.X, uv.Y) != PointFaceRelation.Interior) continue;
                    Point3d p = face.PointAt(uv.X, uv.Y);
                    double clear = double.MaxValue;
                    foreach (var e in edges)
                    {
                        if (e.ClosestPoint(p, out double t)) clear = Math.Min(clear, p.DistanceTo(e.PointAt(t)));
                    }
                    if (clear > clearance)
                    {
                        clearance = clear;
                        best = p;
                    }
                }

                if (clearance >= goodEnough) break;
            }

            return best;
        }

        /// <summary>
        /// Stage 1 GPU Broadphase: evaluates face-AABB pairs on GPU using GpuGeometry.OverlappingBoxPairs.
        /// Explicitly passes ElementType.V3D and Residency.Device to GhostGPU.Booleans.
        /// </summary>
        public static bool CheckFaceAabbOverlap(GpuSession session, Brep a, Brep b, double tolerance)
        {
            int na = a.Faces.Count;
            int nb = b.Faces.Count;
            if (na == 0 || nb == 0) return false;

            var minA = new V3[na];
            var maxA = new V3[na];
            for (int i = 0; i < na; i++)
            {
                var box = a.Faces[i].GetBoundingBox(true);
                minA[i] = new V3(box.Min.X, box.Min.Y, box.Min.Z);
                maxA[i] = new V3(box.Max.X, box.Max.Y, box.Max.Z);
            }

            var minB = new V3[nb];
            var maxB = new V3[nb];
            for (int j = 0; j < nb; j++)
            {
                var box = b.Faces[j].GetBoundingBox(true);
                minB[j] = new V3(box.Min.X, box.Min.Y, box.Min.Z);
                maxB[j] = new V3(box.Max.X, box.Max.Y, box.Max.Z);
            }

            using var minABuf = session.UploadPoints(minA, ElementType.V3D, Residency.Device);
            using var maxABuf = session.UploadPoints(maxA, ElementType.V3D, Residency.Device);
            using var minBBuf = session.UploadPoints(minB, ElementType.V3D, Residency.Device);
            using var maxBBuf = session.UploadPoints(maxB, ElementType.V3D, Residency.Device);

            using var pairsBuf = GpuGeometry.OverlappingBoxPairs(session, minABuf, maxABuf, minBBuf, maxBBuf, tolerance);
            int[] pairs = session.DownloadIndices(pairsBuf);
            return pairs != null && pairs.Length > 0;
        }

        /// <summary>
        /// Stage 2-3 Exact Splitting on RhinoCommon: splits target Brep with cutter Brep,
        /// falling back to intersection curves if Brep.Split(Brep) yields single piece.
        /// </summary>
        public static Brep[] SplitBrep(Brep target, Brep cutter, double tolerance)
        {
            var pieces = target.Split(cutter, tolerance);
            if (pieces != null && pieces.Length > 1)
            {
                return pieces;
            }

            if (Intersection.BrepBrep(target, cutter, tolerance, out Curve[] curves, out _))
            {
                if (curves != null && curves.Length > 0)
                {
                    var curvePieces = target.Split(curves, tolerance);
                    if (curvePieces != null && curvePieces.Length > 1)
                    {
                        return curvePieces;
                    }
                }
            }

            return pieces != null && pieces.Length > 0 ? pieces : new Brep[] { target };
        }

        /// <summary>
        /// Checks if two bounding boxes overlap within the given tolerance.
        /// </summary>
        public static bool BoundingBoxesOverlap(BoundingBox a, BoundingBox b, double tol)
        {
            return (a.Min.X - tol <= b.Max.X && a.Max.X + tol >= b.Min.X) &&
                   (a.Min.Y - tol <= b.Max.Y && a.Max.Y + tol >= b.Min.Y) &&
                   (a.Min.Z - tol <= b.Max.Z && a.Max.Z + tol >= b.Min.Z);
        }

        /// <summary>
        /// Computes a boolean operation between a single pair of Breps through the 5-stage pipeline.
        /// Wraps GPU stages in try/catch to gracefully fall back on budget or driver exceptions.
        /// </summary>
        public static Brep[] ComputePair(
            GpuSession session,
            Brep a,
            Brep b,
            BooleanOperation op,
            double tolerance,
            Action<string> onWarning = null,
            OperandMeshCache meshCache = null,
            BrepKernel kernel = BrepKernel.RhinoCommon)
        {
            if (a == null && b == null) return new Brep[0];
            if (a == null) return op == BooleanOperation.Union ? new Brep[] { b } : new Brep[0];
            if (b == null) return op == BooleanOperation.Difference || op == BooleanOperation.Union ? new Brep[] { a } : new Brep[0];

            // Stage 1: Global Bounding Box check
            var boxA = a.GetBoundingBox(true);
            var boxB = b.GetBoundingBox(true);
            if (!BoundingBoxesOverlap(boxA, boxB, tolerance))
            {
                switch (op)
                {
                    case BooleanOperation.Difference:
                        return new Brep[] { a };
                    case BooleanOperation.Union:
                        return new Brep[] { a, b };
                    case BooleanOperation.Intersection:
                        return new Brep[0];
                }
            }

            bool useKernel = kernel == BrepKernel.GhostGPU;
            // Pipeline failures: kernel mode re-runs the pair on the RhinoCommon path, RhinoCommon mode goes native.
            Func<string, Brep[]> fail = reason => useKernel
                ? KernelFallback(session, a, b, op, tolerance, onWarning, meshCache, reason)
                : FallbackNative(a, b, op, tolerance, onWarning, reason);

            var sw = Stopwatch.StartNew();
            try
            {
                // Stage 1: GPU Face-AABB broadphase
                bool faceOverlap = CheckFaceAabbOverlap(session, a, b, tolerance);
                if (!faceOverlap)
                {
                    var ptB = SampleInteriorPoint(b);
                    var ptA = SampleInteriorPoint(a);
                    bool bInA = ptB != Point3d.Unset && a.IsPointInside(ptB, tolerance, strictlyIn: true);
                    bool aInB = ptA != Point3d.Unset && b.IsPointInside(ptA, tolerance, strictlyIn: true);

                    if (!bInA && !aInB)
                    {
                        // Completely disjoint
                        switch (op)
                        {
                            case BooleanOperation.Difference:
                                return new Brep[] { a };
                            case BooleanOperation.Union:
                                return new Brep[] { a, b };
                            case BooleanOperation.Intersection:
                                return new Brep[0];
                        }
                    }
                }

                Lap(sw, "broadphase");

                // Stages 2-5 on the GhostGPU kernel: no Rhino piece conversion, meshing or GPU winding
                if (useKernel) return KernelPair(session, a, b, op, tolerance, faceOverlap, onWarning, fail);

                // Stage 2-3: exact intersection + split of touched breps on RhinoCommon
                Brep[] piecesA = SplitBrep(a, b, tolerance);
                Brep[] piecesB = SplitBrep(b, a, tolerance);
                Lap(sw, "split (RhinoCommon)");

                // Handle containment without fallback warning when neither operand splits
                if (piecesA.Length <= 1 && piecesB.Length <= 1)
                    return Containment(a, b, op, tolerance, faceOverlap, onWarning, fail);

                // Stage 4: GPU winding number classification
                GpuTriMesh gpuMeshA = null;
                GpuTriMesh gpuMeshB = null;
                bool disposeLocalA = false;
                bool disposeLocalB = false;

                if (meshCache != null)
                {
                    gpuMeshA = meshCache.GetOrUpload(a);
                    gpuMeshB = meshCache.GetOrUpload(b);
                }
                else
                {
                    var triMeshA = TessellateToTriMesh(a);
                    var triMeshB = TessellateToTriMesh(b);
                    if (triMeshA == null || triMeshB == null)
                    {
                        return fail("mesh tessellation for GPU winding failed");
                    }
                    gpuMeshA = GpuTriMesh.Upload(session, triMeshA, ElementType.V3D);
                    gpuMeshB = GpuTriMesh.Upload(session, triMeshB, ElementType.V3D);
                    disposeLocalA = true;
                    disposeLocalB = true;
                }

                try
                {
                    if (gpuMeshA == null || gpuMeshB == null)
                    {
                        return fail("operand mesh upload failed");
                    }
                    Lap(sw, "mesh+upload");

                    var samplePtsA = new V3[piecesA.Length];
                    for (int i = 0; i < piecesA.Length; i++)
                    {
                        var pt = SampleInteriorPoint(piecesA[i]);
                        if (pt == Point3d.Unset)
                        {
                            return fail("could not determine interior sample point on split piece of A");
                        }
                        samplePtsA[i] = new V3(pt.X, pt.Y, pt.Z);
                    }

                    var samplePtsB = new V3[piecesB.Length];
                    for (int j = 0; j < piecesB.Length; j++)
                    {
                        var pt = SampleInteriorPoint(piecesB[j]);
                        if (pt == Point3d.Unset)
                        {
                            return fail("could not determine interior sample point on split piece of B");
                        }
                        samplePtsB[j] = new V3(pt.X, pt.Y, pt.Z);
                    }
                    Lap(sw, "sample points");

                    using var ptsABuf = session.UploadPoints(samplePtsA, ElementType.V3D, Residency.Device);
                    using var ptsBBuf = session.UploadPoints(samplePtsB, ElementType.V3D, Residency.Device);

                    using var wABuf = GpuGeometry.WindingNumbers(session, ptsABuf, gpuMeshB);
                    using var wBBuf = GpuGeometry.WindingNumbers(session, ptsBBuf, gpuMeshA);

                    double[] wA = session.DownloadScalars(wABuf); // Winding of pieces of A vs solid B
                    double[] wB = session.DownloadScalars(wBBuf); // Winding of pieces of B vs solid A
                    Lap(sw, "winding (GPU)");

                    var retainedPieces = new List<Brep>();

                    // Rule table: Inside = |W| >= 0.5
                    // Union: keep A outside B (|W_B| < 0.5) + B outside A (|W_A| < 0.5)
                    // Intersection: keep A inside B (|W_B| >= 0.5) + B inside A (|W_A| >= 0.5)
                    // Difference: keep A outside B (|W_B| < 0.5) + B inside A (|W_A| >= 0.5) FLIPPED
                    switch (op)
                    {
                        case BooleanOperation.Difference:
                            for (int i = 0; i < piecesA.Length; i++)
                            {
                                if (Math.Abs(wA[i]) < 0.5)
                                {
                                    retainedPieces.Add(piecesA[i]);
                                }
                            }
                            for (int j = 0; j < piecesB.Length; j++)
                            {
                                if (Math.Abs(wB[j]) >= 0.5)
                                {
                                    var flippedPiece = piecesB[j].DuplicateBrep();
                                    flippedPiece.Flip();
                                    retainedPieces.Add(flippedPiece);
                                }
                            }
                            break;

                        case BooleanOperation.Union:
                            for (int i = 0; i < piecesA.Length; i++)
                            {
                                if (Math.Abs(wA[i]) < 0.5)
                                {
                                    retainedPieces.Add(piecesA[i]);
                                }
                            }
                            for (int j = 0; j < piecesB.Length; j++)
                            {
                                if (Math.Abs(wB[j]) < 0.5)
                                {
                                    retainedPieces.Add(piecesB[j]);
                                }
                            }
                            break;

                        case BooleanOperation.Intersection:
                            for (int i = 0; i < piecesA.Length; i++)
                            {
                                if (Math.Abs(wA[i]) >= 0.5)
                                {
                                    retainedPieces.Add(piecesA[i]);
                                }
                            }
                            for (int j = 0; j < piecesB.Length; j++)
                            {
                                if (Math.Abs(wB[j]) >= 0.5)
                                {
                                    retainedPieces.Add(piecesB[j]);
                                }
                            }
                            break;
                    }

                    if (retainedPieces.Count == 0)
                    {
                        return new Brep[0];
                    }

                    // Stage 5: JoinBreps + IsValid/IsSolid
                    Brep[] joined = Brep.JoinBreps(retainedPieces, tolerance);
                    Lap(sw, "join (RhinoCommon)");
                    bool valid = joined != null && joined.Length > 0 && Array.TrueForAll(joined, x => x != null && x.IsValid && x.IsSolid);
                    if (valid)
                    {
                        return joined;
                    }

                    return fail("joined result was not a valid solid");
                }
                finally
                {
                    if (disposeLocalA) gpuMeshA?.Dispose();
                    if (disposeLocalB) gpuMeshB?.Dispose();
                }
            }
            catch (Exception ex)
            {
                return fail(useKernel ? $"GPU/kernel stage failed ({ex.GetType().Name}: {ex.Message})" : $"GPU stage failed ({ex.Message})");
            }
        }

        /// <summary>Neither operand split: containment / disjoint rules (shared by both kernels).</summary>
        private static Brep[] Containment(Brep a, Brep b, BooleanOperation op, double tolerance, bool faceOverlap,
                                          Action<string> onWarning, Func<string, Brep[]> fail)
        {
            var ptB = SampleInteriorPoint(b);
            var ptA = SampleInteriorPoint(a);
            bool bInA = ptB != Point3d.Unset && a.IsPointInside(ptB, tolerance, strictlyIn: true);
            bool aInB = ptA != Point3d.Unset && b.IsPointInside(ptA, tolerance, strictlyIn: true);

            if (bInA)
            {
                // B is fully inside A
                switch (op)
                {
                    case BooleanOperation.Union:
                        return new Brep[] { a };
                    case BooleanOperation.Intersection:
                        return new Brep[] { b };
                    case BooleanOperation.Difference:
                        return FallbackNative(a, b, op, tolerance, onWarning, "operand B is contained inside A; carved via native Brep.CreateBooleanDifference");
                }
            }

            if (aInB)
            {
                // A is fully inside B
                switch (op)
                {
                    case BooleanOperation.Union:
                        return new Brep[] { b };
                    case BooleanOperation.Intersection:
                        return new Brep[] { a };
                    case BooleanOperation.Difference:
                        return new Brep[0];
                }
            }

            if (!faceOverlap || (!bInA && !aInB))
            {
                switch (op)
                {
                    case BooleanOperation.Difference:
                        return new Brep[] { a };
                    case BooleanOperation.Union:
                        return new Brep[] { a, b };
                    case BooleanOperation.Intersection:
                        return new Brep[0];
                }
            }

            return fail("sub-face split yielded no multiple segments");
        }

        /// <summary>
        /// Kernel stages 2-5: KBrep copies of both operands, Ssi.Intersect (GPU session), FaceSplit.Split of each side,
        /// classification of each kernel piece by one UV-sampled interior point tested with exact Brep.IsPointInside
        /// against the other operand (no piece -> Rhino conversion, no meshing, no GPU winding: the winding kernel costs
        /// 15-400 ms per call against an operand mesh for a handful of points), flip of kept B pieces for Difference,
        /// BrepJoin.Join, ToRhino + IsValid/IsSolid.
        /// </summary>
        private static Brep[] KernelPair(GpuSession session, Brep a, Brep b, BooleanOperation op, double tolerance,
                                         bool faceOverlap, Action<string> onWarning, Func<string, Brep[]> fail)
        {
            var sw = Stopwatch.StartNew();
            var ka = KernelBrepAdapter.ToKernel(a);
            var kb = KernelBrepAdapter.ToKernel(b);
            Lap(sw, "kernel ToKernel");
            var ssi = K.Ssi.Intersect(ka, kb, tolerance, session);
            Lap(sw, "kernel SSI");
            bool none = ssi.Branches.Count == 0;
            var piecesA = none ? new List<K.KBrep> { ka } : K.FaceSplit.Split(ka, ssi.Branches, true, tolerance);
            var piecesB = none ? new List<K.KBrep> { kb } : K.FaceSplit.Split(kb, ssi.Branches, false, tolerance);
            Lap(sw, "kernel split A+B");
            if (piecesA == null || piecesA.Count == 0 || piecesB == null || piecesB.Count == 0) return fail("FaceSplit returned no pieces");
            if (piecesA.Count <= 1 && piecesB.Count <= 1) return Containment(a, b, op, tolerance, faceOverlap, onWarning, fail);

            // Rule table: Union keeps A outside B + B outside A; Intersection keeps both inside;
            // Difference keeps A outside B + B inside A, flipped.
            var kept = new List<K.KBrep>();
            string why = Classify(piecesA, b, op == BooleanOperation.Intersection, tolerance, kept, "A");
            int firstB = kept.Count;
            why = why ?? Classify(piecesB, a, op != BooleanOperation.Union, tolerance, kept, "B");
            Lap(sw, "kernel classify");
            if (why != null) return fail(why);
            // Brep.Flip semantics (verified live): toggle m_bRev only; loops and trims stay. Pieces are fresh, so in place.
            if (op == BooleanOperation.Difference)
                for (int i = firstB; i < kept.Count; i++)
                    foreach (var f in kept[i].Faces) f.Reversed = !f.Reversed;
            if (kept.Count == 0) return new Brep[0];

            var kJoined = K.BrepJoin.Join(kept, tolerance);
            Lap(sw, "kernel join");
            var joined = new Brep[kJoined.Count];
            for (int i = 0; i < kJoined.Count; i++)
            {
                joined[i] = KernelBrepAdapter.ToRhino(kJoined[i], out string log);
                if (log != null) { Lap(sw, "kernel ToRhino"); return fail($"kernel join result {i} is invalid: {FirstLine(log)}"); }
            }
            bool valid = joined.Length > 0 && Array.TrueForAll(joined, x => x != null && x.IsSolid);
            Lap(sw, "kernel ToRhino");
            if (valid) return joined;
            string naked = string.Join(",", joined.Select(x => x?.Edges.Count(e => e.Valence == EdgeAdjacency.Naked) ?? -1));
            return fail($"joined result was not a valid solid ({joined.Length} breps, naked edges {naked})");
        }

        // Returns null on success, else the failure reason.
        private static string Classify(List<K.KBrep> pieces, Brep other, bool keepInside, double tolerance, List<K.KBrep> kept, string side)
        {
            for (int i = 0; i < pieces.Count; i++)
            {
                var p = KernelSamplePoint(pieces[i], tolerance);
                if (p == Point3d.Unset) return $"could not determine interior sample point on kernel piece {i} of {side}";
                if (other.IsPointInside(p, tolerance, strictlyIn: false) == keepInside) kept.Add(pieces[i]);
            }
            return null;
        }

        /// <summary>
        /// Interior point of a kernel piece without converting or meshing it: per face, the trim loops are sampled to UV
        /// polygons (and their 3D images), a 15x15 grid over the loops' UV box is filtered by even-odd containment, and
        /// the candidate farthest (in 3D) from the face boundary wins. Returns Point3d.Unset if no candidate clears tol.
        /// </summary>
        public static Point3d KernelSamplePoint(K.KBrep piece, double tolerance)
        {
            const int perTrim = 32, grid = 16;
            Point3d best = Point3d.Unset;
            double bestClear = tolerance;
            foreach (var f in piece.Faces)
            {
                var srf = piece.Surfaces[f.Surface];
                var polys = new List<V3[]>();   // UV polygons (closed implicitly)
                var rims = new List<V3[]>();    // their 3D images
                double u0 = double.MaxValue, u1 = double.MinValue, v0 = double.MaxValue, v1 = double.MinValue;
                foreach (int li in f.Loops)
                {
                    var uv = new List<V3>();
                    foreach (int ti in piece.Loops[li].Trims)
                    {
                        var t = piece.Trims[ti];
                        var c = piece.Curves2[t.Curve2];
                        // a UV line on a bilinear (e.g. planar) surface maps to a 3D line: its end points suffice
                        int n = c.Degree == 1 && c.CvCount == 2 && srf.DegreeU == 1 && srf.DegreeV == 1 ? 1 : perTrim;
                        for (int k = 0; k < n; k++) uv.Add(K.NurbsEval.CurvePoint(c, t.Domain.ParameterAt(k / (double)n)));
                    }
                    foreach (var p in uv) { u0 = Math.Min(u0, p.X); u1 = Math.Max(u1, p.X); v0 = Math.Min(v0, p.Y); v1 = Math.Max(v1, p.Y); }
                    polys.Add(uv.ToArray());
                    rims.Add(uv.Select(p => K.NurbsEval.SurfacePoint(srf, p.X, p.Y)).ToArray());
                }
                if (polys.Count == 0) continue;

                for (int i = 1; i < grid; i++)
                    for (int j = 1; j < grid; j++)
                    {
                        double u = u0 + (u1 - u0) * i / grid, v = v0 + (v1 - v0) * j / grid;
                        if (!InsideEvenOdd(polys, u, v)) continue;
                        var x = K.NurbsEval.SurfacePoint(srf, u, v);
                        double clear = double.MaxValue;
                        foreach (var rim in rims)
                            for (int k = 0; k < rim.Length; k++) clear = Math.Min(clear, SegmentDistance(x, rim[k], rim[(k + 1) % rim.Length]));
                        if (clear > bestClear) { bestClear = clear; best = new Point3d(x.X, x.Y, x.Z); }
                    }

                // Early exit (the classifier is exact, so clearance only has to beat tolerance comfortably):
                // 10% of this face's rim box diagonal.
                var lo = rims.SelectMany(r => r).Aggregate((p, q) => new V3(Math.Min(p.X, q.X), Math.Min(p.Y, q.Y), Math.Min(p.Z, q.Z)));
                var hi = rims.SelectMany(r => r).Aggregate((p, q) => new V3(Math.Max(p.X, q.X), Math.Max(p.Y, q.Y), Math.Max(p.Z, q.Z)));
                if (best != Point3d.Unset && bestClear >= 0.1 * (hi - lo).Length) break;
            }
            return best;
        }

        private static bool InsideEvenOdd(List<V3[]> polys, double u, double v)
        {
            bool inside = false;
            foreach (var poly in polys)
                for (int i = 0, j = poly.Length - 1; i < poly.Length; j = i++)
                {
                    V3 a = poly[i], b = poly[j];
                    if ((a.Y > v) != (b.Y > v) && u < (b.X - a.X) * (v - a.Y) / (b.Y - a.Y) + a.X) inside = !inside;
                }
            return inside;
        }

        private static double SegmentDistance(V3 p, V3 a, V3 b)
        {
            var ab = b - a;
            double l2 = V3.Dot(ab, ab);
            double s = l2 > 0 ? Math.Max(0, Math.Min(1, V3.Dot(p - a, ab) / l2)) : 0;
            return (p - (a + ab * s)).Length;
        }

        // ---- per-stage timing (accumulated over all pairs since the last ResetTimings) ----
        // Limitation: one static accumulator, not thread-safe across concurrent solves; fine for GH's serial solver.
        private static readonly List<KeyValuePair<string, double>> _timings = new List<KeyValuePair<string, double>>();

        /// <summary>Clears the stage timings; called at the start of BooleanDifference/Intersection/Union.</summary>
        public static void ResetTimings() { lock (_timings) _timings.Clear(); }

        /// <summary>Per-stage milliseconds accumulated since the last <see cref="ResetTimings"/> (the last set-level run).</summary>
        public static string LastTimings
        {
            get
            {
                lock (_timings)
                    return _timings.Count == 0 ? "(none)"
                        : string.Join(", ", _timings.Select(kv => $"{kv.Key} {kv.Value:F1}")) + $" | total {_timings.Sum(kv => kv.Value):F1} ms";
            }
        }

        private static void Lap(Stopwatch sw, string stage)
        {
            double ms = sw.Elapsed.TotalMilliseconds;
            lock (_timings)
            {
                int i = _timings.FindIndex(kv => kv.Key == stage);
                if (i < 0) _timings.Add(new KeyValuePair<string, double>(stage, ms));
                else _timings[i] = new KeyValuePair<string, double>(stage, _timings[i].Value + ms);
            }
            sw.Restart();
        }

        private static string FirstLine(string log)
        {
            string t = (log ?? "").Trim();
            int nl = t.IndexOf('\n');
            return nl < 0 ? t : t.Substring(0, nl).Trim();
        }

        private static Brep[] KernelFallback(
            GpuSession session,
            Brep a,
            Brep b,
            BooleanOperation op,
            double tolerance,
            Action<string> onWarning,
            OperandMeshCache meshCache,
            string reason)
        {
            onWarning?.Invoke($"[GhostGPU] GhostGPU kernel {op} failed ({reason}); falling back to the RhinoCommon split/join path for this pair.");
            return ComputePair(session, a, b, op, tolerance, onWarning, meshCache, BrepKernel.RhinoCommon);
        }

        private static Brep[] FallbackNative(
            Brep a,
            Brep b,
            BooleanOperation op,
            double tolerance,
            Action<string> onWarning,
            string reason)
        {
            string opName = op.ToString();
            onWarning?.Invoke($"[GhostGPU] Hybrid Brep boolean {opName} stage failed ({reason}); falling back to native Brep.CreateBoolean{opName}.");

            switch (op)
            {
                case BooleanOperation.Difference:
                    return Brep.CreateBooleanDifference(new Brep[] { a }, new Brep[] { b }, tolerance) ?? new Brep[0];
                case BooleanOperation.Union:
                    return Brep.CreateBooleanUnion(new Brep[] { a, b }, tolerance) ?? new Brep[0];
                case BooleanOperation.Intersection:
                    return Brep.CreateBooleanIntersection(new Brep[] { a }, new Brep[] { b }, tolerance) ?? new Brep[0];
                default:
                    return new Brep[0];
            }
        }

        /// <summary>
        /// Executes Boolean Difference across sets of Breps matching native SolidDifferenceManager:
        /// For each cutter in B, subtracts from the running A set.
        /// Caches operand meshes to avoid re-tessellating across cutters.
        /// </summary>
        public static List<Brep> BooleanDifference(
            GpuSession session,
            IEnumerable<Brep> setA,
            IEnumerable<Brep> setB,
            double tolerance,
            Action<string> onWarning = null,
            BrepKernel kernel = BrepKernel.RhinoCommon)
        {
            ResetTimings();
            var current = new List<Brep>(setA.Where(b => b != null));
            using var cache = new OperandMeshCache(session);

            foreach (var cutter in setB)
            {
                if (cutter == null) continue;
                var next = new List<Brep>();
                if (!cutter.IsSolid)
                {
                    // Native CreateBooleanDifference fails on an open cutter and SolidDifferenceManager keeps A.
                    onWarning?.Invoke("[GhostGPU] Cutter is not a closed solid; skipped (A kept unchanged, as native).");
                    continue;
                }
                foreach (var brep in current)
                {
                    if (brep == null) continue;
                    var diff = ComputePair(session, brep, cutter, BooleanOperation.Difference, tolerance, onWarning, cache, kernel);
                    // Native SolidDifferenceManager keeps the brep when the boolean returns null/empty.
                    if (diff != null && diff.Length > 0)
                        next.AddRange(diff);
                    else
                        next.Add(brep);
                }
                current = next;
            }
            return current;
        }

        /// <summary>
        /// Executes Boolean Intersection across sets of Breps matching native SolidIntersectionManager:
        /// Collects Ai ∩ Bj against the ORIGINAL A set (does not chain cutters).
        /// Caches operand meshes across all pairs.
        /// </summary>
        public static List<Brep> BooleanIntersection(
            GpuSession session,
            IEnumerable<Brep> setA,
            IEnumerable<Brep> setB,
            double tolerance,
            Action<string> onWarning = null,
            BrepKernel kernel = BrepKernel.RhinoCommon)
        {
            ResetTimings();
            var initialA = new List<Brep>(setA.Where(b => b != null));
            var results = new List<Brep>();
            using var cache = new OperandMeshCache(session);

            foreach (var cutter in setB)
            {
                if (cutter == null) continue;
                foreach (var brep in initialA)
                {
                    if (brep == null) continue;
                    var inter = ComputePair(session, brep, cutter, BooleanOperation.Intersection, tolerance, onWarning, cache, kernel);
                    if (inter != null && inter.Length > 0)
                    {
                        results.AddRange(inter);
                    }
                }
            }
            return results;
        }

        /// <summary>
        /// Executes Boolean Union across a set of Breps matching native Component_BooleanUnion:
        /// Iterates pairwise merges until a fixed point is reached (no further merges possible),
        /// correctly unifying bridging topology sets such as [A, C, B].
        /// </summary>
        public static List<Brep> BooleanUnion(
            GpuSession session,
            IEnumerable<Brep> set,
            double tolerance,
            Action<string> onWarning = null,
            BrepKernel kernel = BrepKernel.RhinoCommon)
        {
            ResetTimings();
            var all = set.Where(b => b != null).ToList();
            if (all.Count <= 1) return all;
            // Native CreateBooleanUnion passes non-solid (open) operands through unchanged.
            var passthrough = all.Where(b => !b.IsSolid).ToList();
            var pool = all.Where(b => b.IsSolid).ToList();
            if (pool.Count <= 1) return pool.Concat(passthrough).ToList();

            using var cache = new OperandMeshCache(session);

            bool mergedAny = true;
            while (mergedAny && pool.Count > 1)
            {
                mergedAny = false;
                for (int i = 0; i < pool.Count - 1; i++)
                {
                    for (int j = i + 1; j < pool.Count; j++)
                    {
                        var res = ComputePair(session, pool[i], pool[j], BooleanOperation.Union, tolerance, onWarning, cache, kernel);
                        if (res != null && res.Length == 1 && res[0] != null && res[0].IsValid && res[0].IsSolid)
                        {
                            pool[i] = res[0];
                            pool.RemoveAt(j);
                            mergedAny = true;
                            break;
                        }
                    }
                    if (mergedAny) break;
                }
            }

            pool.AddRange(passthrough);
            return pool;
        }
    }
}
