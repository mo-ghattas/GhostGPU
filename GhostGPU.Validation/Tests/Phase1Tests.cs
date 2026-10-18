using System;
using System.Diagnostics;
using GhostGPU.Core;
using GhostGPU.Runtime;
using GhostGPU.Validation.Models;
using ILGPU.Runtime;

namespace GhostGPU.Validation.Tests
{
    /// <summary>Independent numerical examples and safety boundaries for the resident node pack.</summary>
    public static class Phase1Tests
    {
        public static void RunAll(IGpuSession source, Action<TestResultItem> record)
        {
            var session = (GpuSession)source;
            foreach (var residency in new[] { Residency.Host, Residency.Device })
            {
                Case("Construct_Deconstruct_RepeatLast_" + residency, () => Coordinates(session, residency), record);
                Case("Vector_Algebra_" + residency, () => Algebra(session, residency), record);
                Case("Polar_Rotation_Projection_" + residency, () => Rotation(session, residency), record);
                Case("Point_Affine_Interpolation_" + residency, () => Points(session, residency), record);
                Case("Face_Vertex_Analysis_" + residency, () => MeshAnalysis(session, residency), record);
                Case("Mesh_Affine_" + residency, () => MeshTransforms(session, residency), record);
            }
            Case("Empty_And_Invalid_Buffers", () => Boundaries(session), record);
            Case("Compact_FarOrigin", () => Compact(session), record);
            Case("Resident_Expanded_Chain", () => Resident(session), record);
            Case("Resident_Mesh_Transform_Parameters", () => ResidentMeshTransforms(session), record);
            Case("Mesh_Invalid_CSR_Empty_And_Affine", () => MeshBoundaries(session), record);
            Case("Mesh_Device_Index_Control_Readback", () => MeshControlReadback(session), record);
            Case("Deconstruct_Memory_Failure_Rollback", () => AllocationRollback(session), record);
            Case("Affine_Staging_Stream_Reuse", () => AffineStaging(session), record);
            Case("Device_Invalid_CSR_And_Face_Indices", () => DeviceIndexRejection(session), record);
        }

        private static void Case(string name, Action body, Action<TestResultItem> record)
        {
            var clock = Stopwatch.StartNew();
            try { body(); record(new TestResultItem { Name = name, Category = "Phase1", Outcome = TestOutcome.Pass, DurationMs = clock.Elapsed.TotalMilliseconds }); }
            catch (Exception e) { record(new TestResultItem { Name = name, Category = "Phase1", Outcome = TestOutcome.Fail, DurationMs = clock.Elapsed.TotalMilliseconds, Message = e.ToString() }); }
        }

        private static void Check(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
        private static void Near(double actual, double expected, double tolerance = 1e-10)
        {
            if (double.IsNaN(expected)) { Check(double.IsNaN(actual), $"Expected NaN, got {actual:R}"); return; }
            Check(!double.IsNaN(actual) && Math.Abs(actual - expected) <= tolerance * Math.Max(1, Math.Abs(expected)), $"Expected {expected:R}, got {actual:R}");
        }
        private static void Near(V3 actual, V3 expected) { Near(actual.X, expected.X); Near(actual.Y, expected.Y); Near(actual.Z, expected.Z); }
        private static void Throws(Action body)
        {
            try { body(); } catch (ArgumentException) { return; } catch (InvalidOperationException) { return; } catch (IndexOutOfRangeException) { return; }
            throw new InvalidOperationException("Expected rejected invalid input.");
        }
        private static GpuBuffer Numbers(GpuSession s, Residency r, params double[] v) => s.UploadScalars(v, ElementType.F64, r);
        private static GpuBuffer Vectors(GpuSession s, Residency r, params V3[] v) => s.UploadVectors(v, ElementType.V3D, r);
        private static GpuBuffer PointsBuffer(GpuSession s, Residency r, params V3[] v) => s.UploadPoints(v, ElementType.V3D, r);

        private static void Coordinates(GpuSession s, Residency r)
        {
            using var x = Numbers(s, r, 1, 2, 3); using var y = Numbers(s, r, 10); using var z = Numbers(s, r, -1, -2);
            using var points = s.ConstructPoints(x, y, z);
            var p = s.DownloadPoints(points); Check(p.Length == 3, "XYZ repeat-last count");
            Near(p[0], new V3(1, 10, -1)); Near(p[2], new V3(3, 10, -2));
            s.DeconstructPoints(points, out var px, out var py, out var pz);
            using (px) using (py) using (pz) { Near(s.DownloadScalars(px)[2], 3); Near(s.DownloadScalars(py)[2], 10); Near(s.DownloadScalars(pz)[2], -2); }
            using var vectors = s.ConstructVectors(x, y, z);
            s.DeconstructVectors(vectors, out var vx, out var vy, out var vz);
            using (vx) using (vy) using (vz) { Near(s.DownloadScalars(vz)[1], -2); }
            Near(s.DownloadVectors(vectors)[1], new V3(2, 10, -2));
        }

        private static void Algebra(GpuSession s, Residency r)
        {
            using var a = Vectors(s, r, new V3(3, 4, 0), new V3(0, 0, 0));
            using var b = Vectors(s, r, new V3(0, 0, 2)); using var f = Numbers(s, r, -2);
            using var add = s.VectorAdd(a, b); Near(s.DownloadVectors(add)[0], new V3(3, 4, 2));
            using var subtract = s.VectorSubtract(a, b); Near(s.DownloadVectors(subtract)[1], new V3(0, 0, -2));
            using var reverse = s.VectorReverse(a); Near(s.DownloadVectors(reverse)[0], new V3(-3, -4, 0));
            using var length = s.VectorLength(a); Near(s.DownloadScalars(length)[0], 5); Near(s.DownloadScalars(length)[1], 0);
            using var scaled = s.VectorScale(a, f); Near(s.DownloadVectors(scaled)[0], new V3(-6, -8, 0));
            using var divided = s.VectorDivide(a, f); Near(s.DownloadVectors(divided)[0], new V3(-1.5, -2, 0));
            using var ux = s.UnitX(f); Near(s.DownloadVectors(ux)[0], new V3(-2, 0, 0));
            using var uy = s.UnitY(f); Near(s.DownloadVectors(uy)[0], new V3(0, -2, 0));
            using var uz = s.UnitZ(f); Near(s.DownloadVectors(uz)[0], new V3(0, 0, -2));
            using var dot = s.VectorDotProduct(a, b); Near(s.DownloadScalars(dot)[0], 0);
            using var cross = s.VectorCrossProduct(a, b); Near(s.DownloadVectors(cross)[0], new V3(8, -6, 0));
            using var angle = s.VectorAngle(a, b); Near(s.DownloadScalars(angle)[0], Math.PI / 2);
            using var huge = Vectors(s, r, new V3(1e155, 1e155, 0));
            using var hugeLength = s.VectorLength(huge); Near(s.DownloadScalars(hugeLength)[0], Math.Sqrt(2) * 1e155);
            using var hugeA = Vectors(s, r, new V3(1e155, 0, 0)); using var hugeB = Vectors(s, r, new V3(0, 1e155, 0));
            using var hugeAngle = s.VectorAngle(hugeA, hugeB); Near(s.DownloadScalars(hugeAngle)[0], Math.PI / 2);
            using var tiny = Vectors(s, r, new V3(1e-310, 0, 0)); using var tinyDivisor = Numbers(s, r, 1e-310);
            using var tinyQuotient = s.VectorDivide(tiny, tinyDivisor); Near(s.DownloadVectors(tinyQuotient)[0], new V3(1, 0, 0));
        }

        private static void Rotation(GpuSession s, Residency r)
        {
            using var radius = Numbers(s, r, 2); using var angle = Numbers(s, r, Math.PI / 2); using var zero = Numbers(s, r, 0);
            using var cyl = s.ConstructCylindricalPoints(radius, angle, zero); Near(s.DownloadPoints(cyl)[0], new V3(0, 2, 0));
            // Spherical convention is azimuth theta, elevation phi from the XY plane.
            using var spherical = s.ConstructSphericalPoints(radius, angle, zero); Near(s.DownloadPoints(spherical)[0], new V3(0, 2, 0));
            using var v = Vectors(s, r, new V3(1, 0, 3)); using var axis = Vectors(s, r, new V3(0, 0, 2));
            using var projected = s.ProjectVectorToPlane(v, axis); Near(s.DownloadVectors(projected)[0], new V3(1, 0, 0));
            using var rotated = s.RotateVectorAxis(v, axis, angle); Near(s.DownloadVectors(rotated)[0], new V3(0, 1, 3));
        }

        private static void Points(GpuSession s, Residency r)
        {
            using var p = PointsBuffer(s, r, new V3(2, 1, 3)); using var origin = PointsBuffer(s, r, new V3(1, 1, 1));
            using var factor = Numbers(s, r, 2); using var t = Numbers(s, r, 0.25);
            using var scaled = s.PointScale(p, origin, factor); Near(s.DownloadPoints(scaled)[0], new V3(3, 1, 5));
            using var lerp = s.PointInterpolate(origin, p, t); Near(s.DownloadPoints(lerp)[0], new V3(1.25, 1, 1.5));
            using var normal = Vectors(s, r, new V3(0, 0, 2)); using var angle = Numbers(s, r, Math.PI / 2);
            using var rotated = s.PointRotateAxis(p, origin, normal, angle); Near(s.DownloadPoints(rotated)[0], new V3(1, 2, 3));
            using var mirror = s.PointMirror(p, origin, normal); Near(s.DownloadPoints(mirror)[0], new V3(2, 1, -1));
            using var project = s.PointProject(p, origin, normal); Near(s.DownloadPoints(project)[0], new V3(2, 1, 1));
        }

        private static void MeshAnalysis(GpuSession s, Residency r)
        {
            using var vertices = PointsBuffer(s, r, new V3(0, 0, 0), new V3(2, 0, 0), new V3(2, 2, 0), new V3(0, 2, 0), new V3(9, 9, 9));
            using var faces = s.UploadMeshFaceIndices(new[] { new MeshFaceIndices(0, 1, 2, 3), new MeshFaceIndices(0, 0, 0) });
            using var analysis = s.AnalyzeMeshFaces(vertices, faces);
            Near(s.DownloadPoints(analysis.Centers)[0], new V3(1, 1, 0));
            Near(s.DownloadVectors(analysis.Normals)[0], new V3(0, 0, 1)); Near(s.DownloadVectors(analysis.Normals)[1], default);
            Near(s.DownloadScalars(analysis.Areas)[0], 4); Near(s.DownloadScalars(analysis.Areas)[1], 0);
            using var offsets = s.UploadIndices(new[] { 0, 2, 3, 4, 5, 5 });
            using var adjacent = s.UploadIndices(new[] { 0, 1, 0, 0, 0 });
            using var normals = s.MeshVertexNormals(vertices, faces, offsets, adjacent);
            Near(s.DownloadVectors(normals)[0], new V3(0, 0, 1)); Near(s.DownloadVectors(normals)[4], default);
            using var warped = PointsBuffer(s, r, new V3(0, 0, 0), new V3(1, 0, 0), new V3(1, 1, 1), new V3(0, 1, 0));
            using var quad = s.UploadMeshFaceIndices(new[] { new MeshFaceIndices(0, 1, 2, 3) });
            using var area = s.MeshFaceAreas(warped, quad); Near(s.DownloadScalars(area)[0], Math.Sqrt(2));
        }

        private static void MeshTransforms(GpuSession s, Residency r)
        {
            using var points = PointsBuffer(s, r, new V3(2, 0, 0), new V3(0, 2, 0)); using var center = PointsBuffer(s, r, default(V3));
            using var factor = Numbers(s, r, 0.5); using var axis = Vectors(s, r, new V3(0, 0, 1)); using var angle = Numbers(s, r, Math.PI / 2);
            using var scale = s.MeshScale(points, center, factor); Near(s.DownloadPoints(scale)[0], new V3(1, 0, 0));
            using var rotate = s.MeshRotate(points, center, axis, angle); Near(s.DownloadPoints(rotate)[0], new V3(0, 2, 0));
            using var mirror = s.MeshMirror(points, center, axis); Near(s.DownloadPoints(mirror)[1], new V3(0, 2, 0));
            using var affine = s.MeshTransform(points, new double[] { 2, 0, 0, 10, 0, 3, 0, -4, 0, 0, 1, 5, 0, 0, 0, 1 });
            Near(s.DownloadPoints(affine)[0], new V3(14, -4, 5));
        }

        private static void Boundaries(GpuSession s)
        {
            using var empty = Numbers(s, Residency.Device); using var one = Numbers(s, Residency.Device, 1);
            using var points = s.ConstructPoints(empty, one, one); Check(points.Count == 0, "Empty XYZ was expanded.");
            using var vectors = Vectors(s, Residency.Device, new V3(1, 0, 0));
            Throws(() => { using var bad = s.VectorAdd(vectors, one); });
            using var foreign = new GpuSession(new SessionOptions { Backend = s.IsHardwareAccelerated ? GpuBackend.CudaOnly : GpuBackend.CpuDebug });
            using var other = foreign.UploadVectors(new[] { new V3(1, 0, 0) });
            Throws(() => { using var bad = s.VectorAdd(vectors, other); });
            using var badFaces = s.UploadMeshFaceIndices(new[] { new MeshFaceIndices(0, 1, 99) });
            using var v = PointsBuffer(s, Residency.Device, default(V3), new V3(1, 0, 0), new V3(0, 1, 0));
            Throws(() => { using var bad = s.AnalyzeMeshFaces(v, badFaces); });
        }

        private static void Compact(GpuSession s)
        {
            using var p = s.UploadPoints(new[] { new V3(1e8, 2e8, 3e8), new V3(1e8 + 1, 2e8 + 2, 3e8 + 3) }, ElementType.V3F, Residency.Device);
            using var o = s.UploadPoints(new[] { new V3(1e8, 2e8, 3e8) }, ElementType.V3F, Residency.Device); using var f = Numbers(s, Residency.Device, 2);
            using var scaled = s.PointScale(p, o, f); Near(s.DownloadPoints(scaled)[1], new V3(1e8 + 2, 2e8 + 4, 3e8 + 6));
        }

        private static void Resident(GpuSession s)
        {
            using var x = Numbers(s, Residency.Device, 1, 2, 3); using var y = Numbers(s, Residency.Device, 0); using var z = Numbers(s, Residency.Device, 1);
            long downloads = s.Metrics.DownloadedBytes;
            long dispatches = s.Metrics.KernelDispatches;
            using var points = s.ConstructPoints(x, y, z); using var vectors = s.ConstructVectors(x, y, z);
            using var cross = s.VectorCrossProduct(vectors, vectors); using var moved = s.Move(points, cross);
            using var factor = s.VectorLength(vectors); using var scaled = s.PointScale(moved, points, factor);
            s.Synchronize();
            Check(s.Metrics.DownloadedBytes == downloads, "Expanded resident chain downloaded intermediate geometry.");
            Check(s.Metrics.KernelDispatches > dispatches, "Device inputs never dispatched kernels.");
            Check(s.DownloadPoints(scaled).Length == 3, "Expanded resident chain count.");
        }

        private static void ResidentMeshTransforms(GpuSession s)
        {
            using var p = PointsBuffer(s, Residency.Device, new V3(1, 0, 0), new V3(2, 0, 0));
            using var origin = PointsBuffer(s, Residency.Device, default(V3));
            using var axis = Vectors(s, Residency.Device, new V3(0, 0, 1));
            using var angle = Numbers(s, Residency.Device, Math.PI / 2);
            long before = s.Metrics.DownloadedBytes;
            using var rotate = s.MeshRotate(p, origin, axis, angle);
            using var mirror = s.MeshMirror(rotate, origin, axis);
            s.Synchronize();
            Check(s.Metrics.DownloadedBytes == before, "Mesh transforms downloaded resident origin/axis/angle parameters.");
            Near(s.DownloadPoints(mirror)[1], new V3(0, 2, 0));
        }

        private static void MeshBoundaries(GpuSession s)
        {
            using var p = PointsBuffer(s, Residency.Host, default(V3), new V3(1, 0, 0), new V3(0, 1, 0));
            using var faces = s.UploadMeshFaceIndices(new[] { new MeshFaceIndices(0, 1, 2) });
            using var invalidOffsets = s.UploadIndices(new[] { 0, 2, 1, 3 });
            using var adjacency = s.UploadIndices(new[] { 0, 0, 0 });
            Throws(() => { using var bad = s.MeshVertexNormals(p, faces, invalidOffsets, adjacency); });
            using var offsets = s.UploadIndices(new[] { 0, 1, 2, 3 });
            Throws(() => { using var bad = s.MeshVertexNormals(p, faces, offsets, adjacency, (MeshNormalWeighting)99); });
            using var emptyPoints = PointsBuffer(s, Residency.Device);
            Throws(() => { using var bad = s.AnalyzeMeshFaces(emptyPoints, faces); });
            Throws(() => { using var bad = s.MeshTransform(p, new double[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1 }); });
            Throws(() => { using var bad = s.MeshTransform(p, new double[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1e-13, 0, 0, 1 }); });
            using var noFaces = s.UploadMeshFaceIndices(Array.Empty<MeshFaceIndices>());
            using var zeroOffsets = s.UploadIndices(new[] { 0, 0, 0, 0 }); using var noAdjacency = s.UploadIndices(Array.Empty<int>());
            using var normals = s.MeshVertexNormals(p, noFaces, zeroOffsets, noAdjacency);
            Check(normals.Count == p.Count, "Isolated vertices lost normal cardinality.");
            foreach (var normal in s.DownloadVectors(normals)) Near(normal, default);
        }

        private static void MeshControlReadback(GpuSession s)
        {
            using var points = PointsBuffer(s, Residency.Device, default(V3), new V3(1, 0, 0), new V3(0, 1, 0));
            var topology = new MeshFaceIndices[20000];
            for (int i = 0; i < topology.Length; i++) topology[i] = new MeshFaceIndices(0, 1, 2);
            using var faces = s.UploadMeshFaceIndices(topology);
            Check(faces.Residency == Residency.Device, "Fixture must exercise device index validation.");
            long before = s.Metrics.DownloadedBytes;
            using var result = s.AnalyzeMeshFaces(points, faces);
            Check(s.Metrics.DownloadedBytes - before == 4, "Expected only a 4-byte index-validation flag, never a vertex/index array readback.");
            Check(result.Areas.Count == topology.Length, "Device face count changed.");
            var areas = s.DownloadScalars(result.Areas); Near(areas[0], 0.5); Near(areas[areas.Length - 1], 0.5);
        }

        private static void AllocationRollback(GpuSession main)
        {
            using var session = new GpuSession(new SessionOptions { Backend = main.IsHardwareAccelerated ? GpuBackend.CudaOnly : GpuBackend.CpuDebug,
                MemoryLimitBytes = 12288, PoolCacheBytes = 0, HostWorkThreshold = 0 });
            using var points = session.UploadPoints(new V3[256], ElementType.V3D, Residency.Device);
            long before = session.Metrics.ActiveBytes;
            Throws(() => session.DeconstructPoints(points, out _, out _, out _));
            Check(session.Metrics.ActiveBytes == before, "Failed multi-output deconstruction leaked device blocks.");
        }

        private static void AffineStaging(GpuSession s)
        {
            GpuBuffer current = PointsBuffer(s, Residency.Device, new V3(0, 2, 3));
            long before = s.Metrics.DownloadedBytes;
            try
            {
                for (int i = 1; i <= 50; i++)
                {
                    var next = s.MeshTransform(current, new double[] { 1, 0, 0, i, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
                    current.Dispose(); current = next;
                }
                s.Synchronize();
                Check(s.Metrics.DownloadedBytes == before, "Affine chain downloaded intermediate geometry.");
                Near(s.DownloadPoints(current)[0], new V3(1275, 2, 3));
            }
            finally { current.Dispose(); }
        }

        private static void DeviceIndexRejection(GpuSession s)
        {
            using var vertices = PointsBuffer(s, Residency.Device, default(V3), new V3(1, 0, 0), new V3(0, 1, 0));
            using var faces = s.UploadMeshFaceIndices(new[] { new MeshFaceIndices(0, 1, 2) });
            using var badOffsetsLease = s.Rent<int>(4); badOffsetsLease.View.CopyFromCPU(new[] { 0, 2, 1, 3 });
            using var badOffsets = s.Adopt(badOffsetsLease, BufferKind.Indices, ElementType.I32, 4);
            using var adjacent = s.UploadIndices(new[] { 0, 0, 0 });
            Throws(() => { using var invalid = s.MeshVertexNormals(vertices, faces, badOffsets, adjacent); });
            using var badFacesLease = s.Rent<int>(4); badFacesLease.View.CopyFromCPU(new[] { 0, 1, 99, 99 });
            using var badFaces = s.Adopt(badFacesLease, BufferKind.Indices, ElementType.I32, 4);
            Throws(() => { using var invalid = s.AnalyzeMeshFaces(vertices, badFaces); });
        }
    }
}
