using System;
using System.Threading;
using GhostGPU.Core;
using GhostGPU.Runtime.Kernels;
using GhostGPU.Runtime.Memory;
using ILGPU;
using ILGPU.Runtime;

namespace GhostGPU.Runtime
{
    public sealed partial class GpuSession
    {
        // =====================================================================
        // Host / Device Index Helpers
        // =====================================================================

        private int[] HostI32(RuntimeGpuBuffer b)
        {
            var p = b.Storage.ReadPayload(out _);
            if (p is int[] i32) return i32;
            if (p is ushort[] u16)
            {
                var res = new int[u16.Length];
                for (int i = 0; i < u16.Length; i++) res[i] = u16[i];
                return res;
            }
            throw new InvalidOperationException($"Unexpected indices payload: {p?.GetType()}");
        }

        private ArrayView<int> DeviceI32(DeviceArena.Scope scope, RuntimeGpuBuffer b)
        {
            if (b.ElementType == ElementType.I32)
                return b.Storage.GetArrayView<int>();

            if (b.ElementType == ElementType.U16)
            {
                var tmp = scope.Allocate<int>(b.Count);
                GetKernel<Action<Index1D, ArrayView<ushort>, ArrayView<int>>>(
                    (Action<Index1D, ArrayView<ushort>, ArrayView<int>>)MeshAnalysisKernels.WidenU16Kernel)
                    (new Index1D(b.Count), b.Storage.GetArrayView<ushort>(), tmp);
                CountDispatch();
                return tmp;
            }

            throw new InvalidOperationException($"Unsupported index element type: {b.ElementType}");
        }

        private void ValidateIndexBufferBounds(RuntimeGpuBuffer indexBuffer, int maxExclusive, string paramName)
        {
            if (indexBuffer.Count == 0) return;
            if (maxExclusive <= 0)
                throw new IndexOutOfRangeException($"Index buffer '{paramName}' contains entries ({indexBuffer.Count}) but target vertex count is {maxExclusive}.");

            if (indexBuffer.Residency == Residency.Host || indexBuffer.Storage.HostPayload != null)
            {
                int[] arr = HostI32(indexBuffer);
                for (int i = 0; i < arr.Length; i++)
                {
                    if (arr[i] < 0 || arr[i] >= maxExclusive)
                    {
                        throw new IndexOutOfRangeException(
                            $"Index buffer '{paramName}' entry at index {i} ({arr[i]}) is outside valid range [0, {maxExclusive - 1}].");
                    }
                }
                return;
            }

            // Device validation: runs a kernel checking bounds and downloads a single 4-byte invalid flag
            RunDevice(scope =>
            {
                var flag = scope.Allocate<int>(1);
                flag.MemSetToZero();
                var iv = DeviceI32(scope, indexBuffer);

                GetKernel<Action<Index1D, ArrayView<int>, int, int, ArrayView<int>>>(
                    (Action<Index1D, ArrayView<int>, int, int, ArrayView<int>>)MeshAnalysisKernels.ValidateIndicesBoundsKernel)
                    (new Index1D(indexBuffer.Count), iv, indexBuffer.Count, maxExclusive, flag);
                CountDispatch();

                int[] hostFlag = new int[1];
                _stagingRing.Download(flag.Cast<byte>(), 0, hostFlag, 0, sizeof(int));
                Interlocked.Increment(ref _synchronizations);

                if (hostFlag[0] != 0)
                {
                    throw new IndexOutOfRangeException(
                        $"Index buffer '{paramName}' contains indices outside valid range [0, {maxExclusive - 1}].");
                }
                return 0;
            }, indexBuffer);
        }

        private void ValidateCsrOffsets(RuntimeGpuBuffer offsetsBuffer, int vertexCount, int expectedTotalIncidences)
        {
            if (offsetsBuffer.Count != vertexCount + 1)
            {
                throw new ArgumentException(
                    $"Vertex offsets count ({offsetsBuffer.Count}) must equal vertex count + 1 ({vertexCount + 1}).", nameof(offsetsBuffer));
            }

            if (offsetsBuffer.Residency == Residency.Host || offsetsBuffer.Storage.HostPayload != null)
            {
                int[] arr = HostI32(offsetsBuffer);
                if (arr[0] != 0)
                    throw new ArgumentException("First CSR vertex offset must be 0.", nameof(offsetsBuffer));
                if (arr[vertexCount] != expectedTotalIncidences)
                    throw new ArgumentException($"Final CSR vertex offset ({arr[vertexCount]}) must equal total incidences ({expectedTotalIncidences}).", nameof(offsetsBuffer));

                for (int i = 0; i < vertexCount; i++)
                {
                    int start = arr[i];
                    int end = arr[i + 1];
                    if (start < 0 || end > expectedTotalIncidences || start > end)
                    {
                        throw new ArgumentException(
                            $"Invalid non-monotonic or out-of-bounds CSR offset at vertex {i}: [{start}, {end}] in total incidences {expectedTotalIncidences}.", nameof(offsetsBuffer));
                    }
                }
                return;
            }

            // Device CSR validation
            RunDevice(scope =>
            {
                var flag = scope.Allocate<int>(1);
                flag.MemSetToZero();
                var ov = DeviceI32(scope, offsetsBuffer);

                GetKernel<Action<Index1D, ArrayView<int>, int, int, ArrayView<int>>>(
                    (Action<Index1D, ArrayView<int>, int, int, ArrayView<int>>)MeshAnalysisKernels.ValidateCsrOffsetsKernel)
                    (new Index1D(vertexCount + 1), ov, vertexCount, expectedTotalIncidences, flag);
                CountDispatch();

                int[] hostFlag = new int[1];
                _stagingRing.Download(flag.Cast<byte>(), 0, hostFlag, 0, sizeof(int));
                Interlocked.Increment(ref _synchronizations);

                if (hostFlag[0] != 0)
                {
                    throw new ArgumentException(
                        "CSR offsets buffer failed device validation (must start at 0, end at total incidences, be monotonic, and within bounds).", nameof(offsetsBuffer));
                }
                return 0;
            }, offsetsBuffer);
        }

        // =====================================================================
        // Face Indices Upload / Download
        // =====================================================================

        public GpuBuffer UploadMeshFaceIndices(MeshFaceIndices[] faces)
        {
            CheckNotDisposed();
            if (faces == null) throw new ArgumentNullException(nameof(faces));
            if (faces.Length == 0)
                return CreateEmptyBuffer(BufferKind.Indices, ElementType.I32);

            int[] raw = new int[faces.Length * 4];
            for (int i = 0; i < faces.Length; i++)
            {
                int baseIdx = i * 4;
                raw[baseIdx] = faces[i].A;
                raw[baseIdx + 1] = faces[i].B;
                raw[baseIdx + 2] = faces[i].C;
                raw[baseIdx + 3] = faces[i].D;
            }

            return UploadInternal(BufferKind.Indices, ElementType.I32, Residency.Auto, raw, raw.Length, default);
        }

        public GpuBuffer UploadMeshFaceIndices(int[] rawQuadIndices)
        {
            CheckNotDisposed();
            if (rawQuadIndices == null) throw new ArgumentNullException(nameof(rawQuadIndices));
            if (rawQuadIndices.Length % 4 != 0)
                throw new ArgumentException("Raw face indices length must be a multiple of 4 (A, B, C, D per face).", nameof(rawQuadIndices));
            if (rawQuadIndices.Length == 0)
                return CreateEmptyBuffer(BufferKind.Indices, ElementType.I32);

            return UploadInternal(BufferKind.Indices, ElementType.I32, Residency.Auto, (int[])rawQuadIndices.Clone(), rawQuadIndices.Length, default);
        }

        public MeshFaceIndices[] DownloadMeshFaceIndices(GpuBuffer buffer)
        {
            var b = ValidateBuffer(buffer, BufferKind.Indices);
            if (b.Count == 0) return Array.Empty<MeshFaceIndices>();
            if (b.Count % 4 != 0)
                throw new ArgumentException($"Buffer count ({b.Count}) is not a multiple of 4.", nameof(buffer));

            int[] raw = DownloadIndices(b);
            int faceCount = raw.Length / 4;
            var faces = new MeshFaceIndices[faceCount];
            for (int i = 0; i < faceCount; i++)
            {
                int baseIdx = i * 4;
                faces[i] = new MeshFaceIndices(raw[baseIdx], raw[baseIdx + 1], raw[baseIdx + 2], raw[baseIdx + 3]);
            }
            return faces;
        }

        // =====================================================================
        // Face Analysis Operations
        // =====================================================================

        public FaceAnalysisResult AnalyzeMeshFaces(GpuBuffer vertices, GpuBuffer faceIndices)
        {
            var bv = ValidateBuffer(vertices, BufferKind.Points);
            var bf = ValidateBuffer(faceIndices, BufferKind.Indices);

            if (bf.Count % 4 != 0)
                throw new ArgumentException($"Face indices count ({bf.Count}) must be a multiple of 4.", nameof(faceIndices));

            int faceCount = bf.Count / 4;

            // Validate indices BEFORE empty shortcut so nonempty faces with empty vertices reject
            if (faceCount > 0)
            {
                ValidateIndexBufferBounds(bf, bv.Count, nameof(faceIndices));
            }

            if (faceCount == 0)
            {
                return new FaceAnalysisResult(
                    CreateEmptyBuffer(BufferKind.Points, ElementType.V3D),
                    CreateEmptyBuffer(BufferKind.Vectors, ElementType.V3D),
                    CreateEmptyBuffer(BufferKind.Scalars, ElementType.F64));
            }

            if (CanRunOnHost(faceCount, bv, bf))
            {
                var v = HostV3(bv);
                var f = HostI32(bf);

                var outCenters = new V3[faceCount];
                var outNormals = new V3[faceCount];
                var outAreas = new double[faceCount];

                HostFor(faceCount, i =>
                {
                    int baseIdx = i * 4;
                    int aIdx = f[baseIdx];
                    int bIdx = f[baseIdx + 1];
                    int cIdx = f[baseIdx + 2];
                    int dIdx = f[baseIdx + 3];

                    V3 a = v[aIdx];
                    V3 b = v[bIdx];
                    V3 c = v[cIdx];
                    V3 d = v[dIdx];
                    bool isQuad = (cIdx != dIdx);

                    MeshAnalysisKernels.ComputeFaceGeometry(a, b, c, d, isQuad,
                        out outCenters[i], out outNormals[i], out outAreas[i], out _);
                });

                return new FaceAnalysisResult(
                    HostResult(BufferKind.Points, ElementType.V3D, default, outCenters),
                    HostResult(BufferKind.Vectors, ElementType.V3D, default, outNormals),
                    HostResult(BufferKind.Scalars, ElementType.F64, default, outAreas));
            }

            return RunDevice(scope =>
            {
                DeviceLease<V3> leaseCenters = null;
                DeviceLease<V3> leaseNormals = null;
                DeviceLease<double> leaseAreas = null;

                GpuBuffer bufCenters = null;
                GpuBuffer bufNormals = null;
                GpuBuffer bufAreas = null;

                try
                {
                    leaseCenters = Rent<V3>(faceCount);
                    leaseNormals = Rent<V3>(faceCount);
                    leaseAreas = Rent<double>(faceCount);

                    var devV = DeviceV3(scope, bv);
                    var devF = DeviceI32(scope, bf);

                    GetKernel<Action<Index1D, ArrayView<V3>, int, ArrayView<int>, int, ArrayView<V3>, ArrayView<V3>, ArrayView<double>>>(
                        (Action<Index1D, ArrayView<V3>, int, ArrayView<int>, int, ArrayView<V3>, ArrayView<V3>, ArrayView<double>>)MeshAnalysisKernels.FaceAnalysisKernel)
                        (new Index1D(faceCount), devV, bv.Count, devF, faceCount, leaseCenters.View, leaseNormals.View, leaseAreas.View);

                    CountDispatch();

                    bufCenters = Adopt(leaseCenters, BufferKind.Points, ElementType.V3D, faceCount);
                    leaseCenters = null;
                    bufNormals = Adopt(leaseNormals, BufferKind.Vectors, ElementType.V3D, faceCount);
                    leaseNormals = null;
                    bufAreas = Adopt(leaseAreas, BufferKind.Scalars, ElementType.F64, faceCount);
                    leaseAreas = null;

                    var res = new FaceAnalysisResult(bufCenters, bufNormals, bufAreas);
                    bufCenters = null;
                    bufNormals = null;
                    bufAreas = null;
                    return res;
                }
                finally
                {
                    leaseCenters?.Dispose();
                    leaseNormals?.Dispose();
                    leaseAreas?.Dispose();
                    bufCenters?.Dispose();
                    bufNormals?.Dispose();
                    bufAreas?.Dispose();
                }
            }, bv, bf);
        }

        public GpuBuffer MeshFaceCenters(GpuBuffer vertices, GpuBuffer faceIndices)
        {
            using (var res = AnalyzeMeshFaces(vertices, faceIndices))
            {
                return res.Centers.Retain();
            }
        }

        public GpuBuffer MeshFaceNormals(GpuBuffer vertices, GpuBuffer faceIndices)
        {
            using (var res = AnalyzeMeshFaces(vertices, faceIndices))
            {
                return res.Normals.Retain();
            }
        }

        public GpuBuffer MeshFaceAreas(GpuBuffer vertices, GpuBuffer faceIndices)
        {
            using (var res = AnalyzeMeshFaces(vertices, faceIndices))
            {
                return res.Areas.Retain();
            }
        }

        // =====================================================================
        // Vertex Normals Operation (via Precomputed CSR Adjacency)
        // =====================================================================

        public GpuBuffer MeshVertexNormals(
            GpuBuffer vertices,
            GpuBuffer faceIndices,
            GpuBuffer vertexOffsets,
            GpuBuffer adjacentFaces,
            MeshNormalWeighting weighting = MeshNormalWeighting.AreaWeighted)
        {
            if (!Enum.IsDefined(typeof(MeshNormalWeighting), weighting))
                throw new ArgumentOutOfRangeException(nameof(weighting), $"Unsupported mesh normal weighting: {weighting}");

            var bv = ValidateBuffer(vertices, BufferKind.Points);
            var bf = ValidateBuffer(faceIndices, BufferKind.Indices);
            var bo = ValidateBuffer(vertexOffsets, BufferKind.Indices);
            var ba = ValidateBuffer(adjacentFaces, BufferKind.Indices);

            if (bf.Count % 4 != 0)
                throw new ArgumentException($"Face indices count ({bf.Count}) must be a multiple of 4.", nameof(faceIndices));

            int vertexCount = bv.Count;
            int faceCount = bf.Count / 4;

            if (vertexCount == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, ElementType.V3D);

            // Strict CSR offsets validation (start at 0, end at ba.Count, monotonic, within bounds)
            ValidateCsrOffsets(bo, vertexCount, ba.Count);

            if (faceCount > 0)
            {
                ValidateIndexBufferBounds(bf, vertexCount, nameof(faceIndices));
            }
            ValidateIndexBufferBounds(ba, faceCount, nameof(adjacentFaces));

            if (faceCount == 0)
            {
                // Nonempty vertices with zero faces must yield one zero normal per vertex
                if (CanRunOnHost(vertexCount, bv, bf, bo, ba))
                {
                    return HostResult(BufferKind.Vectors, ElementType.V3D, default, new V3[vertexCount]);
                }

                return RunDevice(scope =>
                {
                    return DeviceVectors(scope, BufferKind.Vectors, ElementType.V3D, default, vertexCount, o =>
                    {
                        o.MemSetToZero();
                    });
                }, bv, bf, bo, ba);
            }

            byte weightByte = (byte)weighting;

            if (CanRunOnHost(vertexCount, bv, bf, bo, ba))
            {
                var v = HostV3(bv);
                var f = HostI32(bf);
                var offsets = HostI32(bo);
                var adj = HostI32(ba);

                var outNormals = new V3[vertexCount];

                HostFor(vertexCount, i =>
                {
                    int start = offsets[i];
                    int end = offsets[i + 1];

                    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
                    for (int k = start; k < end; k++)
                    {
                        int faceIdx = adj[k];
                        if (faceIdx < 0 || faceIdx >= faceCount) continue;

                        int baseIdx = faceIdx * 4;
                        int aIdx = f[baseIdx];
                        int bIdx = f[baseIdx + 1];
                        int cIdx = f[baseIdx + 2];
                        int dIdx = f[baseIdx + 3];

                        V3 a = v[aIdx];
                        V3 b = v[bIdx];
                        V3 c = v[cIdx];
                        V3 d = v[dIdx];
                        bool isQuad = (cIdx != dIdx);

                        MeshAnalysisKernels.ComputeFaceGeometry(a, b, c, d, isQuad, out _, out V3 normal, out _, out V3 unnormCross);

                        if (weightByte == 1) // AreaWeighted
                        {
                            sumX += unnormCross.X;
                            sumY += unnormCross.Y;
                            sumZ += unnormCross.Z;
                        }
                        else // Unweighted
                        {
                            sumX += normal.X;
                            sumY += normal.Y;
                            sumZ += normal.Z;
                        }
                    }

                    double len = VectorKernels.GetLengthHelper(sumX, sumY, sumZ);
                    if (len > 2.2250738585072014e-308 && VectorKernels.IsValidDouble(len))
                    {
                        outNormals[i] = new V3(sumX / len, sumY / len, sumZ / len);
                    }
                    else
                    {
                        outNormals[i] = new V3(0, 0, 0);
                    }
                });

                return HostResult(BufferKind.Vectors, ElementType.V3D, default, outNormals);
            }

            return RunDevice(scope =>
            {
                DeviceLease<V3> leaseNormals = null;
                try
                {
                    leaseNormals = Rent<V3>(vertexCount);

                    var devV = DeviceV3(scope, bv);
                    var devF = DeviceI32(scope, bf);
                    var devO = DeviceI32(scope, bo);
                    var devA = DeviceI32(scope, ba);

                    GetKernel<Action<Index1D, ArrayView<V3>, int, ArrayView<int>, int, ArrayView<int>, ArrayView<int>, int, byte, ArrayView<V3>>>(
                        (Action<Index1D, ArrayView<V3>, int, ArrayView<int>, int, ArrayView<int>, ArrayView<int>, int, byte, ArrayView<V3>>)MeshAnalysisKernels.VertexNormalsCSRKernel)
                        (new Index1D(vertexCount), devV, vertexCount, devF, faceCount, devO, devA, ba.Count, weightByte, leaseNormals.View);

                    CountDispatch();

                    var result = Adopt(leaseNormals, BufferKind.Vectors, ElementType.V3D, vertexCount);
                    leaseNormals = null;
                    return result;
                }
                finally
                {
                    leaseNormals?.Dispose();
                }
            }, bv, bf, bo, ba);
        }

        // =====================================================================
        // Fixed-Topology Geometric Transformations
        // =====================================================================

        public GpuBuffer MeshScale(GpuBuffer vertices, GpuBuffer center, GpuBuffer factor)
        {
            var bv = ValidateBuffer(vertices, BufferKind.Points);
            var bc = ValidateBuffer(center, BufferKind.Points);

            if (factor.Kind != BufferKind.Scalars && factor.Kind != BufferKind.Vectors)
                throw new ArgumentException("Scale factor must be a Scalars buffer (uniform) or Vectors buffer (nonuniform).", nameof(factor));

            var bf = ValidateBuffer(factor);

            if (bv.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, ElementType.V3D);
            if (bc.Count == 0 || bf.Count == 0)
                throw new ArgumentException("Scale center and factor buffers must not be empty.");

            // Fixed topology cardinality constraints: parameter buffers must be count 1 or vertex count
            if (bc.Count != 1 && bc.Count != bv.Count)
            {
                throw new InvalidOperationException(
                    $"MeshScale requires center count ({bc.Count}) to equal 1 or mesh vertex count ({bv.Count}) to preserve mesh cardinality.");
            }
            if (bf.Count != 1 && bf.Count != bv.Count)
            {
                throw new InvalidOperationException(
                    $"MeshScale requires factor count ({bf.Count}) to equal 1 or mesh vertex count ({bv.Count}) to preserve mesh cardinality.");
            }

            // Scalar factor: reuse resident PointScale directly with ZERO downloads
            if (bf.Kind == BufferKind.Scalars)
            {
                return PointScale(vertices, center, factor);
            }

            // Vector factor (nonuniform 3D): run resident kernel with ZERO downloads
            int pCount = bv.Count;
            int cCount = bc.Count;
            int fCount = bf.Count;

            if (CanRunOnHost(pCount, bv, bc, bf))
            {
                var pts = HostV3(bv);
                var cnt = HostV3(bc);
                var fac = HostV3(bf);

                var r = new V3[pCount];
                HostFor(pCount, i =>
                {
                    V3 p = pts[i];
                    V3 c = cnt[Rl(i, cCount)];
                    V3 f = fac[Rl(i, fCount)];
                    r[i] = MeshAnalysisKernels.ScalePoint(p, c, f);
                });

                return HostResult(BufferKind.Points, ElementType.V3D, default, r);
            }

            return RunDevice(scope =>
            {
                var devFactors = DeviceV3(scope, bf);
                return DeviceVectors(scope, BufferKind.Points, ElementType.V3D, default, pCount, o =>
                {
                    GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>>(
                        (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>)MeshAnalysisKernels.MeshScaleKernel)
                        (new Index1D(pCount), DeviceV3(scope, bv), DeviceV3(scope, bc), devFactors, pCount, cCount, fCount, o);
                });
            }, bv, bc, bf);
        }

        public GpuBuffer MeshRotate(GpuBuffer vertices, GpuBuffer axisOrigin, GpuBuffer axisDirection, GpuBuffer angleRadians)
        {
            var bv = ValidateBuffer(vertices, BufferKind.Points);
            var bo = ValidateBuffer(axisOrigin, BufferKind.Points);
            var bd = ValidateBuffer(axisDirection, BufferKind.Vectors);
            var ba = ValidateBuffer(angleRadians, BufferKind.Scalars);

            if (bv.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, ElementType.V3D);
            if (bo.Count == 0 || bd.Count == 0 || ba.Count == 0)
                throw new ArgumentException("Axis origin, direction, and angle buffers must not be empty.");

            // Fixed topology cardinality constraints: parameter buffers must be count 1 or vertex count
            if (bo.Count != 1 && bo.Count != bv.Count)
            {
                throw new InvalidOperationException(
                    $"MeshRotate requires axisOrigin count ({bo.Count}) to equal 1 or mesh vertex count ({bv.Count}) to preserve mesh cardinality.");
            }
            if (bd.Count != 1 && bd.Count != bv.Count)
            {
                throw new InvalidOperationException(
                    $"MeshRotate requires axisDirection count ({bd.Count}) to equal 1 or mesh vertex count ({bv.Count}) to preserve mesh cardinality.");
            }
            if (ba.Count != 1 && ba.Count != bv.Count)
            {
                throw new InvalidOperationException(
                    $"MeshRotate requires angleRadians count ({ba.Count}) to equal 1 or mesh vertex count ({bv.Count}) to preserve mesh cardinality.");
            }

            // ZERO DOWNLOADS: reuse PointRotateAxis directly on resident buffers
            return PointRotateAxis(vertices, axisOrigin, axisDirection, angleRadians);
        }

        public GpuBuffer MeshMirror(GpuBuffer vertices, GpuBuffer planeOrigin, GpuBuffer planeNormal)
        {
            var bv = ValidateBuffer(vertices, BufferKind.Points);
            var bo = ValidateBuffer(planeOrigin, BufferKind.Points);
            var bn = ValidateBuffer(planeNormal, BufferKind.Vectors);

            if (bv.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, ElementType.V3D);
            if (bo.Count == 0 || bn.Count == 0)
                throw new ArgumentException("Plane origin and normal buffers must not be empty.");

            // Fixed topology cardinality constraints: parameter buffers must be count 1 or vertex count
            if (bo.Count != 1 && bo.Count != bv.Count)
            {
                throw new InvalidOperationException(
                    $"MeshMirror requires planeOrigin count ({bo.Count}) to equal 1 or mesh vertex count ({bv.Count}) to preserve mesh cardinality.");
            }
            if (bn.Count != 1 && bn.Count != bv.Count)
            {
                throw new InvalidOperationException(
                    $"MeshMirror requires planeNormal count ({bn.Count}) to equal 1 or mesh vertex count ({bv.Count}) to preserve mesh cardinality.");
            }

            // ZERO DOWNLOADS: reuse PointMirror directly on resident buffers
            return PointMirror(vertices, planeOrigin, planeNormal);
        }

        public GpuBuffer MeshTransform(GpuBuffer vertices, double[] matrix4x4)
        {
            CheckNotDisposed();
            var bv = ValidateBuffer(vertices, BufferKind.Points);

            if (matrix4x4 == null) throw new ArgumentNullException(nameof(matrix4x4));
            if (matrix4x4.Length != 16)
                throw new ArgumentException("Affine transform matrix must contain exactly 16 elements (4x4 row-major).", nameof(matrix4x4));

            for (int i = 0; i < 16; i++)
            {
                if (!Check.Finite(matrix4x4[i]))
                    throw new ArgumentException($"Matrix entry at {i} is not finite ({matrix4x4[i]}).");
            }

            // Strictly require last row to be exactly [0, 0, 0, 1] (affine matrix). Reject perspective/projective transforms.
            if (matrix4x4[12] != 0.0 || matrix4x4[13] != 0.0 || matrix4x4[14] != 0.0 || matrix4x4[15] != 1.0)
            {
                throw new ArgumentException(
                    $"MeshTransform requires an affine transformation matrix where row 3 is exactly [0, 0, 0, 1]. " +
                    $"Projective / perspective transformation matrices (row 3: [{matrix4x4[12]}, {matrix4x4[13]}, {matrix4x4[14]}, {matrix4x4[15]}]) are rejected.", nameof(matrix4x4));
            }

            int pCount = bv.Count;
            if (pCount == 0)
                return CreateEmptyBuffer(BufferKind.Points, ElementType.V3D);

            if (CanRunOnHost(pCount, bv))
            {
                var pts = HostV3(bv);
                var r = new V3[pCount];
                double[] m = matrix4x4;

                HostFor(pCount, i =>
                {
                    r[i] = MeshAnalysisKernels.AffineTransformPoint(
                        pts[i],
                        m[0], m[1], m[2], m[3],
                        m[4], m[5], m[6], m[7],
                        m[8], m[9], m[10], m[11]);
                });

                return HostResult(BufferKind.Points, ElementType.V3D, default, r);
            }

            return RunDevice(scope =>
            {
                var devM = scope.Allocate<double>(16);
                _stagingRing.Upload(matrix4x4, 0, devM.Cast<byte>(), 0, 16 * sizeof(double));
                _stagingRing.FlushPending();

                return DeviceVectors(scope, BufferKind.Points, ElementType.V3D, default, pCount, o =>
                {
                    GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<double>, int, ArrayView<V3>>>(
                        (Action<Index1D, ArrayView<V3>, ArrayView<double>, int, ArrayView<V3>>)MeshAnalysisKernels.MeshAffineTransformKernel)
                        (new Index1D(pCount), DeviceV3(scope, bv), devM, pCount, o);
                });
            }, bv);
        }
    }
}
