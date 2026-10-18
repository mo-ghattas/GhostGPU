using System;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Booleans
{
    public enum BooleanOperation
    {
        Union,
        Difference,
        Intersection
    }

    public sealed class TriMesh
    {
        public V3[] Vertices;
        public int[] Triangles; // 3 ints per tri, CCW = outward normal

        public TriMesh() { }

        public TriMesh(V3[] vertices, int[] triangles)
        {
            Vertices = vertices ?? Array.Empty<V3>();
            Triangles = triangles ?? Array.Empty<int>();
        }

        public int TriangleCount => Triangles != null ? Triangles.Length / 3 : 0;
        public int VertexCount => Vertices != null ? Vertices.Length : 0;
    }

    /// <summary>
    /// Resident GPU triangle mesh with positions, indices, and per-triangle bounding boxes (AABBs).
    /// </summary>
    public sealed class GpuTriMesh : IDisposable
    {
        private bool _isDisposed;

        public GpuBuffer Vertices { get; }
        public GpuBuffer Triangles { get; }
        public int TriangleCount { get; }
        public GpuBuffer MinAABB { get; }
        public GpuBuffer MaxAABB { get; }
        public TriMesh HostMesh { get; }

        internal GpuTriMesh(GpuBuffer vertices, GpuBuffer triangles, int triangleCount, GpuBuffer minAabb, GpuBuffer maxAabb, TriMesh hostMesh)
        {
            Vertices = vertices;
            Triangles = triangles;
            TriangleCount = triangleCount;
            MinAABB = minAabb;
            MaxAABB = maxAabb;
            HostMesh = hostMesh;
        }

        public static GpuTriMesh Upload(GpuSession s, TriMesh m, ElementType positionFormat = ElementType.V3D)
        {
            if (s == null)
                throw new ArgumentNullException(nameof(s));
            if (m == null)
                throw new ArgumentNullException(nameof(m));

            var verts = m.Vertices ?? Array.Empty<V3>();
            var tris = m.Triangles ?? Array.Empty<int>();
            if (tris.Length % 3 != 0)
                throw new ArgumentException("Triangles length must be a multiple of 3.", nameof(m));
            int triCount = tris.Length / 3;

            GpuBuffer gpuVerts = null;
            GpuBuffer gpuTris = null;
            GpuBuffer minAabb = null;
            GpuBuffer maxAabb = null;

            try
            {
                gpuVerts = s.UploadPoints(verts, positionFormat, Residency.Device);
                gpuTris = s.UploadIndices(tris);

                // Compute per-triangle AABB
                V3[] minBoxes = new V3[triCount];
                V3[] maxBoxes = new V3[triCount];

                for (int t = 0; t < triCount; t++)
                {
                    int i0 = tris[3 * t];
                    int i1 = tris[3 * t + 1];
                    int i2 = tris[3 * t + 2];

                    if ((uint)i0 >= (uint)verts.Length || (uint)i1 >= (uint)verts.Length || (uint)i2 >= (uint)verts.Length)
                        throw new ArgumentException($"Triangle {t} references a vertex index outside [0, {verts.Length}).", nameof(m));
                    V3 v0 = verts[i0];
                    V3 v1 = verts[i1];
                    V3 v2 = verts[i2];

                    double minX = Math.Min(v0.X, Math.Min(v1.X, v2.X));
                    double minY = Math.Min(v0.Y, Math.Min(v1.Y, v2.Y));
                    double minZ = Math.Min(v0.Z, Math.Min(v1.Z, v2.Z));

                    double maxX = Math.Max(v0.X, Math.Max(v1.X, v2.X));
                    double maxY = Math.Max(v0.Y, Math.Max(v1.Y, v2.Y));
                    double maxZ = Math.Max(v0.Z, Math.Max(v1.Z, v2.Z));

                    minBoxes[t] = new V3(minX, minY, minZ);
                    maxBoxes[t] = new V3(maxX, maxY, maxZ);
                }

                minAabb = s.UploadPoints(minBoxes, ElementType.V3D, Residency.Device);
                maxAabb = s.UploadPoints(maxBoxes, ElementType.V3D, Residency.Device);

                return new GpuTriMesh(gpuVerts, gpuTris, triCount, minAabb, maxAabb, m);
            }
            catch
            {
                gpuVerts?.Dispose();
                gpuTris?.Dispose();
                minAabb?.Dispose();
                maxAabb?.Dispose();
                throw;
            }
        }

        private (GpuBvhNode[] Nodes, int[] SortedTris)? _bvh;
        private readonly object _bvhLock = new object();

        /// <summary>Host-built fast-winding-number BVH, built once per mesh (counted as a host dispatch).</summary>
        internal (GpuBvhNode[] Nodes, int[] SortedTris) GetBvh(GpuSession s)
        {
            lock (_bvhLock)
            {
                if (_bvh == null)
                {
                    _bvh = WindingKernels.BuildBvh(HostMesh.Vertices ?? Array.Empty<V3>(), HostMesh.Triangles ?? Array.Empty<int>());
                    s.CountHostDispatch();
                }
                return _bvh.Value;
            }
        }

        public void Dispose()
        {
            if (_isDisposed)
                return;
            _isDisposed = true;

            Vertices?.Dispose();
            Triangles?.Dispose();
            MinAABB?.Dispose();
            MaxAABB?.Dispose();
        }
    }
}
