using System;
using System.Collections.Generic;
using GhostGPU.Core;
using GhostGPU.Runtime;
using ILGPU;
using ILGPU.Runtime;

namespace GhostGPU.Booleans
{
    public static class GpuGeometry
    {
        /// <summary>Above this triangle count WindingNumbers uses the BVH (fast winding number) kernel.</summary>
        public const int BvhThreshold = 1024;

        /// <summary>
        /// Far-field acceptance: node radius &lt; theta * distance, theta = 0.1 with the order-2 expansion. Measured max
        /// |W - brute| on a 2048-triangle sphere (queries 1e-3 off the surface and in the volume): theta 1/3 -&gt; 5e-4,
        /// 0.2 -&gt; 2e-5, 0.15 -&gt; 4e-6, 0.1 -&gt; 3e-7 (error ~ theta^6).
        /// </summary>
        public const double BvhThetaSq = 0.01;

        /// <summary>
        /// Stage 1 broadphase. Boxes = min/max point buffers. Returns Indices buffer of (i,j) pairs, 2 ints per pair,
        /// ordered by i then j. Brute O(nA*nB) count + write ILGPU kernels; the exclusive scan of the per-row counts
        /// between them runs on the host (counted as a host dispatch).
        /// </summary>
        public static GpuBuffer OverlappingBoxPairs(
            GpuSession s, GpuBuffer minA, GpuBuffer maxA, GpuBuffer minB, GpuBuffer maxB, double tolerance)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));
            if (minA == null) throw new ArgumentNullException(nameof(minA));
            if (maxA == null) throw new ArgumentNullException(nameof(maxA));
            if (minB == null) throw new ArgumentNullException(nameof(minB));
            if (maxB == null) throw new ArgumentNullException(nameof(maxB));
            if (minA.Count != maxA.Count || minB.Count != maxB.Count)
                throw new ArgumentException("Min and max box buffers must have equal counts.");

            int countA = minA.Count;
            int countB = minB.Count;
            if (countA == 0 || countB == 0)
                return s.UploadIndices(Array.Empty<int>());

            var scope = new List<IDisposable>();
            try
            {
                var vMinA = PointsView(s, minA, scope);
                var vMaxA = PointsView(s, maxA, scope);
                var vMinB = PointsView(s, minB, scope);
                var vMaxB = PointsView(s, maxB, scope);

                int[] counts;
                var countsLease = s.Rent<int>(countA);
                try
                {
                    var countKernel = s.GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, double, ArrayView<int>>>(
                        (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, double, ArrayView<int>>)BroadphaseKernels.CountOverlapsKernel);
                    countKernel(new Index1D(countA), vMinA, vMaxA, vMinB, vMaxB, countB, tolerance, countsLease.View);
                    s.CountDispatch();
                }
                catch
                {
                    countsLease.Dispose();
                    throw;
                }
                using (var countsBuf = s.Adopt(countsLease, BufferKind.Indices, ElementType.I32, countA))
                    counts = s.DownloadIndices(countsBuf);

                // Host exclusive scan of per-row counts (O(nA), sequential).
                int[] offsets = new int[countA];
                long total = 0;
                for (int i = 0; i < countA; i++)
                {
                    offsets[i] = (int)total;
                    total += counts[i];
                }
                s.CountHostDispatch();
                if (total == 0)
                    return s.UploadIndices(Array.Empty<int>());
                if (total * 2 > int.MaxValue)
                    throw new InvalidOperationException($"Broadphase produced {total} pairs; exceeds the Indices buffer limit.");

                var offsetsBuf = s.UploadIndices(offsets);
                scope.Add(offsetsBuf);
                var pairsLease = s.Rent<int>(total * 2);
                try
                {
                    if (offsetsBuf.ElementType == ElementType.U16)
                    {
                        var k = s.GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, double, ArrayView<ushort>, ArrayView<int>>>(
                            (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, double, ArrayView<ushort>, ArrayView<int>>)BroadphaseKernels.WriteOverlapsKernelU16);
                        k(new Index1D(countA), vMinA, vMaxA, vMinB, vMaxB, countB, tolerance, View<ushort>(s, offsetsBuf, scope), pairsLease.View);
                    }
                    else
                    {
                        var k = s.GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, double, ArrayView<int>, ArrayView<int>>>(
                            (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, double, ArrayView<int>, ArrayView<int>>)BroadphaseKernels.WriteOverlapsKernelI32);
                        k(new Index1D(countA), vMinA, vMaxA, vMinB, vMaxB, countB, tolerance, View<int>(s, offsetsBuf, scope), pairsLease.View);
                    }
                    s.CountDispatch();
                    return s.Adopt(pairsLease, BufferKind.Indices, ElementType.I32, (int)(total * 2));
                }
                catch
                {
                    pairsLease.Dispose();
                    throw;
                }
            }
            finally
            {
                s.Synchronize();
                DisposeScope(scope);
            }
        }

        /// <summary>
        /// Stage 4. Generalized winding number of each point vs a closed or near-closed mesh. Scalars buffer (F64).
        /// Inside = |W| &gt;= 0.5. Brute O(points*tris) kernel up to <see cref="BvhThreshold"/> triangles,
        /// BVH-clustered order-2 fast winding number kernel above it (BVH built once per mesh on the host).
        /// </summary>
        public static GpuBuffer WindingNumbers(GpuSession s, GpuBuffer points, GpuTriMesh mesh)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));
            if (points == null) throw new ArgumentNullException(nameof(points));
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));

            int pointCount = points.Count;
            if (pointCount == 0)
                return s.UploadScalars(Array.Empty<double>(), ElementType.F64, Residency.Device);
            int triCount = mesh.TriangleCount;
            if (triCount == 0)
                return s.UploadScalars(new double[pointCount], ElementType.F64, Residency.Device);

            var scope = new List<IDisposable>();
            try
            {
                var vPoints = PointsView(s, points, scope);
                var vVerts = PointsView(s, mesh.Vertices, scope);
                var lease = s.Rent<double>(pointCount);
                try
                {
                    if (triCount <= BvhThreshold)
                    {
                        if (mesh.Triangles.ElementType == ElementType.U16)
                        {
                            var k = s.GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<ushort>, int, ArrayView<double>>>(
                                (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<ushort>, int, ArrayView<double>>)WindingKernels.BruteWindingNumberKernelU16);
                            k(new Index1D(pointCount), vPoints, vVerts, View<ushort>(s, mesh.Triangles, scope), triCount, lease.View);
                        }
                        else
                        {
                            var k = s.GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<int>, int, ArrayView<double>>>(
                                (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<int>, int, ArrayView<double>>)WindingKernels.BruteWindingNumberKernelI32);
                            k(new Index1D(pointCount), vPoints, vVerts, View<int>(s, mesh.Triangles, scope), triCount, lease.View);
                        }
                    }
                    else
                    {
                        var bvh = mesh.GetBvh(s);
                        var nodes = HostToLease(s, bvh.Nodes, scope);
                        var sorted = HostToLease(s, bvh.SortedTris, scope);
                        var k = s.GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<GpuBvhNode>, ArrayView<V3>, ArrayView<int>, int, double, ArrayView<double>>>(
                            (Action<Index1D, ArrayView<V3>, ArrayView<GpuBvhNode>, ArrayView<V3>, ArrayView<int>, int, double, ArrayView<double>>)WindingKernels.BvhWindingNumberKernel);
                        k(new Index1D(pointCount), vPoints, nodes, vVerts, sorted, triCount, BvhThetaSq, lease.View);
                    }
                    s.CountDispatch();
                    return s.Adopt(lease, BufferKind.Scalars, ElementType.F64, pointCount);
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }
            finally
            {
                s.Synchronize();
                DisposeScope(scope);
            }
        }

        /// <summary>
        /// Host path of <see cref="WindingNumbers"/> for small query counts (runtime convention: work below
        /// SessionOptions.HostWorkThreshold runs on the host). A few hundred FP64 queries cannot fill a GPU, and on
        /// consumer cards (FP64 at 1/64 rate) one such launch costs 30-40 ms; the host does it in well under 1 ms.
        /// Same element functions and BVH as the kernels; counted as a host dispatch.
        /// </summary>
        internal static double[] HostWindingNumbers(GpuSession s, V3[] points, GpuTriMesh mesh)
        {
            var w = new double[points.Length];
            V3[] verts = mesh.HostMesh.Vertices ?? Array.Empty<V3>();
            int[] tris = mesh.HostMesh.Triangles ?? Array.Empty<int>();
            if (tris.Length / 3 <= BvhThreshold)
                System.Threading.Tasks.Parallel.For(0, points.Length, i => w[i] = WindingKernels.HostBruteWinding(points[i], verts, tris));
            else
            {
                var bvh = mesh.GetBvh(s);
                System.Threading.Tasks.Parallel.For(0, points.Length, i => w[i] = WindingKernels.HostBvhWinding(points[i], bvh.Nodes, verts, bvh.SortedTris, BvhThetaSq));
            }
            s.CountHostDispatch();
            return w;
        }

        // The single place buffers become kernel views: pinned (no spill, device-resident) until the scope is disposed.
        private static ArrayView<T> View<T>(GpuSession s, GpuBuffer buffer, List<IDisposable> scope) where T : unmanaged
        {
            scope.Add(s.Pin(buffer));
            return s.DeviceView<T>(buffer);
        }

        // Kernels compute in double: compact V3F buffers are widened into a temporary V3D device copy.
        private static ArrayView<V3> PointsView(GpuSession s, GpuBuffer buffer, List<IDisposable> scope)
        {
            if (buffer.ElementType == ElementType.V3D)
                return View<V3>(s, buffer, scope);
            var wide = s.UploadPoints(s.DownloadPoints(buffer), ElementType.V3D, Residency.Device);
            scope.Add(wide);
            return View<V3>(s, wide, scope);
        }

        private static ArrayView<T> HostToLease<T>(GpuSession s, T[] data, List<IDisposable> scope) where T : unmanaged
        {
            var lease = s.Rent<T>(data.Length);
            scope.Add(lease);
            lease.View.CopyFromCPU(s.Accelerator.DefaultStream, data);
            return lease.View;
        }

        private static void DisposeScope(List<IDisposable> scope)
        {
            for (int i = scope.Count - 1; i >= 0; i--)
                scope[i].Dispose();
        }
    }
}
