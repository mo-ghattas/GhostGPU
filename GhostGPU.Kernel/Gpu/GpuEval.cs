using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using GhostGPU.Core;
using GhostGPU.Runtime;
using ILGPU;

namespace GhostGPU.Kernel
{
    public static class GpuEval
    {
        private sealed class SurfaceGpuCache
        {
            public Guid SessionId;
            public GpuBuffer KnotsU;
            public GpuBuffer KnotsV;
            public GpuBuffer Cvs;
        }

        private static readonly ConditionalWeakTable<NurbsSurface, SurfaceGpuCache> _surfaceCache
            = new ConditionalWeakTable<NurbsSurface, SurfaceGpuCache>();
        private static readonly object _cacheLock = new object();

        private static (GpuBuffer knotsU, GpuBuffer knotsV, GpuBuffer cvs) GetOrUploadSurface(GpuSession s, NurbsSurface srf)
        {
            lock (_cacheLock)
            {
                if (_surfaceCache.TryGetValue(srf, out var cache) &&
                    cache.SessionId == s.SessionId &&
                    !cache.KnotsU.IsDisposed &&
                    !cache.KnotsV.IsDisposed &&
                    !cache.Cvs.IsDisposed)
                {
                    return (cache.KnotsU, cache.KnotsV, cache.Cvs);
                }

                // If previous cache exists from another session, dispose old buffers if needed
                if (cache != null)
                {
                    try { cache.KnotsU?.Dispose(); } catch { }
                    try { cache.KnotsV?.Dispose(); } catch { }
                    try { cache.Cvs?.Dispose(); } catch { }
                }

                var bU = s.UploadScalars(srf.KnotsU, ElementType.F64, Residency.Device);
                var bV = s.UploadScalars(srf.KnotsV, ElementType.F64, Residency.Device);
                var bCv = s.UploadScalars(srf.Cvs, ElementType.F64, Residency.Device);

                var newCache = new SurfaceGpuCache
                {
                    SessionId = s.SessionId,
                    KnotsU = bU,
                    KnotsV = bV,
                    Cvs = bCv
                };

                _surfaceCache.Remove(srf);
                _surfaceCache.Add(srf, newCache);

                return (bU, bV, bCv);
            }
        }

        public static GpuBuffer EvaluateSurfacePoints(GpuSession s, NurbsSurface srf, GpuBuffer uv)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));
            if (srf == null) throw new ArgumentNullException(nameof(srf));
            if (uv == null) throw new ArgumentNullException(nameof(uv));

            int count = uv.Count;
            if (count == 0)
            {
                return s.UploadPoints(Array.Empty<V3>(), ElementType.V3D, Residency.Device);
            }

            GpuBuffer uvBuffer = uv;
            bool ownsUvBuffer = false;
            if (uv.ElementType != ElementType.V3D)
            {
                uvBuffer = s.Convert(uv, ElementType.V3D, Residency.Device);
                ownsUvBuffer = true;
            }

            try
            {
                var (bKnotsU, bKnotsV, bCvs) = GetOrUploadSurface(s, srf);
                var lease = s.Rent<V3>(count);
                try
                {
                    using (s.Pin(bKnotsU))
                    using (s.Pin(bKnotsV))
                    using (s.Pin(bCvs))
                    using (s.Pin(uvBuffer))
                    {
                        var viewUv = s.DeviceView<V3>(uvBuffer);
                        var viewKnotsU = s.DeviceView<double>(bKnotsU);
                        var viewKnotsV = s.DeviceView<double>(bKnotsV);
                        var viewCvs = s.DeviceView<double>(bCvs);

                        var kernel = s.GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, int, int, ArrayView<V3>>>(
                            (Action<Index1D, ArrayView<V3>, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, int, int, ArrayView<V3>>)GpuEvalKernels.EvaluateSurfacePointsKernel);

                        kernel(new Index1D(count), viewUv, viewKnotsU, viewKnotsV, viewCvs,
                               srf.DegreeU, srf.DegreeV, srf.CvCountU, srf.CvCountV, srf.IsRational ? 1 : 0,
                               lease.View);
                        s.CountDispatch();
                        return s.Adopt(lease, BufferKind.Points, ElementType.V3D, count);
                    }
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }
            finally
            {
                if (ownsUvBuffer)
                {
                    uvBuffer.Dispose();
                }
            }
        }

        public static GpuBuffer PatchBoxes(GpuSession s, IReadOnlyList<BezierPatch> patches)
        {
            if (s == null) throw new ArgumentNullException(nameof(s));
            if (patches == null) throw new ArgumentNullException(nameof(patches));

            int n = patches.Count;
            if (n == 0)
            {
                return s.UploadPoints(Array.Empty<V3>(), ElementType.V3D, Residency.Device);
            }

            int totalCvs = 0;
            for (int i = 0; i < n; i++)
            {
                var p = patches[i];
                if (p == null) throw new ArgumentNullException(nameof(patches), $"Patch at index {i} is null.");
                totalCvs += (p.DegreeU + 1) * (p.DegreeV + 1);
            }

            double[] meta = new double[n * 2];
            double[] allCvs = new double[totalCvs * 4];

            int cvOffset = 0;
            for (int i = 0; i < n; i++)
            {
                var p = patches[i];
                int cvCount = (p.DegreeU + 1) * (p.DegreeV + 1);
                meta[i * 2] = cvOffset;
                meta[i * 2 + 1] = cvCount;

                int dstBase = cvOffset * 4;
                Array.Copy(p.Cvs, 0, allCvs, dstBase, cvCount * 4);

                cvOffset += cvCount;
            }

            using (var bMeta = s.UploadScalars(meta, ElementType.F64, Residency.Device))
            using (var bCvs = s.UploadScalars(allCvs, ElementType.F64, Residency.Device))
            {
                var lease = s.Rent<V3>(2 * n);
                try
                {
                    using (s.Pin(bMeta))
                    using (s.Pin(bCvs))
                    {
                        var viewMeta = s.DeviceView<double>(bMeta);
                        var viewCvs = s.DeviceView<double>(bCvs);

                        var kernel = s.GetKernel<Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<V3>>>(
                            (Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<V3>>)GpuEvalKernels.PatchBoxesKernel);

                        kernel(new Index1D(n), viewMeta, viewCvs, lease.View);
                        s.CountDispatch();
                        return s.Adopt(lease, BufferKind.Points, ElementType.V3D, 2 * n);
                    }
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }
        }
    }
}
