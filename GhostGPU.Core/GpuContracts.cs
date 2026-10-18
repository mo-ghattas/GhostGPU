using System;

namespace GhostGPU.Core
{
    public enum BufferKind { Points, Vectors, Scalars, Indices, Colours, Mask }
    public enum GpuBackend { CudaOnly, CpuDebug }
    public enum ScalarOperation { Add, Subtract, Multiply, Divide }
    public enum MapFunction { Linear, SmoothStep, Sine }

    /// <summary>
    /// Storage format of one element. Compute is always double precision; compact formats
    /// are widened on load inside kernels and narrowed on store.
    /// V3F stores float offsets from a per-buffer double Origin so large world coordinates
    /// keep their magnitude (precision ~ 1e-7 relative to the buffer extent).
    /// </summary>
    public enum ElementType : byte { F64, F32, I32, U16, U8, U32, V3D, V3F }

    /// <summary>Where a buffer's payload lives. Auto lets the session decide per op.</summary>
    public enum Residency : byte { Auto, Device, Host }

    /// <summary>What to do when a device allocation would exceed MemoryLimitBytes.</summary>
    public enum SpillPolicy : byte { Fail, SpillToHost }

    public static class ElementTypes
    {
        public static int SizeOf(ElementType t)
        {
            switch (t)
            {
                case ElementType.F64: return 8;
                case ElementType.F32: return 4;
                case ElementType.I32: return 4;
                case ElementType.U16: return 2;
                case ElementType.U8: return 1;
                case ElementType.U32: return 4;
                case ElementType.V3D: return 24;
                case ElementType.V3F: return 12;
                default: throw new ArgumentOutOfRangeException(nameof(t));
            }
        }
    }

    /// <summary>A lease to immutable storage. Retain creates an independent lease.</summary>
    public abstract class GpuBuffer : IDisposable
    {
        public abstract Guid SessionId { get; }
        public abstract long Id { get; }
        public abstract BufferKind Kind { get; }
        public abstract int Count { get; }
        public abstract bool IsDisposed { get; }
        /// <summary>Storage format of the payload.</summary>
        public abstract ElementType ElementType { get; }
        /// <summary>Current location of the payload (never Auto). May change after spill or promotion.</summary>
        public abstract Residency Residency { get; }
        /// <summary>Payload bytes, Count * SizeOf(ElementType).</summary>
        public long ByteSize => (long)Count * ElementTypes.SizeOf(ElementType);
        public abstract GpuBuffer Retain();
        public abstract void Dispose();
    }

    public sealed class SessionOptions
    {
        public GpuBackend Backend { get; set; } = GpuBackend.CudaOnly;
        /// <summary>Device budget: live leases plus pooled-but-free device bytes.</summary>
        public long MemoryLimitBytes { get; set; } = 512L * 1024 * 1024;
        /// <summary>Guard for brute-force nearest scans only; spatial-grid queries are not bound by it.</summary>
        public long MaximumNearestPairs { get; set; } = 100000000;
        /// <summary>Max free device bytes kept in the size-class pool for reuse. 0 disables pooling.</summary>
        public long PoolCacheBytes { get; set; } = 128L * 1024 * 1024;
        /// <summary>Pinned host staging ring: slot count and bytes per slot. Transfers larger than a slot are chunked.</summary>
        public int StagingSlots { get; set; } = 4;
        public long StagingSlotBytes { get; set; } = 16L * 1024 * 1024;
        /// <summary>Auto residency: ops whose output element count is below this run on the host when inputs are host-resident.</summary>
        public int HostWorkThreshold { get; set; } = 65536;
        public SpillPolicy Spill { get; set; } = SpillPolicy.SpillToHost;
        /// <summary>Default storage for uploaded points/vectors (V3D or V3F) and scalars (F64 or F32).</summary>
        public ElementType DefaultVectorFormat { get; set; } = ElementType.V3D;
        public ElementType DefaultScalarFormat { get; set; } = ElementType.F64;

        public SessionOptions Clone() => (SessionOptions)MemberwiseClone();
    }

    public sealed class GpuMetrics
    {
        public long UploadedBytes { get; set; }
        public long DownloadedBytes { get; set; }
        /// <summary>Device bytes held by live leases.</summary>
        public long ActiveBytes { get; set; }
        public long PeakBytes { get; set; }
        public long LiveAllocations { get; set; }
        public long KernelDispatches { get; set; }
        public long Synchronizations { get; set; }
        /// <summary>Free device bytes cached in the pool.</summary>
        public long PooledBytes { get; set; }
        public long PoolHits { get; set; }
        public long PoolMisses { get; set; }
        /// <summary>Bytes of buffers whose payload lives on the host.</summary>
        public long HostResidentBytes { get; set; }
        /// <summary>Ops executed on the host path instead of a device dispatch.</summary>
        public long HostDispatches { get; set; }
        /// <summary>Bytes moved device-to-host by spilling under budget pressure.</summary>
        public long SpilledBytes { get; set; }
        /// <summary>Bytes moved through the pinned staging ring.</summary>
        public long StagedBytes { get; set; }
    }

    public sealed class NearestResult : IDisposable
    {
        public GpuBuffer Points { get; }
        public GpuBuffer Indices { get; }
        public GpuBuffer Distances { get; }
        public NearestResult(GpuBuffer points, GpuBuffer indices, GpuBuffer distances)
        { Points = points; Indices = indices; Distances = distances; }
        public void Dispose() { Points.Dispose(); Indices.Dispose(); Distances.Dispose(); }
    }

    /// <summary>Numeric buffers only; Rhino topology and GH path descriptors live in the host adapter.</summary>
    public interface IGpuSession : IDisposable
    {
        Guid SessionId { get; }
        string DeviceName { get; }
        bool IsHardwareAccelerated { get; }
        GpuMetrics Metrics { get; }
        GpuBuffer UploadPoints(V3[] values);
        GpuBuffer UploadVectors(V3[] values);
        GpuBuffer UploadScalars(double[] values);
        /// <summary>Stored as U16 when every value fits [0, 65535], else I32.</summary>
        GpuBuffer UploadIndices(int[] values);
        GpuBuffer UploadColours(uint[] argbValues);
        /// <summary>format: V3D or V3F. residency Auto keeps small uploads host-side until a device op needs them.</summary>
        GpuBuffer UploadPoints(V3[] values, ElementType format, Residency residency);
        GpuBuffer UploadVectors(V3[] values, ElementType format, Residency residency);
        /// <summary>format: F64 or F32.</summary>
        GpuBuffer UploadScalars(double[] values, ElementType format, Residency residency);
        V3[] DownloadPoints(GpuBuffer buffer);
        V3[] DownloadVectors(GpuBuffer buffer);
        double[] DownloadScalars(GpuBuffer buffer);
        int[] DownloadIndices(GpuBuffer buffer);
        uint[] DownloadColours(GpuBuffer buffer);
        NearestResult ClosestPoints(GpuBuffer queries, GpuBuffer targets);
        GpuBuffer Distance(GpuBuffer a, GpuBuffer b);
        GpuBuffer VectorBetween(GpuBuffer a, GpuBuffer b);
        GpuBuffer Unitize(GpuBuffer vectors);
        GpuBuffer Amplitude(GpuBuffer vectors, GpuBuffer magnitudes);
        GpuBuffer Move(GpuBuffer points, GpuBuffer vectors);
        GpuBuffer ScalarMath(GpuBuffer a, GpuBuffer b, ScalarOperation operation);
        GpuBuffer Bounds(GpuBuffer scalars);
        GpuBuffer Remap(GpuBuffer scalars, GpuBuffer sourceBounds, double targetMin, double targetMax, bool clamp);
        GpuBuffer Map(GpuBuffer scalars, MapFunction function);
        GpuBuffer ColourRamp(GpuBuffer scalars, uint lowArgb, uint highArgb);
        GpuBuffer RepeatColours(GpuBuffer colours, int count);
        /// <summary>Copy to a new buffer in another format/residency (e.g. V3D to V3F, Host to Device).</summary>
        GpuBuffer Convert(GpuBuffer buffer, ElementType format, Residency residency);
        /// <summary>Release all free pooled device memory back to the driver.</summary>
        void TrimPool();
        void Synchronize();
        void Reset();
    }
}
