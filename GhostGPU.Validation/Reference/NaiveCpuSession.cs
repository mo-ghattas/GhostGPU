using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Validation.Reference
{
    /// <summary>
    /// In-memory CPU reference implementation of IGpuSession.
    /// Follows the identical lease lifecycle, metrics tracking, and boundary rules as GpuContracts.cs.
    /// Used for independent verification and as an unvectorized baseline.
    /// </summary>
    public sealed class NaiveCpuSession : IGpuSession
    {
        private readonly SessionOptions _options;
        private readonly Guid _sessionId = Guid.NewGuid();
        private readonly HashSet<long> _activeBufferIds = new();
        private readonly Dictionary<long, CpuStorage> _storages = new();
        private readonly GpuMetrics _metrics = new();
        private long _nextBufferId = 1;
        private bool _isDisposed;

        public Guid SessionId => _sessionId;
        public string DeviceName => "GhostGPU.NaiveCpuReference (Host Managed)";
        public bool IsHardwareAccelerated => false;
        public GpuMetrics Metrics => new()
        {
            UploadedBytes = _metrics.UploadedBytes,
            DownloadedBytes = _metrics.DownloadedBytes,
            ActiveBytes = _metrics.ActiveBytes,
            PeakBytes = _metrics.PeakBytes,
            LiveAllocations = _metrics.LiveAllocations,
            KernelDispatches = _metrics.KernelDispatches,
            Synchronizations = _metrics.Synchronizations
        };

        public NaiveCpuSession(SessionOptions? options = null)
        {
            _options = options ?? new SessionOptions { Backend = GpuBackend.CpuDebug };
        }

        private sealed class CpuStorage
        {
            public long StorageId { get; }
            public BufferKind Kind { get; }
            public ElementType ElementType { get; }
            public Residency Residency { get; set; }
            public int Count { get; }
            public long ByteSize { get; }
            public object Data { get; }
            public int RefCount { get; set; }

            public CpuStorage(long storageId, BufferKind kind, ElementType elementType, Residency residency, int count, long byteSize, object data)
            {
                StorageId = storageId;
                Kind = kind;
                ElementType = elementType;
                Residency = residency;
                Count = count;
                ByteSize = byteSize;
                Data = data;
                RefCount = 1;
            }
        }

        private sealed class CpuBufferLease : GpuBuffer
        {
            private readonly NaiveCpuSession _session;
            private readonly CpuStorage _storage;
            private readonly Guid _sessionId;
            private readonly long _id;
            private bool _disposed;

            public override Guid SessionId => _sessionId;
            public override long Id => _id;
            public override BufferKind Kind => _storage.Kind;
            public override int Count => _storage.Count;
            public override bool IsDisposed => _disposed;
            public override ElementType ElementType => _storage.ElementType;
            public override Residency Residency => _storage.Residency;

            internal CpuStorage Storage => _storage;

            public CpuBufferLease(NaiveCpuSession session, CpuStorage storage, long id)
            {
                _session = session;
                _storage = storage;
                _sessionId = session._sessionId;
                _id = id;
            }

            public override GpuBuffer Retain()
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(CpuBufferLease), "Cannot retain a disposed buffer lease.");

                return _session.RetainLease(this);
            }

            public override void Dispose()
            {
                if (_disposed) return;
                _disposed = true;
                _session.ReleaseLease(this);
            }
        }

        private void CheckSession()
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(NaiveCpuSession), "Session is disposed.");
        }

        private CpuStorage ValidateAndGetStorage(GpuBuffer buffer, BufferKind expectedKind)
        {
            CheckSession();
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            if (buffer.IsDisposed)
                throw new ObjectDisposedException(nameof(buffer), "Buffer is already disposed.");
            if (buffer.SessionId != _sessionId)
                throw new InvalidOperationException($"Buffer belongs to foreign session {buffer.SessionId}, not {_sessionId}.");
            if (!_activeBufferIds.Contains(buffer.Id))
                throw new InvalidOperationException($"Buffer handle {buffer.Id} is stale or unrecognized.");
            if (buffer.Kind != expectedKind)
                throw new ArgumentException($"Buffer kind mismatch: expected {expectedKind}, got {buffer.Kind}.");

            if (buffer is not CpuBufferLease lease || !_storages.TryGetValue(lease.Storage.StorageId, out var storage))
                throw new InvalidOperationException("Invalid storage backing lease.");

            return storage;
        }

        private GpuBuffer Allocate(BufferKind kind, ElementType elementType, Residency residency, int count, long byteSize, object data)
        {
            if (_metrics.ActiveBytes + byteSize > _options.MemoryLimitBytes)
                throw new InvalidOperationException($"Allocation of {byteSize} bytes exceeds MemoryLimitBytes ({_options.MemoryLimitBytes}).");

            long storageId = _nextBufferId++;
            var storage = new CpuStorage(storageId, kind, elementType, residency, count, byteSize, data);
            _storages[storageId] = storage;

            _metrics.ActiveBytes += byteSize;
            if (_metrics.ActiveBytes > _metrics.PeakBytes)
                _metrics.PeakBytes = _metrics.ActiveBytes;
            _metrics.LiveAllocations++;

            long leaseId = _nextBufferId++;
            _activeBufferIds.Add(leaseId);
            return new CpuBufferLease(this, storage, leaseId);
        }

        private GpuBuffer Allocate(BufferKind kind, int count, long byteSize, object data) =>
            Allocate(kind, kind == BufferKind.Points || kind == BufferKind.Vectors ? ElementType.V3D : (kind == BufferKind.Scalars ? ElementType.F64 : (kind == BufferKind.Indices ? ElementType.I32 : ElementType.U32)), Residency.Host, count, byteSize, data);

        private GpuBuffer RetainLease(CpuBufferLease source)
        {
            CheckSession();
            source.Storage.RefCount++;
            long newLeaseId = _nextBufferId++;
            _activeBufferIds.Add(newLeaseId);
            return new CpuBufferLease(this, source.Storage, newLeaseId);
        }

        private void ReleaseLease(CpuBufferLease lease)
        {
            _activeBufferIds.Remove(lease.Id);
            lease.Storage.RefCount--;
            if (lease.Storage.RefCount <= 0)
            {
                if (_storages.Remove(lease.Storage.StorageId))
                {
                    _metrics.ActiveBytes -= lease.Storage.ByteSize;
                    _metrics.LiveAllocations--;
                }
            }
        }

        public GpuBuffer UploadPoints(V3[] values) =>
            UploadPoints(values, ElementType.V3D, Residency.Auto);

        public GpuBuffer UploadVectors(V3[] values) =>
            UploadVectors(values, ElementType.V3D, Residency.Auto);

        public GpuBuffer UploadScalars(double[] values) =>
            UploadScalars(values, ElementType.F64, Residency.Auto);

        public GpuBuffer UploadPoints(V3[] values, ElementType format, Residency residency)
        {
            CheckSession();
            ArgumentNullException.ThrowIfNull(values);
            foreach (var v in values)
            {
                if (!v.Finite)
                    throw new ArgumentException("Uploaded points must contain finite coordinates.");
            }

            long byteSize = values.Length * (long)ElementTypes.SizeOf(format);
            _metrics.UploadedBytes += byteSize;
            return Allocate(BufferKind.Points, format, residency, values.Length, byteSize, (V3[])values.Clone());
        }

        public GpuBuffer UploadVectors(V3[] values, ElementType format, Residency residency)
        {
            CheckSession();
            ArgumentNullException.ThrowIfNull(values);
            foreach (var v in values)
            {
                if (!v.Finite)
                    throw new ArgumentException("Uploaded vectors must contain finite coordinates.");
            }

            long byteSize = values.Length * (long)ElementTypes.SizeOf(format);
            _metrics.UploadedBytes += byteSize;
            return Allocate(BufferKind.Vectors, format, residency, values.Length, byteSize, (V3[])values.Clone());
        }

        public GpuBuffer UploadScalars(double[] values, ElementType format, Residency residency)
        {
            CheckSession();
            ArgumentNullException.ThrowIfNull(values);
            foreach (var s in values)
            {
                if (!Check.Finite(s))
                    throw new ArgumentException("Uploaded scalars must be finite.");
            }

            long byteSize = values.Length * (long)ElementTypes.SizeOf(format);
            _metrics.UploadedBytes += byteSize;
            return Allocate(BufferKind.Scalars, format, residency, values.Length, byteSize, (double[])values.Clone());
        }

        public GpuBuffer UploadIndices(int[] values)
        {
            CheckSession();
            ArgumentNullException.ThrowIfNull(values);
            long byteSize = values.Length * 4L;
            _metrics.UploadedBytes += byteSize;
            return Allocate(BufferKind.Indices, ElementType.I32, Residency.Auto, values.Length, byteSize, (int[])values.Clone());
        }

        public GpuBuffer UploadColours(uint[] argbValues)
        {
            CheckSession();
            ArgumentNullException.ThrowIfNull(argbValues);
            long byteSize = argbValues.Length * 4L;
            _metrics.UploadedBytes += byteSize;
            return Allocate(BufferKind.Colours, ElementType.U32, Residency.Auto, argbValues.Length, byteSize, (uint[])argbValues.Clone());
        }

        public GpuBuffer Convert(GpuBuffer buffer, ElementType format, Residency residency)
        {
            CheckSession();
            if (buffer.Kind == BufferKind.Points)
                return UploadPoints(DownloadPoints(buffer), format, residency);
            if (buffer.Kind == BufferKind.Vectors)
                return UploadVectors(DownloadVectors(buffer), format, residency);
            if (buffer.Kind == BufferKind.Scalars)
                return UploadScalars(DownloadScalars(buffer), format, residency);
            if (buffer.Kind == BufferKind.Indices)
                return UploadIndices(DownloadIndices(buffer));
            return UploadColours(DownloadColours(buffer));
        }

        public void TrimPool() { }

        public V3[] DownloadPoints(GpuBuffer buffer)
        {
            var storage = ValidateAndGetStorage(buffer, BufferKind.Points);
            _metrics.DownloadedBytes += storage.ByteSize;
            return (V3[])((V3[])storage.Data).Clone();
        }

        public V3[] DownloadVectors(GpuBuffer buffer)
        {
            var storage = ValidateAndGetStorage(buffer, BufferKind.Vectors);
            _metrics.DownloadedBytes += storage.ByteSize;
            return (V3[])((V3[])storage.Data).Clone();
        }

        public double[] DownloadScalars(GpuBuffer buffer)
        {
            var storage = ValidateAndGetStorage(buffer, BufferKind.Scalars);
            _metrics.DownloadedBytes += storage.ByteSize;
            return (double[])((double[])storage.Data).Clone();
        }

        public int[] DownloadIndices(GpuBuffer buffer)
        {
            var storage = ValidateAndGetStorage(buffer, BufferKind.Indices);
            _metrics.DownloadedBytes += storage.ByteSize;
            return (int[])((int[])storage.Data).Clone();
        }

        public uint[] DownloadColours(GpuBuffer buffer)
        {
            var storage = ValidateAndGetStorage(buffer, BufferKind.Colours);
            _metrics.DownloadedBytes += storage.ByteSize;
            return (uint[])((uint[])storage.Data).Clone();
        }

        public NearestResult ClosestPoints(GpuBuffer queries, GpuBuffer targets)
        {
            var qStorage = ValidateAndGetStorage(queries, BufferKind.Points);
            var tStorage = ValidateAndGetStorage(targets, BufferKind.Points);

            int qCount = qStorage.Count;
            int tCount = tStorage.Count;

            if (qCount > 0 && tCount == 0)
                throw new ArgumentException("Nonempty queries with empty targets is an error.");

            long pairCount = (long)qCount * tCount;
            if (pairCount > _options.MaximumNearestPairs)
                throw new InvalidOperationException($"Pair count {pairCount} exceeds MaximumNearestPairs limit ({_options.MaximumNearestPairs}).");

            _metrics.KernelDispatches++;

            var qPoints = (V3[])qStorage.Data;
            var tPoints = (V3[])tStorage.Data;

            var (pts, idx, dst) = NaiveCpuReference.ClosestPoints(qPoints, tPoints);

            var outPts = Allocate(BufferKind.Points, pts.Length, pts.Length * 24L, pts);
            var outIdx = Allocate(BufferKind.Indices, idx.Length, idx.Length * 4L, idx);
            var outDst = Allocate(BufferKind.Scalars, dst.Length, dst.Length * 8L, dst);

            return new NearestResult(outPts, outIdx, outDst);
        }

        public GpuBuffer Distance(GpuBuffer a, GpuBuffer b)
        {
            var aStorage = ValidateAndGetStorage(a, BufferKind.Points);
            var bStorage = ValidateAndGetStorage(b, BufferKind.Points);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.Distance((V3[])aStorage.Data, (V3[])bStorage.Data);
            return Allocate(BufferKind.Scalars, result.Length, result.Length * 8L, result);
        }

        public GpuBuffer VectorBetween(GpuBuffer a, GpuBuffer b)
        {
            var aStorage = ValidateAndGetStorage(a, BufferKind.Points);
            var bStorage = ValidateAndGetStorage(b, BufferKind.Points);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.VectorBetween((V3[])aStorage.Data, (V3[])bStorage.Data);
            return Allocate(BufferKind.Vectors, result.Length, result.Length * 24L, result);
        }

        public GpuBuffer Unitize(GpuBuffer vectors)
        {
            var storage = ValidateAndGetStorage(vectors, BufferKind.Vectors);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.Unitize((V3[])storage.Data);
            return Allocate(BufferKind.Vectors, result.Length, result.Length * 24L, result);
        }

        public GpuBuffer Amplitude(GpuBuffer vectors, GpuBuffer magnitudes)
        {
            var vStorage = ValidateAndGetStorage(vectors, BufferKind.Vectors);
            var mStorage = ValidateAndGetStorage(magnitudes, BufferKind.Scalars);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.Amplitude((V3[])vStorage.Data, (double[])mStorage.Data);
            return Allocate(BufferKind.Vectors, result.Length, result.Length * 24L, result);
        }

        public GpuBuffer Move(GpuBuffer points, GpuBuffer vectors)
        {
            var pStorage = ValidateAndGetStorage(points, BufferKind.Points);
            var vStorage = ValidateAndGetStorage(vectors, BufferKind.Vectors);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.Move((V3[])pStorage.Data, (V3[])vStorage.Data);
            return Allocate(BufferKind.Points, result.Length, result.Length * 24L, result);
        }

        public GpuBuffer ScalarMath(GpuBuffer a, GpuBuffer b, ScalarOperation operation)
        {
            var aStorage = ValidateAndGetStorage(a, BufferKind.Scalars);
            var bStorage = ValidateAndGetStorage(b, BufferKind.Scalars);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.ScalarMath((double[])aStorage.Data, (double[])bStorage.Data, operation);
            return Allocate(BufferKind.Scalars, result.Length, result.Length * 8L, result);
        }

        public GpuBuffer Bounds(GpuBuffer scalars)
        {
            var storage = ValidateAndGetStorage(scalars, BufferKind.Scalars);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.Bounds((double[])storage.Data);
            return Allocate(BufferKind.Scalars, result.Length, result.Length * 8L, result);
        }

        public GpuBuffer Remap(GpuBuffer scalars, GpuBuffer sourceBounds, double targetMin, double targetMax, bool clamp)
        {
            var sStorage = ValidateAndGetStorage(scalars, BufferKind.Scalars);
            var bStorage = ValidateAndGetStorage(sourceBounds, BufferKind.Scalars);
            if (bStorage.Count != 2)
                throw new ArgumentException("sourceBounds must contain exactly two scalars [min, max].");

            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.Remap((double[])sStorage.Data, (double[])bStorage.Data, targetMin, targetMax, clamp);
            return Allocate(BufferKind.Scalars, result.Length, result.Length * 8L, result);
        }

        public GpuBuffer Map(GpuBuffer scalars, MapFunction function)
        {
            var storage = ValidateAndGetStorage(scalars, BufferKind.Scalars);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.Map((double[])storage.Data, function);
            return Allocate(BufferKind.Scalars, result.Length, result.Length * 8L, result);
        }

        public GpuBuffer ColourRamp(GpuBuffer scalars, uint lowArgb, uint highArgb)
        {
            var storage = ValidateAndGetStorage(scalars, BufferKind.Scalars);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.ColourRamp((double[])storage.Data, lowArgb, highArgb);
            return Allocate(BufferKind.Colours, result.Length, result.Length * 4L, result);
        }

        public GpuBuffer RepeatColours(GpuBuffer colours, int count)
        {
            var storage = ValidateAndGetStorage(colours, BufferKind.Colours);
            _metrics.KernelDispatches++;

            var result = NaiveCpuReference.RepeatColours((uint[])storage.Data, count);
            return Allocate(BufferKind.Colours, result.Length, result.Length * 4L, result);
        }

        public void Synchronize()
        {
            CheckSession();
            _metrics.Synchronizations++;
        }

        public void Reset()
        {
            CheckSession();
            _activeBufferIds.Clear();
            _storages.Clear();
            _metrics.ActiveBytes = 0;
            _metrics.LiveAllocations = 0;
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;
            _activeBufferIds.Clear();
            _storages.Clear();
            _metrics.ActiveBytes = 0;
            _metrics.LiveAllocations = 0;
        }
    }
}
