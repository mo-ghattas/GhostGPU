using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using GhostGPU.Core;
using GhostGPU.Runtime.Kernels;
using GhostGPU.Runtime.Memory;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace GhostGPU.Runtime
{
    /// <summary>
    /// GPU-accelerated execution session implementation using ILGPU.
    /// Operates on resident immutable device and host buffers with ref-counted lease semantics,
    /// power-of-two size-class memory pooling, pinned staging ring transfers, and compact datatypes.
    /// </summary>
    public sealed partial class GpuSession : IGpuSession
    {
        private readonly SessionOptions _options;
        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly string _deviceName;
        private readonly bool _isHardwareAccelerated;
        private readonly Guid _sessionId = Guid.NewGuid();

        private long _generation = 1;
        private long _nextBufferId = 1;
        private long _uploadedBytes;
        private long _downloadedBytes;
        private long _stagedBytes;
        private long _spilledBytes;
        private long _hostResidentBytes;
        private long _hostDispatches;
        private long _kernelDispatches;
        private long _synchronizations;

        private readonly DeviceMemoryPool _memoryPool;
        private readonly StagingRing _stagingRing;
        private readonly DeviceArena _deviceArena;
        private readonly KernelCache _kernelCache;

        private readonly HashSet<DeviceBufferStorage> _activeStorages = new HashSet<DeviceBufferStorage>();
        private readonly object _storageLock = new object();
        private bool _isDisposed;

        public GpuSession(SessionOptions options = null)
        {
            _options = options != null ? options.Clone() : new SessionOptions();

            if (!Enum.IsDefined(typeof(GpuBackend), _options.Backend))
                throw new ArgumentOutOfRangeException(nameof(options.Backend), $"Unsupported backend: {_options.Backend}");
            if (_options.MemoryLimitBytes <= 0)
                throw new ArgumentOutOfRangeException(nameof(options.MemoryLimitBytes), "MemoryLimitBytes must be positive.");
            if (_options.MaximumNearestPairs <= 0)
                throw new ArgumentOutOfRangeException(nameof(options.MaximumNearestPairs), "MaximumNearestPairs must be positive.");

            try
            {
                // Initialize ILGPU Context
                try
                {
                    _context = Context.Create(builder =>
                    {
                        try
                        {
                            builder.Cuda();
                        }
                        catch
                        {
                            // CUDA driver service registration fallback
                        }
                        builder.CPU();
                        builder.EnableAlgorithms();
                    });
                }
                catch
                {
                    _context = Context.Create(builder =>
                    {
                        builder.CPU();
                        builder.EnableAlgorithms();
                    });
                }

                if (_options.Backend == GpuBackend.CudaOnly)
                {
                    CudaDevice cudaDevice = null;
                    try
                    {
                        foreach (var device in _context.GetCudaDevices())
                        {
                            cudaDevice = device;
                            break;
                        }
                    }
                    catch
                    {
                    }

                    if (cudaDevice == null)
                    {
                        throw new InvalidOperationException(
                            "CUDA accelerator is not available on this system. Default backend requires CUDA hardware. " +
                            "For CPU testing, explicitly specify SessionOptions.Backend = GpuBackend.CpuDebug.");
                    }

                    _accelerator = cudaDevice.CreateAccelerator(_context);
                    _deviceName = _accelerator.Name;
                    _isHardwareAccelerated = true;
                }
                else if (_options.Backend == GpuBackend.CpuDebug)
                {
                    CPUDevice cpuDevice = null;
                    try
                    {
                        foreach (var device in _context.GetCPUDevices())
                        {
                            cpuDevice = device;
                            break;
                        }
                    }
                    catch
                    {
                    }

                    if (cpuDevice == null)
                    {
                        throw new InvalidOperationException("CPU accelerator is not available.");
                    }

                    _accelerator = cpuDevice.CreateAccelerator(_context);
                    _deviceName = _accelerator.Name + " (CPU Debug Mode)";
                    _isHardwareAccelerated = false;
                }
                else
                {
                    throw new ArgumentOutOfRangeException(nameof(options.Backend), $"Unsupported backend: {_options.Backend}");
                }

                _memoryPool = new DeviceMemoryPool(this, _accelerator);
                _stagingRing = new StagingRing(this, _accelerator, _options.StagingSlots, _options.StagingSlotBytes);
                _deviceArena = new DeviceArena(this);
                _kernelCache = new KernelCache(_accelerator);
            }
            catch
            {
                try { _accelerator?.Dispose(); } catch { }
                try { _context?.Dispose(); } catch { }
                throw;
            }
        }

        // =====================================================================
        // Public Extension API (Contract Section 4)
        // =====================================================================

        public Accelerator Accelerator => _accelerator;
        public SessionOptions Options => _options.Clone();
        public Guid SessionId => _sessionId;
        public string DeviceName => _deviceName;
        public bool IsHardwareAccelerated => _isHardwareAccelerated;
        public bool IsDisposed => _isDisposed;
        public long Generation => Volatile.Read(ref _generation);

        internal SessionOptions OptionsInternal => _options;
        internal DeviceMemoryPool MemoryPool => _memoryPool;
        internal StagingRing StagingRing => _stagingRing;
        internal DeviceArena DeviceArena => _deviceArena;

        public TDelegate GetKernel<TDelegate>(Delegate kernelMethod) where TDelegate : Delegate
        {
            CheckNotDisposed();
            return _kernelCache.GetKernel<TDelegate>(kernelMethod);
        }

        public DeviceLease<T> Rent<T>(long count) where T : unmanaged
        {
            CheckNotDisposed();
            if (count <= 0)
                return new DeviceLease<T>(this, null, default, 0);

            long byteSize = count * System.Runtime.InteropServices.Marshal.SizeOf<T>();
            var block = _memoryPool.Rent(byteSize);
            var byteView = (ArrayView<byte>)block.Buffer.View;
            var view = byteView.Cast<T>().SubView(0, count);
            return new DeviceLease<T>(this, block, view, count);
        }

        public GpuBuffer Adopt<T>(DeviceLease<T> lease, BufferKind kind, ElementType type, int count, V3 origin = default) where T : unmanaged
        {
            CheckNotDisposed();
            if (lease == null) throw new ArgumentNullException(nameof(lease));
            if (lease.IsDisposed) throw new ObjectDisposedException(nameof(lease), "Cannot adopt a disposed lease.");

            lease.IsAdopted = true;
            if (lease.Block == null || count == 0)
            {
                return CreateEmptyBuffer(kind, type, origin);
            }

            var storage = new DeviceBufferStorage(
                this,
                kind,
                type,
                count,
                origin,
                lease.Block,
                lease.Block.Buffer,
                null,
                Residency.Device,
                Generation,
                NextBufferId());

            lock (_storageLock)
            {
                _activeStorages.Add(storage);
            }

            return new RuntimeGpuBuffer(this, storage, NextBufferId());
        }

        private sealed class BufferPin : IDisposable
        {
            private DeviceBufferStorage _storage;

            public BufferPin(DeviceBufferStorage storage)
            {
                storage.MarkInUse();
                try
                {
                    storage.EnsureDeviceResident();
                }
                catch
                {
                    storage.UnmarkInUse();
                    throw;
                }
                _storage = storage;
            }

            public void Dispose()
            {
                var s = Interlocked.Exchange(ref _storage, null);
                s?.UnmarkInUse();
            }
        }

        /// <summary>
        /// Marks the buffer in use (never spilled) and promotes it to the device until disposed.
        /// Take a pin before <see cref="DeviceView{T}"/> whenever the view outlives a single call.
        /// </summary>
        public IDisposable Pin(GpuBuffer buffer)
        {
            CheckNotDisposed();
            var b = ValidateBuffer(buffer);
            return new BufferPin(b.Storage);
        }

        public ArrayView<T> DeviceView<T>(GpuBuffer buffer) where T : unmanaged
        {
            CheckNotDisposed();
            var b = ValidateBuffer(buffer);
            int tSize = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            int elemSize = ElementTypes.SizeOf(b.ElementType);
            if (tSize != elemSize)
            {
                throw new ArgumentException(
                    $"Requested view type {typeof(T).Name} size ({tSize} bytes) does not match buffer element type {b.ElementType} size ({elemSize} bytes).");
            }
            var view = b.Storage.GetArrayView<T>();
            _stagingRing.FlushPending(); // a promotion's async upload must land before external kernels read it
            return view;
        }

        public ReadOnlySpan<T> HostView<T>(GpuBuffer buffer) where T : unmanaged
        {
            CheckNotDisposed();
            var b = ValidateBuffer(buffer);
            if (b.Residency != Residency.Host)
                throw new InvalidOperationException($"Buffer is not host-resident (current residency: {b.Residency}).");

            if (b.Storage.HostPayload is T[] arr)
            {
                return new ReadOnlySpan<T>(arr, 0, b.Count);
            }

            throw new InvalidOperationException($"Host payload array type {b.Storage.HostPayload?.GetType()} does not match requested {typeof(T)}[].");
        }

        public V3 OriginOf(GpuBuffer buffer)
        {
            CheckNotDisposed();
            var b = ValidateBuffer(buffer);
            return b.Origin;
        }

        public void CountDispatch()
        {
            Interlocked.Increment(ref _kernelDispatches);
        }

        public void CountHostDispatch()
        {
            Interlocked.Increment(ref _hostDispatches);
        }

        public GpuMetrics Metrics => new GpuMetrics
        {
            UploadedBytes = Interlocked.Read(ref _uploadedBytes),
            DownloadedBytes = Interlocked.Read(ref _downloadedBytes),
            ActiveBytes = _memoryPool.ActiveBytes,
            PeakBytes = _memoryPool.PeakBytes,
            LiveAllocations = _memoryPool.LiveAllocations,
            KernelDispatches = Interlocked.Read(ref _kernelDispatches),
            Synchronizations = Interlocked.Read(ref _synchronizations),
            PooledBytes = _memoryPool.PooledBytes,
            PoolHits = _memoryPool.PoolHits,
            PoolMisses = _memoryPool.PoolMisses,
            HostResidentBytes = Interlocked.Read(ref _hostResidentBytes),
            HostDispatches = Interlocked.Read(ref _hostDispatches),
            SpilledBytes = Interlocked.Read(ref _spilledBytes),
            StagedBytes = Interlocked.Read(ref _stagedBytes)
        };

        internal void AddUploadedBytes(long bytes) => Interlocked.Add(ref _uploadedBytes, bytes);
        internal void AddDownloadedBytes(long bytes) => Interlocked.Add(ref _downloadedBytes, bytes);
        internal void AddStagedBytes(long bytes) => Interlocked.Add(ref _stagedBytes, bytes);
        internal void AddSpilledBytes(long bytes) => Interlocked.Add(ref _spilledBytes, bytes);
        internal void AddHostResidentBytes(long bytes) => Interlocked.Add(ref _hostResidentBytes, bytes);
        internal long NextBufferId() => Interlocked.Increment(ref _nextBufferId);

        private void CheckNotDisposed()
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(GpuSession), "Session has been disposed.");
        }

        internal void RequestSpill(long neededBytes)
        {
            if (_options.Spill != SpillPolicy.SpillToHost)
                return;

            List<DeviceBufferStorage> candidates;
            lock (_storageLock)
            {
                candidates = new List<DeviceBufferStorage>(_activeStorages);
            }

            // Sort by LastAccessTimestamp (LRU first)
            candidates.Sort((a, b) => a.LastAccessTimestamp.CompareTo(b.LastAccessTimestamp));

            long freed = 0;
            foreach (var storage in candidates)
            {
                if (freed >= neededBytes)
                    break;

                long size = storage.ByteSize;
                if (storage.TrySpillToHost())
                {
                    freed += size;
                }
            }
        }

        // =====================================================================
        // Buffer Creation & Validation
        // =====================================================================

        internal GpuBuffer CreateBuffer(
            BufferKind kind,
            ElementType elementType,
            int count,
            V3 origin,
            PooledBlock pooledBlock,
            Array hostPayload,
            Residency residency)
        {
            var storage = new DeviceBufferStorage(
                this,
                kind,
                elementType,
                count,
                origin,
                pooledBlock,
                pooledBlock?.Buffer,
                hostPayload,
                residency,
                Generation,
                NextBufferId());

            lock (_storageLock)
            {
                _activeStorages.Add(storage);
            }

            return new RuntimeGpuBuffer(this, storage, NextBufferId());
        }

        internal GpuBuffer CreateEmptyBuffer(BufferKind kind, ElementType elementType = ElementType.F64, V3 origin = default)
        {
            return CreateBuffer(kind, elementType, 0, origin, null, null, Residency.Device);
        }

        internal void OnStorageDisposed(DeviceBufferStorage storage)
        {
            lock (_storageLock)
            {
                _activeStorages.Remove(storage);
            }
        }

        private RuntimeGpuBuffer ValidateBuffer(GpuBuffer buffer, BufferKind? expectedKind = null)
        {
            CheckNotDisposed();
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            if (!(buffer is RuntimeGpuBuffer runtimeBuffer))
                throw new ArgumentException("Buffer was not created by this runtime session.", nameof(buffer));

            if (runtimeBuffer.SessionId != _sessionId)
                throw new ArgumentException($"Buffer belongs to session {runtimeBuffer.SessionId}, but expected {_sessionId}.", nameof(buffer));

            if (runtimeBuffer.Storage.Generation != Generation || runtimeBuffer.IsDisposed)
                throw new ObjectDisposedException(nameof(buffer), "Buffer is disposed or was invalidated by a session Reset.");

            if (expectedKind.HasValue && runtimeBuffer.Kind != expectedKind.Value)
                throw new ArgumentException($"Buffer kind {runtimeBuffer.Kind} does not match expected {expectedKind.Value}.", nameof(buffer));

            return runtimeBuffer;
        }

        // =====================================================================
        // Payload helpers. Invariant: a host payload's array type always matches ElementType
        // (V3D->V3[], V3F->V3F[], F64->double[], F32->float[], I32->int[], U16->ushort[], U8->byte[], U32->uint[]).
        // =====================================================================

        internal static Array NewPayload(ElementType type, int count)
        {
            switch (type)
            {
                case ElementType.F64: return new double[count];
                case ElementType.F32: return new float[count];
                case ElementType.I32: return new int[count];
                case ElementType.U16: return new ushort[count];
                case ElementType.U8: return new byte[count];
                case ElementType.U32: return new uint[count];
                case ElementType.V3D: return new V3[count];
                case ElementType.V3F: return new V3F[count];
                default: throw new ArgumentOutOfRangeException(nameof(type));
            }
        }

        internal void UploadHostPayloadToDevice(Array hostPayload, ElementType type, int count, ArrayView<byte> targetView)
        {
            _stagingRing.Upload(hostPayload, 0, targetView, 0, (long)count * ElementTypes.SizeOf(type));
        }

        internal Array DownloadDevicePayloadToHost(MemoryBuffer deviceBuffer, ElementType type, int count)
        {
            var target = NewPayload(type, count);
            var view = ((MemoryBuffer1D<byte, Stride1D.Dense>)deviceBuffer).View;
            _stagingRing.Download(view, 0, target, 0, (long)count * ElementTypes.SizeOf(type));
            Interlocked.Increment(ref _synchronizations);
            return target;
        }

        /// <summary>Payload for a public Download*: cached host payload (counted as downloaded) or a device readback.</summary>
        private Array DownloadPayload(RuntimeGpuBuffer b, out bool shared)
        {
            var payload = b.Storage.ReadPayload(out shared);
            if (shared) AddDownloadedBytes(b.ByteSize);
            return payload;
        }

        private static V3[] WidenV3(V3F[] src, V3 origin)
        {
            var dst = new V3[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = src[i].ToV3(origin);
            return dst;
        }

        private static V3F[] NarrowV3(V3[] src, V3 origin)
        {
            var dst = new V3F[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = V3F.FromV3(src[i], origin);
            return dst;
        }

        private static double[] WidenF64(float[] src)
        {
            var dst = new double[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = src[i];
            return dst;
        }

        private static float[] NarrowF32(double[] src)
        {
            var dst = new float[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
            return dst;
        }

        // =====================================================================
        // IGpuSession Upload Overloads (defaults: options formats, Residency.Auto)
        // =====================================================================

        public GpuBuffer UploadPoints(V3[] values) =>
            UploadPoints(values, _options.DefaultVectorFormat, Residency.Auto);

        public GpuBuffer UploadVectors(V3[] values) =>
            UploadVectors(values, _options.DefaultVectorFormat, Residency.Auto);

        public GpuBuffer UploadScalars(double[] values) =>
            UploadScalars(values, _options.DefaultScalarFormat, Residency.Auto);

        public GpuBuffer UploadPoints(V3[] values, ElementType format, Residency residency) =>
            UploadV3(BufferKind.Points, values, format, residency);

        public GpuBuffer UploadVectors(V3[] values, ElementType format, Residency residency) =>
            UploadV3(BufferKind.Vectors, values, format, residency);

        private GpuBuffer UploadV3(BufferKind kind, V3[] values, ElementType format, Residency residency)
        {
            CheckNotDisposed();
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (format != ElementType.V3D && format != ElementType.V3F)
                throw new ArgumentException($"{kind} format must be V3D or V3F, but was {format}.", nameof(format));
            CheckResidency(residency);

            for (int i = 0; i < values.Length; i++)
            {
                if (!values[i].Finite)
                    throw new ArgumentException($"{(kind == BufferKind.Points ? "Point" : "Vector")} at index {i} contains non-finite coordinates ({values[i].X}, {values[i].Y}, {values[i].Z}).");
            }

            if (values.Length == 0)
                return CreateEmptyBuffer(kind, format);

            if (format == ElementType.V3F)
            {
                V3 origin = CompactTypeHelper.ComputeOrigin(values);
                return UploadInternal(kind, format, residency, NarrowV3(values, origin), values.Length, origin);
            }
            return UploadInternal(kind, format, residency, (V3[])values.Clone(), values.Length, default);
        }

        public GpuBuffer UploadScalars(double[] values, ElementType format, Residency residency)
        {
            CheckNotDisposed();
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (format != ElementType.F64 && format != ElementType.F32)
                throw new ArgumentException($"Scalars format must be F64 or F32, but was {format}.", nameof(format));
            CheckResidency(residency);

            for (int i = 0; i < values.Length; i++)
            {
                if (!Check.Finite(values[i]))
                    throw new ArgumentException($"Scalar at index {i} is not finite ({values[i]}).");
            }

            if (values.Length == 0)
                return CreateEmptyBuffer(BufferKind.Scalars, format);

            Array payload = format == ElementType.F32 ? (Array)NarrowF32(values) : (double[])values.Clone();
            return UploadInternal(BufferKind.Scalars, format, residency, payload, values.Length, default);
        }

        public GpuBuffer UploadIndices(int[] values)
        {
            CheckNotDisposed();
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (values.Length == 0)
                return CreateEmptyBuffer(BufferKind.Indices, ElementType.I32);

            bool canNarrow = true;
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] < 0 || values[i] > 65535)
                {
                    canNarrow = false;
                    break;
                }
            }

            if (canNarrow)
            {
                var u16 = new ushort[values.Length];
                for (int i = 0; i < values.Length; i++)
                    u16[i] = (ushort)values[i];
                return UploadInternal(BufferKind.Indices, ElementType.U16, Residency.Auto, u16, values.Length, default);
            }
            return UploadInternal(BufferKind.Indices, ElementType.I32, Residency.Auto, (int[])values.Clone(), values.Length, default);
        }

        public GpuBuffer UploadColours(uint[] argbValues)
        {
            CheckNotDisposed();
            if (argbValues == null) throw new ArgumentNullException(nameof(argbValues));
            if (argbValues.Length == 0)
                return CreateEmptyBuffer(BufferKind.Colours, ElementType.U32);

            return UploadInternal(BufferKind.Colours, ElementType.U32, Residency.Auto, (uint[])argbValues.Clone(), argbValues.Length, default);
        }

        private static void CheckResidency(Residency residency)
        {
            if (!Enum.IsDefined(typeof(Residency), residency))
                throw new ArgumentOutOfRangeException(nameof(residency), $"Invalid Residency: {residency}");
        }

        private GpuBuffer UploadInternal(
            BufferKind kind,
            ElementType format,
            Residency residency,
            Array hostPayload,
            int count,
            V3 origin)
        {
            // Auto keeps small uploads on the host until a device op needs them (Core contract);
            // uploads at/above HostWorkThreshold go straight to the device. Host payloads are not
            // bound by MemoryLimitBytes (device budget), so huge host inputs can stream.
            bool keepOnHost = residency == Residency.Host || (residency == Residency.Auto && count < _options.HostWorkThreshold);
            if (keepOnHost)
            {
                return CreateBuffer(kind, format, count, origin, null, hostPayload, Residency.Host);
            }

            long totalBytes = (long)count * ElementTypes.SizeOf(format);
            if (totalBytes > _options.MemoryLimitBytes)
                throw new InvalidOperationException($"Device upload of {totalBytes} bytes exceeds MemoryLimitBytes ({_options.MemoryLimitBytes}). Use Residency.Host/Auto or raise the budget.");

            var block = _memoryPool.Rent(totalBytes);
            try
            {
                UploadHostPayloadToDevice(hostPayload, format, count, block.Buffer.View);
                return CreateBuffer(kind, format, count, origin, block, null, Residency.Device);
            }
            catch
            {
                _memoryPool.Return(block);
                throw;
            }
        }

        // =====================================================================
        // IGpuSession Download Methods
        // =====================================================================

        public V3[] DownloadPoints(GpuBuffer buffer) => DownloadV3(ValidateBuffer(buffer, BufferKind.Points));

        public V3[] DownloadVectors(GpuBuffer buffer) => DownloadV3(ValidateBuffer(buffer, BufferKind.Vectors));

        private V3[] DownloadV3(RuntimeGpuBuffer b)
        {
            if (b.Count == 0) return new V3[0];
            var p = DownloadPayload(b, out bool shared);
            if (p is V3F[] f) return WidenV3(f, b.Origin);
            return shared ? (V3[])((V3[])p).Clone() : (V3[])p;
        }

        public double[] DownloadScalars(GpuBuffer buffer)
        {
            var b = ValidateBuffer(buffer, BufferKind.Scalars);
            if (b.Count == 0) return new double[0];
            var p = DownloadPayload(b, out bool shared);
            if (p is float[] f) return WidenF64(f);
            return shared ? (double[])((double[])p).Clone() : (double[])p;
        }

        public int[] DownloadIndices(GpuBuffer buffer)
        {
            var b = ValidateBuffer(buffer, BufferKind.Indices);
            if (b.Count == 0) return new int[0];
            var p = DownloadPayload(b, out bool shared);
            if (p is ushort[] u)
            {
                var res = new int[u.Length];
                for (int i = 0; i < u.Length; i++) res[i] = u[i];
                return res;
            }
            return shared ? (int[])((int[])p).Clone() : (int[])p;
        }

        public uint[] DownloadColours(GpuBuffer buffer)
        {
            var b = ValidateBuffer(buffer, BufferKind.Colours);
            if (b.Count == 0) return new uint[0];
            var p = DownloadPayload(b, out bool shared);
            return shared ? (uint[])((uint[])p).Clone() : (uint[])p;
        }

        // =====================================================================
        // IGpuSession Convert & TrimPool
        // =====================================================================

        public GpuBuffer Convert(GpuBuffer buffer, ElementType format, Residency residency)
        {
            CheckNotDisposed();
            var b = ValidateBuffer(buffer);

            if (b.Count == 0)
                return CreateEmptyBuffer(b.Kind, format);

            switch (b.Kind)
            {
                case BufferKind.Points:
                    return UploadPoints(DownloadPoints(b), format, residency);
                case BufferKind.Vectors:
                    return UploadVectors(DownloadVectors(b), format, residency);
                case BufferKind.Scalars:
                    return UploadScalars(DownloadScalars(b), format, residency);
                case BufferKind.Indices:
                    return UploadIndices(DownloadIndices(b));
                default:
                    return UploadColours(DownloadColours(b));
            }
        }

        public void TrimPool()
        {
            CheckNotDisposed();
            _memoryPool.TrimPool();
        }

        // =====================================================================
        // Op plumbing. Every op picks ONE path:
        //  - host: output count < HostWorkThreshold and all inputs host-resident. Plain loop /
        //    Parallel.For over the same static element functions the kernels call. CountHostDispatch.
        //  - device: inputs marked in use (not spillable), promoted, pending uploads fenced, compact
        //    inputs widened into an arena temp, V3D/F64 kernel, compact outputs narrowed.
        // Output formats (contract section 1): points/vectors = widest V3* input; scalars F64 unless
        // every scalar input is F32. V3F output origins are a pure function of input origins, so the
        // host and device paths narrow identically.
        // =====================================================================

        private bool CanRunOnHost(int count, params RuntimeGpuBuffer[] inputs)
        {
            if (count >= _options.HostWorkThreshold)
                return false;

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i].Residency != Residency.Host)
                    return false;
            }
            return true;
        }

        private static int Rl(int i, int length) => i < length ? i : length - 1;

        private static void HostFor(int count, Action<int> body)
        {
            if (count < 4096)
            {
                for (int i = 0; i < count; i++) body(i);
            }
            else
            {
                Parallel.For(0, count, body);
            }
        }

        private static ElementType WidestV3(RuntimeGpuBuffer a, RuntimeGpuBuffer b) =>
            a.ElementType == ElementType.V3F && b.ElementType == ElementType.V3F ? ElementType.V3F : ElementType.V3D;

        private static ElementType ScalarOut(params RuntimeGpuBuffer[] scalarInputs)
        {
            foreach (var s in scalarInputs)
                if (s.ElementType != ElementType.F32) return ElementType.F64;
            return ElementType.F32;
        }

        private V3[] HostV3(RuntimeGpuBuffer b)
        {
            var p = b.Storage.ReadPayload(out _);
            return p as V3[] ?? WidenV3((V3F[])p, b.Origin);
        }

        private double[] HostF64(RuntimeGpuBuffer b)
        {
            var p = b.Storage.ReadPayload(out _);
            return p as double[] ?? WidenF64((float[])p);
        }

        private GpuBuffer HostResult(BufferKind kind, ElementType format, V3 origin, Array values)
        {
            Array payload = values;
            if (format == ElementType.V3F) payload = NarrowV3((V3[])values, origin);
            else if (format == ElementType.F32) payload = NarrowF32((double[])values);
            else origin = default;
            CountHostDispatch();
            return CreateBuffer(kind, format, values.Length, origin, null, payload, Residency.Host);
        }

        private T RunDevice<T>(Func<DeviceArena.Scope, T> body, params RuntimeGpuBuffer[] inputs)
        {
            foreach (var b in inputs) b.Storage.MarkInUse();
            try
            {
                foreach (var b in inputs) b.Storage.EnsureDeviceResident();
                _stagingRing.FlushPending();
                using (var scope = _deviceArena.Begin())
                {
                    return body(scope);
                }
            }
            finally
            {
                foreach (var b in inputs) b.Storage.UnmarkInUse();
            }
        }

        private ArrayView<V3> DeviceV3(DeviceArena.Scope scope, RuntimeGpuBuffer b)
        {
            if (b.ElementType == ElementType.V3D)
                return b.Storage.GetArrayView<V3>();

            var tmp = scope.Allocate<V3>(b.Count);
            GetKernel<Action<Index1D, ArrayView<V3F>, V3, ArrayView<V3>>>(
                (Action<Index1D, ArrayView<V3F>, V3, ArrayView<V3>>)VectorKernels.WidenV3FKernel)
                (new Index1D(b.Count), b.Storage.GetArrayView<V3F>(), b.Origin, tmp);
            CountDispatch();
            return tmp;
        }

        private ArrayView<double> DeviceF64(DeviceArena.Scope scope, RuntimeGpuBuffer b)
        {
            if (b.ElementType == ElementType.F64)
                return b.Storage.GetArrayView<double>();

            var tmp = scope.Allocate<double>(b.Count);
            GetKernel<Action<Index1D, ArrayView<float>, ArrayView<double>>>(
                (Action<Index1D, ArrayView<float>, ArrayView<double>>)ScalarKernels.WidenF32Kernel)
                (new Index1D(b.Count), b.Storage.GetArrayView<float>(), tmp);
            CountDispatch();
            return tmp;
        }

        /// <summary>Runs compute into a V3 view and adopts it as kind/format (narrowing to V3F with origin if needed).</summary>
        private GpuBuffer DeviceVectors(DeviceArena.Scope scope, BufferKind kind, ElementType format, V3 origin, int count, Action<ArrayView<V3>> compute)
        {
            if (format == ElementType.V3D)
            {
                var lease = Rent<V3>(count);
                try
                {
                    compute(lease.View);
                    CountDispatch();
                    return Adopt(lease, kind, ElementType.V3D, count);
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }

            var tmp = scope.Allocate<V3>(count);
            compute(tmp);
            CountDispatch();
            return NarrowToV3F(tmp, kind, origin, count);
        }

        private GpuBuffer NarrowToV3F(ArrayView<V3> src, BufferKind kind, V3 origin, int count)
        {
            var lease = Rent<V3F>(count);
            try
            {
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3F>, V3>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3F>, V3>)VectorKernels.NarrowV3Kernel)
                    (new Index1D(count), src, lease.View, origin);
                CountDispatch();
                return Adopt(lease, kind, ElementType.V3F, count, origin);
            }
            catch
            {
                lease.Dispose();
                throw;
            }
        }

        /// <summary>Runs compute into a double view and adopts it as Scalars (narrowing to F32 if needed).</summary>
        private GpuBuffer DeviceScalars(DeviceArena.Scope scope, ElementType format, int count, Action<ArrayView<double>> compute)
        {
            if (format == ElementType.F64)
            {
                var lease = Rent<double>(count);
                try
                {
                    compute(lease.View);
                    CountDispatch();
                    return Adopt(lease, BufferKind.Scalars, ElementType.F64, count);
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }

            var tmp = scope.Allocate<double>(count);
            compute(tmp);
            CountDispatch();
            var outLease = Rent<float>(count);
            try
            {
                GetKernel<Action<Index1D, ArrayView<double>, ArrayView<float>>>(
                    (Action<Index1D, ArrayView<double>, ArrayView<float>>)ScalarKernels.NarrowF64Kernel)
                    (new Index1D(count), tmp, outLease.View);
                CountDispatch();
                return Adopt(outLease, BufferKind.Scalars, ElementType.F32, count);
            }
            catch
            {
                outLease.Dispose();
                throw;
            }
        }

        // =====================================================================
        // Operations
        // =====================================================================

        /// <summary>Queries above this count that are host-resident stream through the staging ring against small targets.</summary>
        private const int StreamQueryThreshold = 100000;
        private const int StreamTargetLimit = 10000;

        private sealed class NearestGrid
        {
            public ArrayView<int> SortedIndices, CellStart, CellEnd;
            public V3 Min;
            public double CellSize, CellSizeInv;
            public int DimX, DimY, DimZ;
        }

        public NearestResult ClosestPoints(GpuBuffer queries, GpuBuffer targets)
        {
            var q = ValidateBuffer(queries, BufferKind.Points);
            var t = ValidateBuffer(targets, BufferKind.Points);
            var outFormat = WidestV3(q, t);
            // V3F output (both inputs V3F) stores target points relative to the targets' origin.
            V3 outOrigin = outFormat == ElementType.V3F ? t.Origin : default;

            if (q.Count == 0)
            {
                return new NearestResult(
                    CreateEmptyBuffer(BufferKind.Points, outFormat),
                    CreateEmptyBuffer(BufferKind.Indices, ElementType.I32),
                    CreateEmptyBuffer(BufferKind.Scalars, ElementType.F64));
            }

            if (t.Count == 0)
                throw new ArgumentException("Target points buffer cannot be empty when query points are nonempty.", nameof(targets));

            long pairs = (long)q.Count * t.Count;
            bool useBrute = pairs <= 65536 || t.Count <= 128;
            if (useBrute && pairs > _options.MaximumNearestPairs)
                throw new InvalidOperationException($"Maximum nearest pairs exceeded: {pairs} > {_options.MaximumNearestPairs}");

            if (useBrute && CanRunOnHost(q.Count, q, t))
            {
                var qp = HostV3(q);
                var tp = HostV3(t);
                var pts = new V3[q.Count];
                var idx = new int[q.Count];
                var dist = new double[q.Count];
                HostFor(q.Count, i =>
                {
                    double bestDist = double.PositiveInfinity;
                    int bestIdx = 0;
                    for (int j = 0; j < tp.Length; j++)
                    {
                        double d = VectorKernels.RobustDistance(qp[i], tp[j]);
                        if (SpatialKernels.IsBetter(d, j, bestDist, bestIdx)) { bestDist = d; bestIdx = j; }
                    }
                    pts[i] = tp[bestIdx];
                    idx[i] = bestIdx;
                    dist[i] = bestDist;
                });
                var pBuf = HostResult(BufferKind.Points, outFormat, outOrigin, pts);
                var iBuf = CreateBuffer(BufferKind.Indices, ElementType.I32, q.Count, default, null, idx, Residency.Host);
                var dBuf = CreateBuffer(BufferKind.Scalars, ElementType.F64, q.Count, default, null, dist, Residency.Host);
                return new NearestResult(pBuf, iBuf, dBuf);
            }

            // Streaming: one huge host-resident query set vs small targets; queries never fully on device.
            if (q.Residency == Residency.Host && q.Count > StreamQueryThreshold && t.Count <= StreamTargetLimit)
            {
                return RunDevice(scope => StreamClosestPoints(scope, q, t, useBrute, outFormat, outOrigin), t);
            }

            return RunDevice(scope =>
            {
                var qv = DeviceV3(scope, q);
                var tv = DeviceV3(scope, t);
                var grid = useBrute ? null : BuildGrid(scope, tv, t.Count);

                DeviceLease<int> iLease = null;
                DeviceLease<double> dLease = null;
                try
                {
                    iLease = Rent<int>(q.Count);
                    dLease = Rent<double>(q.Count);
                    var il = iLease; var dl = dLease;
                    var pBuf = DeviceVectors(scope, BufferKind.Points, outFormat, outOrigin, q.Count,
                        pv => DeviceNearest(qv, q.Count, tv, t.Count, grid, pv, il.View, dl.View));
                    try
                    {
                        var iBuf = Adopt(iLease, BufferKind.Indices, ElementType.I32, q.Count);
                        var dBuf = Adopt(dLease, BufferKind.Scalars, ElementType.F64, q.Count);
                        return new NearestResult(pBuf, iBuf, dBuf);
                    }
                    catch
                    {
                        pBuf.Dispose();
                        throw;
                    }
                }
                catch
                {
                    iLease?.Dispose();
                    dLease?.Dispose();
                    throw;
                }
            }, q, t);
        }

        private void DeviceNearest(
            ArrayView<V3> qv, int qCount, ArrayView<V3> tv, int tCount, NearestGrid grid,
            ArrayView<V3> outPoints, ArrayView<int> outIndices, ArrayView<double> outDistances)
        {
            if (grid == null)
            {
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, ArrayView<V3>, ArrayView<int>, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, ArrayView<V3>, ArrayView<int>, ArrayView<double>>)SpatialKernels.BruteClosestPointsKernel)
                    (new Index1D(qCount), qv, tv, tCount, outPoints, outIndices, outDistances);
            }
            else
            {
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<int>, ArrayView<int>, ArrayView<int>, V3, double, double, int, int, int, ArrayView<V3>, ArrayView<int>, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<int>, ArrayView<int>, ArrayView<int>, V3, double, double, int, int, int, ArrayView<V3>, ArrayView<int>, ArrayView<double>>)SpatialKernels.GridClosestPointsKernel)
                    (new Index1D(qCount), qv, tv, grid.SortedIndices, grid.CellStart, grid.CellEnd,
                     grid.Min, grid.CellSize, grid.CellSizeInv, grid.DimX, grid.DimY, grid.DimZ,
                     outPoints, outIndices, outDistances);
            }
        }

        private RadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense> _radixSortPairs;

        /// <summary>Uniform grid over the targets, built entirely on the device (bbox reduction, cell keys, radix sort, cell ranges).</summary>
        private NearestGrid BuildGrid(DeviceArena.Scope scope, ArrayView<V3> tv, int targetCount)
        {
            // Target bbox: 3 component reductions, then one 48-byte readback to size the grid on the host.
            var bbox = scope.Allocate<double>(6);
            var asDoubles = tv.Cast<double>();
            for (int c = 0; c < 3; c++)
                ReduceBounds(scope, asDoubles, targetCount, 3, c, bbox.SubView(2 * c, 2));
            var b = new double[6];
            _stagingRing.Download(bbox.Cast<byte>(), 0, b, 0, 6 * sizeof(double));
            Interlocked.Increment(ref _synchronizations);

            double extentX = Math.Max(1e-6, b[1] - b[0]);
            double extentY = Math.Max(1e-6, b[3] - b[2]);
            double extentZ = Math.Max(1e-6, b[5] - b[4]);
            double cellCountTarget = Math.Max(1.0, Math.Pow(targetCount, 1.0 / 3.0));
            double cellSize = Math.Max(extentX, Math.Max(extentY, extentZ)) / cellCountTarget;
            if (cellSize <= 1e-9) cellSize = 1.0;

            var g = new NearestGrid
            {
                Min = new V3(b[0], b[2], b[4]),
                CellSize = cellSize,
                CellSizeInv = 1.0 / cellSize,
            };
            g.DimX = Math.Max(1, (int)(extentX * g.CellSizeInv) + 1);
            g.DimY = Math.Max(1, (int)(extentY * g.CellSizeInv) + 1);
            g.DimZ = Math.Max(1, (int)(extentZ * g.CellSizeInv) + 1);
            int totalCells = checked(g.DimX * g.DimY * g.DimZ);

            var keys = scope.Allocate<int>(targetCount);
            g.SortedIndices = scope.Allocate<int>(targetCount);
            GetKernel<Action<Index1D, ArrayView<V3>, int, V3, double, int, int, int, ArrayView<int>, ArrayView<int>>>(
                (Action<Index1D, ArrayView<V3>, int, V3, double, int, int, int, ArrayView<int>, ArrayView<int>>)SpatialKernels.AssignGridCellsKernel)
                (new Index1D(targetCount), tv, targetCount, g.Min, g.CellSizeInv, g.DimX, g.DimY, g.DimZ, keys, g.SortedIndices);
            CountDispatch();

            // Device radix sort of (cell key, target index) pairs. Stable, so equal keys keep ascending index.
            if (_radixSortPairs == null)
                _radixSortPairs = _accelerator.CreateRadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense, AscendingInt32>();
            var tempSize = _accelerator.ComputeRadixSortPairsTempStorageSize<int, int, AscendingInt32>(targetCount);
            var temp = scope.Allocate<int>(tempSize);
            _radixSortPairs(_accelerator.DefaultStream, keys, g.SortedIndices, temp);
            CountDispatch();

            g.CellStart = scope.Allocate<int>(totalCells);
            g.CellEnd = scope.Allocate<int>(totalCells);
            g.CellStart.MemSet(_accelerator.DefaultStream, 0xFF); // -1 = empty cell
            g.CellEnd.MemSet(_accelerator.DefaultStream, 0xFF);
            GetKernel<Action<Index1D, ArrayView<int>, int, ArrayView<int>, ArrayView<int>>>(
                (Action<Index1D, ArrayView<int>, int, ArrayView<int>, ArrayView<int>>)SpatialKernels.FindCellBoundsKernel)
                (new Index1D(targetCount), keys, targetCount, g.CellStart, g.CellEnd);
            CountDispatch();
            return g;
        }

        private NearestResult StreamClosestPoints(DeviceArena.Scope scope, RuntimeGpuBuffer q, RuntimeGpuBuffer t, bool useBrute, ElementType outFormat, V3 outOrigin)
        {
            var qPayload = q.Storage.ReadPayload(out _);
            int qElem = ElementTypes.SizeOf(q.ElementType);
            int outElem = ElementTypes.SizeOf(outFormat);

            var tv = DeviceV3(scope, t);
            var grid = useBrute ? null : BuildGrid(scope, tv, t.Count);

            int chunk = (int)Math.Max(1, Math.Min(q.Count, _stagingRing.SlotBytes / Math.Max(qElem, outElem)));
            var rawQ = scope.Allocate<byte>((long)chunk * qElem);
            var qWide = q.ElementType == ElementType.V3F ? scope.Allocate<V3>(chunk) : default;
            var pv = scope.Allocate<V3>(chunk);
            var pNarrow = outFormat == ElementType.V3F ? scope.Allocate<V3F>(chunk) : default;
            var iv = scope.Allocate<int>(chunk);
            var dv = scope.Allocate<double>(chunk);

            var outPts = NewPayload(outFormat, q.Count);
            var outIdx = new int[q.Count];
            var outDist = new double[q.Count];

            for (int offset = 0; offset < q.Count; offset += chunk)
            {
                int n = Math.Min(chunk, q.Count - offset);
                _stagingRing.Upload(qPayload, (long)offset * qElem, rawQ, 0, (long)n * qElem);
                _stagingRing.FlushPending();

                ArrayView<V3> qv;
                if (q.ElementType == ElementType.V3F)
                {
                    GetKernel<Action<Index1D, ArrayView<V3F>, V3, ArrayView<V3>>>(
                        (Action<Index1D, ArrayView<V3F>, V3, ArrayView<V3>>)VectorKernels.WidenV3FKernel)
                        (new Index1D(n), rawQ.Cast<V3F>().SubView(0, n), q.Origin, qWide);
                    CountDispatch();
                    qv = qWide.SubView(0, n);
                }
                else
                {
                    qv = rawQ.Cast<V3>().SubView(0, n);
                }

                DeviceNearest(qv, n, tv, t.Count, grid, pv, iv, dv);
                CountDispatch();

                ArrayView<byte> pBytes;
                if (outFormat == ElementType.V3F)
                {
                    GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3F>, V3>>(
                        (Action<Index1D, ArrayView<V3>, ArrayView<V3F>, V3>)VectorKernels.NarrowV3Kernel)
                        (new Index1D(n), pv, pNarrow, outOrigin);
                    CountDispatch();
                    pBytes = pNarrow.Cast<byte>();
                }
                else
                {
                    pBytes = pv.Cast<byte>();
                }

                _stagingRing.Download(pBytes, 0, outPts, (long)offset * outElem, (long)n * outElem);
                _stagingRing.Download(iv.Cast<byte>(), 0, outIdx, (long)offset * sizeof(int), (long)n * sizeof(int));
                _stagingRing.Download(dv.Cast<byte>(), 0, outDist, (long)offset * sizeof(double), (long)n * sizeof(double));
                Interlocked.Increment(ref _synchronizations);
            }

            return new NearestResult(
                CreateBuffer(BufferKind.Points, outFormat, q.Count, outOrigin, null, outPts, Residency.Host),
                CreateBuffer(BufferKind.Indices, ElementType.I32, q.Count, default, null, outIdx, Residency.Host),
                CreateBuffer(BufferKind.Scalars, ElementType.F64, q.Count, default, null, outDist, Residency.Host));
        }

        public GpuBuffer Distance(GpuBuffer a, GpuBuffer b)
        {
            var ba = ValidateBuffer(a, BufferKind.Points);
            var bb = ValidateBuffer(b, BufferKind.Points);

            if (ba.Count == 0 || bb.Count == 0)
                return CreateEmptyBuffer(BufferKind.Scalars, ElementType.F64);

            int count = Math.Max(ba.Count, bb.Count);

            if (CanRunOnHost(count, ba, bb))
            {
                var pa = HostV3(ba);
                var pb = HostV3(bb);
                var r = new double[count];
                HostFor(count, i => r[i] = VectorKernels.DistanceElement(pa[Rl(i, pa.Length)], pb[Rl(i, pb.Length)]));
                return HostResult(BufferKind.Scalars, ElementType.F64, default, r);
            }

            return RunDevice(scope => DeviceScalars(scope, ElementType.F64, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<double>>)VectorKernels.DistanceKernel)
                    (new Index1D(count), DeviceV3(scope, ba), DeviceV3(scope, bb), ba.Count, bb.Count, o)), ba, bb);
        }

        public GpuBuffer VectorBetween(GpuBuffer a, GpuBuffer b)
        {
            var ba = ValidateBuffer(a, BufferKind.Points);
            var bb = ValidateBuffer(b, BufferKind.Points);
            var format = WidestV3(ba, bb);
            // b - a = (oB - oA) + (offB - offA): offsets stay within the inputs' extents.
            V3 origin = format == ElementType.V3F ? VectorKernels.VectorBetweenElement(ba.Origin, bb.Origin) : default;

            if (ba.Count == 0 || bb.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = Math.Max(ba.Count, bb.Count);

            if (CanRunOnHost(count, ba, bb))
            {
                var pa = HostV3(ba);
                var pb = HostV3(bb);
                var r = new V3[count];
                HostFor(count, i => r[i] = VectorKernels.VectorBetweenElement(pa[Rl(i, pa.Length)], pb[Rl(i, pb.Length)]));
                return HostResult(BufferKind.Vectors, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<V3>>)VectorKernels.VectorBetweenKernel)
                    (new Index1D(count), DeviceV3(scope, ba), DeviceV3(scope, bb), ba.Count, bb.Count, o)), ba, bb);
        }

        public GpuBuffer Unitize(GpuBuffer vectors)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var format = bv.ElementType;
            V3 origin = default; // unit vectors: offsets from zero

            if (bv.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            if (CanRunOnHost(bv.Count, bv))
            {
                var v = HostV3(bv);
                var r = new V3[bv.Count];
                HostFor(bv.Count, i => r[i] = VectorKernels.UnitizeElement(v[i]));
                return HostResult(BufferKind.Vectors, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, origin, bv.Count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>>)VectorKernels.UnitizeKernel)
                    (new Index1D(bv.Count), DeviceV3(scope, bv), o)), bv);
        }

        public GpuBuffer Amplitude(GpuBuffer vectors, GpuBuffer magnitudes)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var bm = ValidateBuffer(magnitudes, BufferKind.Scalars);
            var format = bv.ElementType;
            V3 origin = default; // results scale unit vectors: offsets from zero

            if (bv.Count == 0 || bm.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = Math.Max(bv.Count, bm.Count);

            if (CanRunOnHost(count, bv, bm))
            {
                var v = HostV3(bv);
                var m = HostF64(bm);
                var r = new V3[count];
                HostFor(count, i => r[i] = VectorKernels.AmplitudeElement(v[Rl(i, v.Length)], m[Rl(i, m.Length)]));
                return HostResult(BufferKind.Vectors, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<double>, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<double>, int, int, ArrayView<V3>>)VectorKernels.AmplitudeKernel)
                    (new Index1D(count), DeviceV3(scope, bv), DeviceF64(scope, bm), bv.Count, bm.Count, o)), bv, bm);
        }

        public GpuBuffer Move(GpuBuffer points, GpuBuffer vectors)
        {
            var bp = ValidateBuffer(points, BufferKind.Points);
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var format = WidestV3(bp, bv);
            // p + v = (oP + oV) + (offP + offV)
            V3 origin = format == ElementType.V3F ? VectorKernels.MoveElement(bp.Origin, bv.Origin) : default;

            if (bp.Count == 0 || bv.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format);

            int count = Math.Max(bp.Count, bv.Count);

            if (CanRunOnHost(count, bp, bv))
            {
                var p = HostV3(bp);
                var v = HostV3(bv);
                var r = new V3[count];
                HostFor(count, i => r[i] = VectorKernels.MoveElement(p[Rl(i, p.Length)], v[Rl(i, v.Length)]));
                return HostResult(BufferKind.Points, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<V3>>)VectorKernels.MoveKernel)
                    (new Index1D(count), DeviceV3(scope, bp), DeviceV3(scope, bv), bp.Count, bv.Count, o)), bp, bv);
        }

        public GpuBuffer ScalarMath(GpuBuffer a, GpuBuffer b, ScalarOperation operation)
        {
            if (!Enum.IsDefined(typeof(ScalarOperation), operation))
                throw new ArgumentOutOfRangeException(nameof(operation), $"Invalid ScalarOperation: {operation}");

            var ba = ValidateBuffer(a, BufferKind.Scalars);
            var bb = ValidateBuffer(b, BufferKind.Scalars);
            var format = ScalarOut(ba, bb);

            if (ba.Count == 0 || bb.Count == 0)
                return CreateEmptyBuffer(BufferKind.Scalars, format);

            int count = Math.Max(ba.Count, bb.Count);
            int op = (int)operation;

            if (CanRunOnHost(count, ba, bb))
            {
                var sa = HostF64(ba);
                var sb = HostF64(bb);
                var r = new double[count];
                HostFor(count, i => r[i] = ScalarKernels.ScalarMathElement(sa[Rl(i, sa.Length)], sb[Rl(i, sb.Length)], op));
                return HostResult(BufferKind.Scalars, format, default, r);
            }

            return RunDevice(scope => DeviceScalars(scope, format, count, o =>
                GetKernel<Action<Index1D, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<double>>)ScalarKernels.ScalarMathKernel)
                    (new Index1D(count), DeviceF64(scope, ba), DeviceF64(scope, bb), ba.Count, bb.Count, op, o)), ba, bb);
        }

        public GpuBuffer Bounds(GpuBuffer scalars)
        {
            var bs = ValidateBuffer(scalars, BufferKind.Scalars);
            var format = bs.ElementType; // min/max are input values: F32 in, F32 out is exact

            if (bs.Count == 0)
                return HostResult(BufferKind.Scalars, format, default, new[] { double.NaN, double.NaN });

            if (CanRunOnHost(bs.Count, bs))
            {
                // Same rule as ReductionKernels.SerialBoundsKernel (the v1 oracle).
                var vals = HostF64(bs);
                double min = double.PositiveInfinity;
                double max = double.NegativeInfinity;
                for (int i = 0; i < vals.Length; i++)
                {
                    double v = vals[i];
                    if (!double.IsNaN(v))
                    {
                        if (v < min) min = v;
                        if (v > max) max = v;
                    }
                }
                if (double.IsPositiveInfinity(min)) min = double.NaN;
                if (double.IsNegativeInfinity(max)) max = double.NaN;
                return HostResult(BufferKind.Scalars, format, default, new[] { min, max });
            }

            return RunDevice(scope => DeviceScalars(scope, format, 2, o =>
                ReduceBounds(scope, DeviceF64(scope, bs), bs.Count, 1, 0, o)), bs);
        }

        /// <summary>
        /// Parallel bounds of input[i * stride + offset], i in [0, count), into output[0..1].
        /// Pass 1 runs up to 64K threads grid-stride; partial levels shrink 256x until one thread finalizes.
        /// </summary>
        private void ReduceBounds(DeviceArena.Scope scope, ArrayView<double> input, int count, int stride, int offset, ArrayView<double> output)
        {
            var pass1 = GetKernel<Action<Index1D, ArrayView<double>, int, int, int, int, ArrayView<double>, ArrayView<double>>>(
                (Action<Index1D, ArrayView<double>, int, int, int, int, ArrayView<double>, ArrayView<double>>)ReductionKernels.BoundsPass1Kernel);
            var reduce = GetKernel<Action<Index1D, ArrayView<double>, ArrayView<double>, int, int, ArrayView<double>, ArrayView<double>, int>>(
                (Action<Index1D, ArrayView<double>, ArrayView<double>, int, int, ArrayView<double>, ArrayView<double>, int>)ReductionKernels.BoundsReducePartialsKernel);

            int threads = Math.Min(count, 1 << 16);
            var mins = scope.Allocate<double>(threads);
            var maxs = scope.Allocate<double>(threads);
            pass1(new Index1D(threads), input, count, stride, offset, threads, mins, maxs);
            CountDispatch();

            int len = threads;
            while (len > 1)
            {
                int next = len > 256 ? (len + 255) / 256 : 1;
                bool last = next == 1;
                var nMins = last ? output.SubView(0, 1) : scope.Allocate<double>(next);
                var nMaxs = last ? output.SubView(1, 1) : scope.Allocate<double>(next);
                reduce(new Index1D(next), mins, maxs, len, next, nMins, nMaxs, last ? 1 : 0);
                CountDispatch();
                mins = nMins; maxs = nMaxs; len = next;
            }

            if (threads == 1)
            {
                // Single partial: still apply the finalize rule.
                reduce(new Index1D(1), mins, maxs, 1, 1, output.SubView(0, 1), output.SubView(1, 1), 1);
                CountDispatch();
            }
        }

        public GpuBuffer Remap(GpuBuffer scalars, GpuBuffer sourceBounds, double targetMin, double targetMax, bool clamp)
        {
            var bs = ValidateBuffer(scalars, BufferKind.Scalars);
            var bb = ValidateBuffer(sourceBounds, BufferKind.Scalars);

            if (bb.Count != 2)
                throw new ArgumentException("sourceBounds buffer must contain exactly 2 values [min, max].", nameof(sourceBounds));

            var format = ScalarOut(bs, bb);
            if (bs.Count == 0)
                return CreateEmptyBuffer(BufferKind.Scalars, format);

            if (CanRunOnHost(bs.Count, bs, bb))
            {
                var s = HostF64(bs);
                var bnds = HostF64(bb);
                var r = new double[bs.Count];
                HostFor(bs.Count, i => r[i] = ScalarKernels.RemapElement(s[i], bnds[0], bnds[1], targetMin, targetMax, clamp));
                return HostResult(BufferKind.Scalars, format, default, r);
            }

            return RunDevice(scope => DeviceScalars(scope, format, bs.Count, o =>
                GetKernel<Action<Index1D, ArrayView<double>, ArrayView<double>, double, double, int, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<double>, ArrayView<double>, double, double, int, ArrayView<double>>)ScalarKernels.RemapKernel)
                    (new Index1D(bs.Count), DeviceF64(scope, bs), DeviceF64(scope, bb), targetMin, targetMax, clamp ? 1 : 0, o)), bs, bb);
        }

        public GpuBuffer Map(GpuBuffer scalars, MapFunction function)
        {
            if (!Enum.IsDefined(typeof(MapFunction), function))
                throw new ArgumentOutOfRangeException(nameof(function), $"Invalid MapFunction: {function}");

            var bs = ValidateBuffer(scalars, BufferKind.Scalars);
            var format = bs.ElementType;
            int fn = (int)function;

            if (bs.Count == 0)
                return CreateEmptyBuffer(BufferKind.Scalars, format);

            if (CanRunOnHost(bs.Count, bs))
            {
                var s = HostF64(bs);
                var r = new double[bs.Count];
                HostFor(bs.Count, i => r[i] = ScalarKernels.MapElement(s[i], fn));
                return HostResult(BufferKind.Scalars, format, default, r);
            }

            return RunDevice(scope => DeviceScalars(scope, format, bs.Count, o =>
                GetKernel<Action<Index1D, ArrayView<double>, int, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<double>, int, ArrayView<double>>)ScalarKernels.MapKernel)
                    (new Index1D(bs.Count), DeviceF64(scope, bs), fn, o)), bs);
        }

        public GpuBuffer ColourRamp(GpuBuffer scalars, uint lowArgb, uint highArgb)
        {
            var bs = ValidateBuffer(scalars, BufferKind.Scalars);
            if (bs.Count == 0)
                return CreateEmptyBuffer(BufferKind.Colours, ElementType.U32);

            if (CanRunOnHost(bs.Count, bs))
            {
                var s = HostF64(bs);
                var r = new uint[bs.Count];
                HostFor(bs.Count, i => r[i] = ColourKernels.ColourRampElement(s[i], lowArgb, highArgb));
                return HostResult(BufferKind.Colours, ElementType.U32, default, r);
            }

            return RunDevice(scope =>
            {
                var input = DeviceF64(scope, bs);
                var lease = Rent<uint>(bs.Count);
                try
                {
                    GetKernel<Action<Index1D, ArrayView<double>, uint, uint, ArrayView<uint>>>(
                        (Action<Index1D, ArrayView<double>, uint, uint, ArrayView<uint>>)ColourKernels.ColourRampKernel)
                        (new Index1D(bs.Count), input, lowArgb, highArgb, lease.View);
                    CountDispatch();
                    return Adopt(lease, BufferKind.Colours, ElementType.U32, bs.Count);
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }, bs);
        }

        public GpuBuffer RepeatColours(GpuBuffer colours, int count)
        {
            var bc = ValidateBuffer(colours, BufferKind.Colours);

            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative.");

            if (count == 0)
                return CreateEmptyBuffer(BufferKind.Colours, ElementType.U32);

            if (bc.Count == 0)
                throw new InvalidOperationException("Empty input with requested nonzero output is an error.");

            if (CanRunOnHost(count, bc))
            {
                var c = (uint[])bc.Storage.ReadPayload(out _);
                var r = new uint[count];
                HostFor(count, i => r[i] = c[i % c.Length]);
                return HostResult(BufferKind.Colours, ElementType.U32, default, r);
            }

            return RunDevice(scope =>
            {
                var lease = Rent<uint>(count);
                try
                {
                    GetKernel<Action<Index1D, ArrayView<uint>, int, ArrayView<uint>>>(
                        (Action<Index1D, ArrayView<uint>, int, ArrayView<uint>>)ColourKernels.RepeatColoursKernel)
                        (new Index1D(count), bc.Storage.GetArrayView<uint>(), bc.Count, lease.View);
                    CountDispatch();
                    return Adopt(lease, BufferKind.Colours, ElementType.U32, count);
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }, bc);
        }

        public void Synchronize()
        {
            CheckNotDisposed();
            _stagingRing?.SynchronizeAll();
            _accelerator.Synchronize();
            Interlocked.Increment(ref _synchronizations);
        }

        private void ResetInternal()
        {
            Interlocked.Increment(ref _generation);

            List<DeviceBufferStorage> storagesToDispose;
            lock (_storageLock)
            {
                storagesToDispose = new List<DeviceBufferStorage>(_activeStorages);
                _activeStorages.Clear();
            }

            foreach (var storage in storagesToDispose)
            {
                storage.ForceDispose();
            }

            _memoryPool?.TrimPool();

            try
            {
                _stagingRing?.SynchronizeAll();
                _accelerator?.Synchronize();
                Interlocked.Increment(ref _synchronizations);
            }
            catch { }
        }

        public void Reset()
        {
            CheckNotDisposed();
            ResetInternal();
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;

            ResetInternal();

            try { _deviceArena?.Dispose(); } catch { }
            try { _stagingRing?.Dispose(); } catch { }
            try { _memoryPool?.Dispose(); } catch { }
            try { _accelerator?.Dispose(); } catch { }
            try { _context?.Dispose(); } catch { }
        }
    }
}
