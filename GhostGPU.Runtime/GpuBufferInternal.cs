using System;
using System.Diagnostics;
using System.Threading;
using GhostGPU.Core;
using GhostGPU.Runtime.Memory;
using ILGPU;
using ILGPU.Runtime;

namespace GhostGPU.Runtime
{
    /// <summary>
    /// Tracks underlying buffer storage (Device or Host resident) and its reference count across buffer leases.
    /// Supports automatic promotion, spilling, and pooling.
    /// </summary>
    internal sealed class DeviceBufferStorage : IDisposable
    {
        private readonly GpuSession _session;
        private readonly BufferKind _kind;
        private readonly ElementType _elementType;
        private readonly int _count;
        private readonly long _byteSize;
        private readonly V3 _origin;
        private readonly long _generation;
        private readonly long _storageId;

        private PooledBlock _pooledBlock;
        private MemoryBuffer _deviceBuffer;
        private Array _hostPayload;
        private Residency _residency;

        private int _refCount;
        private bool _isDisposed;
        private long _lastAccessTimestamp;
        private int _inUseCount;
        private readonly object _lock = new object();

        public DeviceBufferStorage(
            GpuSession session,
            BufferKind kind,
            ElementType elementType,
            int count,
            V3 origin,
            PooledBlock pooledBlock,
            MemoryBuffer deviceBuffer,
            Array hostPayload,
            Residency residency,
            long generation,
            long storageId)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _kind = kind;
            _elementType = elementType;
            _count = count;
            _byteSize = (long)count * ElementTypes.SizeOf(elementType);
            _origin = origin;
            _pooledBlock = pooledBlock;
            _deviceBuffer = deviceBuffer ?? pooledBlock?.Buffer;
            _hostPayload = hostPayload;
            _residency = residency;
            _generation = generation;
            _storageId = storageId;
            _refCount = 1;
            _lastAccessTimestamp = Stopwatch.GetTimestamp();

            if (_residency == Residency.Host)
            {
                _session.AddHostResidentBytes(_byteSize);
            }
        }

        public GpuSession Session => _session;
        public BufferKind Kind => _kind;
        public ElementType ElementType => _elementType;
        public int Count => _count;
        public long ByteSize => _byteSize;
        public V3 Origin => _origin;
        public long Generation => _generation;
        public long StorageId => _storageId;
        public long LastAccessTimestamp => Volatile.Read(ref _lastAccessTimestamp);

        public Residency Residency
        {
            get
            {
                lock (_lock)
                {
                    return _residency;
                }
            }
        }

        public Array HostPayload
        {
            get
            {
                lock (_lock)
                {
                    return _hostPayload;
                }
            }
        }

        public MemoryBuffer DeviceBuffer
        {
            get
            {
                lock (_lock)
                {
                    return _deviceBuffer;
                }
            }
        }

        public bool IsDisposed
        {
            get
            {
                lock (_lock)
                {
                    return _isDisposed;
                }
            }
        }

        public bool IsInUse
        {
            get
            {
                lock (_lock)
                {
                    return _inUseCount > 0;
                }
            }
        }

        public void MarkInUse()
        {
            lock (_lock)
            {
                _inUseCount++;
                _lastAccessTimestamp = Stopwatch.GetTimestamp();
            }
        }

        public void UnmarkInUse()
        {
            lock (_lock)
            {
                if (_inUseCount > 0)
                    _inUseCount--;
            }
        }

        public ArrayView<T> GetArrayView<T>() where T : unmanaged
        {
            lock (_lock)
            {
                if (_isDisposed)
                    throw new ObjectDisposedException(nameof(DeviceBufferStorage), "Buffer storage has been disposed.");

                Touch_Locked();

                if (_residency != Residency.Device)
                {
                    PromoteToDevice_Locked();
                }

                if (_deviceBuffer == null || _count == 0)
                    return default;

                var byteBuffer = (MemoryBuffer1D<byte, Stride1D.Dense>)_deviceBuffer;
                var byteView = (ArrayView<byte>)byteBuffer.View;
                long countT = _byteSize / System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
                return byteView.Cast<T>().SubView(0, countT);
            }
        }

        public void EnsureDeviceResident()
        {
            lock (_lock)
            {
                if (_isDisposed)
                    throw new ObjectDisposedException(nameof(DeviceBufferStorage));

                Touch_Locked();

                if (_residency != Residency.Device)
                {
                    PromoteToDevice_Locked();
                }
            }
        }

        private void PromoteToDevice_Locked()
        {
            if (_count == 0)
            {
                _residency = Residency.Device;
                return;
            }

            if (_hostPayload == null)
                throw new InvalidOperationException("Cannot promote to device without host payload.");

            // Rent device block from session pool
            var block = _session.MemoryPool.Rent(_byteSize);
            _pooledBlock = block;
            _deviceBuffer = block.Buffer;

            // Upload from host payload
            _session.UploadHostPayloadToDevice(_hostPayload, _elementType, _count, _pooledBlock.Buffer.View);

            _residency = Residency.Device;
            _session.AddHostResidentBytes(-_byteSize);
        }

        /// <summary>
        /// Typed payload matching ElementType. Returns the cached host payload when one exists
        /// (shared = true, caller must not mutate it), else downloads a fresh copy.
        /// </summary>
        public Array ReadPayload(out bool shared)
        {
            lock (_lock)
            {
                if (_isDisposed)
                    throw new ObjectDisposedException(nameof(DeviceBufferStorage), "Buffer storage has been disposed.");
                Touch_Locked();
                if (_hostPayload != null)
                {
                    shared = true;
                    return _hostPayload;
                }
                shared = false;
                return _session.DownloadDevicePayloadToHost(_deviceBuffer, _elementType, _count);
            }
        }

        public bool TrySpillToHost()
        {
            // Never block here: the caller may hold another storage's lock (Promote -> pool Rent -> spill).
            // A storage whose lock is busy is being promoted/read right now, so it is not a spill candidate anyway.
            // This makes spill lock-order free: no thread ever waits on a storage lock while holding one.
            if (!Monitor.TryEnter(_lock))
                return false;
            try
            {
                if (_isDisposed || _residency != Residency.Device || _inUseCount > 0 || _count == 0)
                    return false;

                // Download device content to host if not already cached
                if (_hostPayload == null)
                {
                    _hostPayload = _session.DownloadDevicePayloadToHost(_deviceBuffer, _elementType, _count);
                    _session.AddSpilledBytes(_byteSize);
                }

                if (_pooledBlock != null)
                {
                    _session.MemoryPool.Return(_pooledBlock);
                    _pooledBlock = null;
                    _deviceBuffer = null;
                }
                else if (_deviceBuffer != null)
                {
                    try { _deviceBuffer.Dispose(); } catch { }
                    _deviceBuffer = null;
                }

                _residency = Residency.Host;
                _session.AddHostResidentBytes(_byteSize);
                return true;
            }
            finally
            {
                Monitor.Exit(_lock);
            }
        }

        private void Touch_Locked()
        {
            _lastAccessTimestamp = Stopwatch.GetTimestamp();
        }

        public void AddRef()
        {
            lock (_lock)
            {
                if (_isDisposed)
                    throw new ObjectDisposedException(nameof(DeviceBufferStorage), "Cannot retain a disposed buffer storage.");
                _refCount++;
                Touch_Locked();
            }
        }

        public void ReleaseRef()
        {
            bool doDispose = false;
            lock (_lock)
            {
                if (_isDisposed)
                    return;
                _refCount--;
                if (_refCount <= 0)
                {
                    _isDisposed = true;
                    doDispose = true;
                }
            }

            if (doDispose)
            {
                DisposeInternal();
            }
        }

        public void ForceDispose()
        {
            bool doDispose = false;
            lock (_lock)
            {
                if (_isDisposed)
                    return;
                _isDisposed = true;
                _refCount = 0;
                doDispose = true;
            }

            if (doDispose)
            {
                DisposeInternal();
            }
        }

        private void DisposeInternal()
        {
            try
            {
                if (_residency == Residency.Host)
                {
                    _session.AddHostResidentBytes(-_byteSize);
                }

                if (_pooledBlock != null)
                {
                    _session.MemoryPool?.Return(_pooledBlock);
                    _pooledBlock = null;
                    _deviceBuffer = null;
                }
                else
                {
                    _deviceBuffer?.Dispose();
                    _deviceBuffer = null;
                }
            }
            catch { }
            finally
            {
                _session.OnStorageDisposed(this);
            }
        }

        public void Dispose()
        {
            ReleaseRef();
        }
    }

    /// <summary>
    /// Immutable lease to device or host buffer storage. Retain creates an independent lease.
    /// </summary>
    internal sealed class RuntimeGpuBuffer : GpuBuffer
    {
        private readonly GpuSession _session;
        private readonly DeviceBufferStorage _storage;
        private readonly long _id;
        private int _disposed;

        public RuntimeGpuBuffer(GpuSession session, DeviceBufferStorage storage, long id)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _storage = storage ?? throw new ArgumentNullException(nameof(storage));
            _id = id;
        }

        public override Guid SessionId => _session.SessionId;
        public override long Id => _id;
        public override BufferKind Kind => _storage.Kind;
        public override int Count => _storage.Count;
        public override ElementType ElementType => _storage.ElementType;
        public override Residency Residency => _storage.Residency;
        public V3 Origin => _storage.Origin;

        public override bool IsDisposed
        {
            get
            {
                if (Volatile.Read(ref _disposed) != 0)
                    return true;
                if (_storage.IsDisposed)
                    return true;
                if (_session.IsDisposed)
                    return true;
                if (_storage.Generation != _session.Generation)
                    return true;
                return false;
            }
        }

        internal DeviceBufferStorage Storage => _storage;

        public override GpuBuffer Retain()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(RuntimeGpuBuffer), "Cannot retain a disposed buffer.");

            _storage.AddRef();
            return new RuntimeGpuBuffer(_session, _storage, _session.NextBufferId());
        }

        ~RuntimeGpuBuffer()
        {
            DisposeInternal();
        }

        public override void Dispose()
        {
            DisposeInternal();
            GC.SuppressFinalize(this);
        }

        private void DisposeInternal()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            _storage.ReleaseRef();
        }
    }
}
