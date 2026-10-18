using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using GhostGPU.Core;
using ILGPU;
using ILGPU.Runtime;

namespace GhostGPU.Runtime.Memory
{
    internal sealed class PooledBlock
    {
        public MemoryBuffer1D<byte, Stride1D.Dense> Buffer { get; }
        public long SizeClassBytes { get; }
        public long LastUsedTimestamp { get; set; }

        public PooledBlock(MemoryBuffer1D<byte, Stride1D.Dense> buffer, long sizeClassBytes)
        {
            Buffer = buffer;
            SizeClassBytes = sizeClassBytes;
            LastUsedTimestamp = Stopwatch.GetTimestamp();
        }

        public void Dispose()
        {
            try
            {
                Buffer?.Dispose();
            }
            catch { }
        }
    }

    /// <summary>
    /// Power-of-two size-class device memory pool with LRU cache eviction and strict budget enforcement.
    /// Budget = Live (ActiveBytes) + Pooled (PooledBytes) <= MemoryLimitBytes.
    /// </summary>
    internal sealed class DeviceMemoryPool : IDisposable
    {
        private readonly GpuSession _session;
        private readonly Accelerator _accelerator;
        private readonly object _lock = new object();

        // SizeClassBytes -> List of free blocks
        private readonly Dictionary<long, List<PooledBlock>> _freeLists = new Dictionary<long, List<PooledBlock>>();
        // Flat list for LRU eviction: most recent at end
        private readonly List<PooledBlock> _lruList = new List<PooledBlock>();

        private long _pooledBytes;
        private long _activeBytes;
        private long _peakBytes;
        private long _poolHits;
        private long _poolMisses;
        private long _liveAllocations;
        private bool _isDisposed;

        public DeviceMemoryPool(GpuSession session, Accelerator accelerator)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        }

        public long PooledBytes => Interlocked.Read(ref _pooledBytes);
        public long ActiveBytes => Interlocked.Read(ref _activeBytes);
        public long PeakBytes => Interlocked.Read(ref _peakBytes);
        public long PoolHits => Interlocked.Read(ref _poolHits);
        public long PoolMisses => Interlocked.Read(ref _poolMisses);
        public long LiveAllocations => Interlocked.Read(ref _liveAllocations);

        public static long GetSizeClass(long bytes)
        {
            if (bytes <= 256) return 256;
            long v = bytes - 1;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            return v + 1;
        }

        public PooledBlock Rent(long byteSize)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(DeviceMemoryPool));

            long sizeClass = GetSizeClass(byteSize);
            PooledBlock block = null;
            long neededToSpill = 0;

            lock (_lock)
            {
                if (_freeLists.TryGetValue(sizeClass, out var list) && list.Count > 0)
                {
                    // Reuse existing pooled block (Pool Hit)
                    int lastIdx = list.Count - 1;
                    block = list[lastIdx];
                    list.RemoveAt(lastIdx);
                    _lruList.Remove(block);

                    _pooledBytes -= sizeClass;
                    _activeBytes += sizeClass;
                    Interlocked.Increment(ref _poolHits);
                    Interlocked.Increment(ref _liveAllocations);
                    UpdatePeak_Locked();
                    return block;
                }

                // Pool Miss
                Interlocked.Increment(ref _poolMisses);

                long limit = _session.OptionsInternal.MemoryLimitBytes;
                if (_activeBytes + _pooledBytes + sizeClass > limit)
                {
                    // 1. Trim pool first to recover budget
                    TrimPool_Locked(limit - _activeBytes - sizeClass);

                    // 2. If still exceeding budget, calculate needed spill
                    if (_activeBytes + _pooledBytes + sizeClass > limit)
                    {
                        neededToSpill = (_activeBytes + _pooledBytes + sizeClass) - limit;
                    }
                }

                if (neededToSpill == 0)
                {
                    // Allocate fresh block from accelerator
                    var mem = _accelerator.Allocate1D<byte>(sizeClass);
                    block = new PooledBlock(mem, sizeClass);
                    _activeBytes += sizeClass;
                    Interlocked.Increment(ref _liveAllocations);
                    UpdatePeak_Locked();
                    return block;
                }
            }

            // Outside _lock: Request spill to host.
            // Spilling acquires storage._lock, so releasing _lock here prevents deadlock with Promote (which holds storage._lock and calls Rent).
            _session.RequestSpill(neededToSpill);

            lock (_lock)
            {
                long limit = _session.OptionsInternal.MemoryLimitBytes;
                TrimPool_Locked(limit - _activeBytes - sizeClass);
                if (_activeBytes + _pooledBytes + sizeClass > limit)
                {
                    throw new InvalidOperationException(
                        $"Exceeded session memory limit of {limit} bytes. " +
                        $"Active: {_activeBytes} bytes, Pooled: {_pooledBytes} bytes, Requested: {sizeClass} bytes ({byteSize} bytes unaligned).");
                }

                // Fresh block after spill
                var mem = _accelerator.Allocate1D<byte>(sizeClass);
                block = new PooledBlock(mem, sizeClass);
                _activeBytes += sizeClass;
                Interlocked.Increment(ref _liveAllocations);
                UpdatePeak_Locked();
                return block;
            }
        }

        public void Return(PooledBlock block)
        {
            if (block == null) return;

            lock (_lock)
            {
                _activeBytes -= block.SizeClassBytes;
                Interlocked.Decrement(ref _liveAllocations);

                if (_isDisposed || _session.OptionsInternal.PoolCacheBytes <= 0)
                {
                    block.Dispose();
                    return;
                }

                long cacheCap = _session.OptionsInternal.PoolCacheBytes;

                // Evict LRU blocks if adding this block exceeds PoolCacheBytes
                while (_lruList.Count > 0 && (_pooledBytes + block.SizeClassBytes > cacheCap))
                {
                    var oldest = _lruList[0];
                    _lruList.RemoveAt(0);
                    if (_freeLists.TryGetValue(oldest.SizeClassBytes, out var oldList))
                    {
                        oldList.Remove(oldest);
                    }
                    _pooledBytes -= oldest.SizeClassBytes;
                    oldest.Dispose();
                }

                if (_pooledBytes + block.SizeClassBytes <= cacheCap)
                {
                    block.LastUsedTimestamp = Stopwatch.GetTimestamp();
                    if (!_freeLists.TryGetValue(block.SizeClassBytes, out var list))
                    {
                        list = new List<PooledBlock>();
                        _freeLists[block.SizeClassBytes] = list;
                    }
                    list.Add(block);
                    _lruList.Add(block);
                    _pooledBytes += block.SizeClassBytes;
                }
                else
                {
                    // Block itself exceeds PoolCacheBytes cap
                    block.Dispose();
                }
            }
        }

        public void TrimPool()
        {
            lock (_lock)
            {
                TrimPool_Locked(0);
            }
        }

        private void TrimPool_Locked(long targetPooledBytes)
        {
            if (targetPooledBytes < 0) targetPooledBytes = 0;
            while (_lruList.Count > 0 && _pooledBytes > targetPooledBytes)
            {
                var block = _lruList[0];
                _lruList.RemoveAt(0);
                if (_freeLists.TryGetValue(block.SizeClassBytes, out var list))
                {
                    list.Remove(block);
                }
                _pooledBytes -= block.SizeClassBytes;
                block.Dispose();
            }
        }

        private void UpdatePeak_Locked()
        {
            long current = _activeBytes + _pooledBytes;
            if (current > _peakBytes)
            {
                _peakBytes = current;
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_isDisposed) return;
                _isDisposed = true;

                foreach (var block in _lruList)
                {
                    block.Dispose();
                }
                _lruList.Clear();
                _freeLists.Clear();
                _pooledBytes = 0;
            }
        }
    }
}
