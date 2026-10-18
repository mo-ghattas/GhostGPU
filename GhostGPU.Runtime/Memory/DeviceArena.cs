using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using ILGPU;
using ILGPU.Runtime;

namespace GhostGPU.Runtime.Memory
{
    /// <summary>
    /// Bump allocator over one pooled device block for per-op temporaries (widened compact inputs,
    /// reduction partials, grid cells). Use through <see cref="Begin"/>: the scope owns the arena
    /// for one op and resets it on dispose. Blocks outgrown mid-op are retired until the scope ends,
    /// so views handed out earlier in the op stay valid. Reuse after reset is safe because every
    /// kernel runs on the default stream (in order) and uploads synchronize it before writing.
    /// </summary>
    internal sealed class DeviceArena : IDisposable
    {
        private readonly GpuSession _session;
        private readonly object _lock = new object();
        private readonly List<PooledBlock> _retired = new List<PooledBlock>();
        private PooledBlock _block;
        private long _offset;
        private bool _isDisposed;

        public DeviceArena(GpuSession session)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
        }

        /// <summary>Exclusive per-op scope. Limitation: one arena per session, device ops serialize on it; per-thread arenas if host-side concurrency matters.</summary>
        public Scope Begin()
        {
            Monitor.Enter(_lock);
            if (_isDisposed)
            {
                Monitor.Exit(_lock);
                throw new ObjectDisposedException(nameof(DeviceArena));
            }
            return new Scope(this);
        }

        private ArrayView<T> Allocate<T>(long count) where T : unmanaged
        {
            if (count <= 0)
                return default;

            long byteSize = count * Unsafe.SizeOf<T>();
            long alignedBytes = (byteSize + 255) & ~255L;

            if (_block == null || _offset + alignedBytes > _block.SizeClassBytes)
            {
                long want = alignedBytes;
                if (_block != null)
                {
                    want = Math.Max(want, _block.SizeClassBytes * 2);
                    _retired.Add(_block);
                }
                _block = null;
                _block = _session.MemoryPool.Rent(want);
                _offset = 0;
            }

            var byteView = (ArrayView<byte>)_block.Buffer.View;
            var view = byteView.SubView(_offset, byteSize).Cast<T>();
            _offset += alignedBytes;
            return view;
        }

        private void End()
        {
            try
            {
                // Return everything so budget/ActiveBytes reflect only live buffers between ops;
                // the next op gets the block back as a pool hit.
                if (_block != null) _retired.Add(_block);
                _block = null;
                foreach (var b in _retired)
                    _session.MemoryPool?.Return(b);
                _retired.Clear();
                _offset = 0;
            }
            finally
            {
                Monitor.Exit(_lock);
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_isDisposed) return;
                _isDisposed = true;
                foreach (var b in _retired)
                    _session.MemoryPool?.Return(b);
                _retired.Clear();
                if (_block != null)
                {
                    _session.MemoryPool?.Return(_block);
                    _block = null;
                }
            }
        }

        public readonly struct Scope : IDisposable
        {
            private readonly DeviceArena _arena;
            internal Scope(DeviceArena arena) { _arena = arena; }
            public ArrayView<T> Allocate<T>(long count) where T : unmanaged => _arena.Allocate<T>(count);
            public void Dispose() => _arena?.End();
        }
    }
}
