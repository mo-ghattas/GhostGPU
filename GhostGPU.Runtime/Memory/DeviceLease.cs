using System;
using ILGPU;

namespace GhostGPU.Runtime
{
    /// <summary>
    /// Rented device memory lease allocated from DeviceMemoryPool.
    /// Can be used with external kernels or adopted into an immutable GpuBuffer.
    /// </summary>
    public sealed class DeviceLease<T> : IDisposable where T : unmanaged
    {
        public ArrayView<T> View;
        public long Length;

        internal Memory.PooledBlock Block { get; }
        internal GpuSession Session { get; }
        internal bool IsAdopted { get; set; }
        private bool _isDisposed;

        internal DeviceLease(GpuSession session, Memory.PooledBlock block, ArrayView<T> view, long length)
        {
            Session = session;
            Block = block;
            View = view;
            Length = length;
        }

        public bool IsDisposed => _isDisposed;

        public void Dispose()
        {
            if (_isDisposed)
                return;
            _isDisposed = true;

            if (!IsAdopted && Block != null)
            {
                Session?.MemoryPool?.Return(Block);
            }
        }
    }
}
