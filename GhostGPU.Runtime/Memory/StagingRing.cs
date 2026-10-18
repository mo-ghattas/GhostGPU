using System;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace GhostGPU.Runtime.Memory
{
    internal sealed class StagingSlot : IDisposable
    {
        public AcceleratorStream Stream { get; }
        public PageLockedArray1D<byte> PinnedArray { get; }
        /// <summary>An async copy was enqueued on this slot's stream and nobody has fenced it yet.</summary>
        public bool Pending;
        private bool _isDisposed;

        public StagingSlot(Accelerator accelerator, long slotBytes)
        {
            Stream = accelerator.CreateStream();
            PinnedArray = accelerator.AllocatePageLocked1D<byte>(slotBytes);
        }

        public void Fence()
        {
            if (!Pending) return;
            Stream.Synchronize();
            Pending = false;
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;

            try { Stream.Synchronize(); } catch { }
            try { Stream.Dispose(); } catch { }
            try { PinnedArray.Dispose(); } catch { }
        }
    }

    /// <summary>
    /// Page-locked host staging ring, one stream per slot, round-robin.
    /// Uploads are asynchronous: the managed source is copied once into a pinned slot and the
    /// slot's stream DMAs it to the device while the next chunk is being filled. Consumers of
    /// device data (kernels, DeviceView, downloads) call <see cref="FlushPending"/> first, which
    /// fences only slots that still have copies in flight. Transfers larger than a slot are chunked.
    /// </summary>
    internal sealed class StagingRing : IDisposable
    {
        private readonly GpuSession _session;
        private readonly Accelerator _accelerator;
        private readonly StagingSlot[] _slots;
        private readonly long _slotBytes;
        private int _currentSlot;
        private readonly object _lock = new object();
        private bool _isDisposed;

        public StagingRing(GpuSession session, Accelerator accelerator, int slotCount, long slotBytes)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            // Span-based slot copies index with int.
            _slotBytes = Math.Min(Math.Max(256, slotBytes), int.MaxValue);
            int count = Math.Max(1, slotCount);
            _slots = new StagingSlot[count];

            for (int i = 0; i < count; i++)
            {
                _slots[i] = new StagingSlot(accelerator, _slotBytes);
            }
        }

        public long SlotBytes => _slotBytes;

        /// <summary>Copies bytes [sourceOffset, +totalBytes) of a blittable managed array to the device view. Async.</summary>
        public void Upload(Array source, long sourceOffset, ArrayView<byte> targetView, long targetOffset, long totalBytes)
        {
            if (totalBytes <= 0) return;

            // The target may be a recycled pool block still read/written by kernels on the default stream.
            _accelerator.DefaultStream.Synchronize();

            lock (_lock)
            {
                var handle = GCHandle.Alloc(source, GCHandleType.Pinned);
                try
                {
                    IntPtr baseSrc = handle.AddrOfPinnedObject();
                    long done = 0;
                    while (done < totalBytes)
                    {
                        int chunk = (int)Math.Min(totalBytes - done, _slotBytes);
                        var slot = NextSlot_Locked();
                        slot.Fence(); // slot's pinned memory may still be the source of an in-flight copy

                        unsafe
                        {
                            new ReadOnlySpan<byte>((byte*)baseSrc + sourceOffset + done, chunk)
                                .CopyTo(slot.PinnedArray.Span.Slice(0, chunk));
                        }

                        slot.PinnedArray.ArrayView.SubView(0, chunk)
                            .CopyTo(slot.Stream, targetView.SubView(targetOffset + done, chunk));
                        slot.Pending = true;

                        _session.AddStagedBytes(chunk);
                        _session.AddUploadedBytes(chunk);
                        done += chunk;
                    }
                }
                finally
                {
                    handle.Free();
                }
            }
        }

        /// <summary>Copies device bytes into a blittable managed array at targetOffset bytes. Synchronous.</summary>
        public void Download(ArrayView<byte> sourceView, long sourceOffset, Array target, long targetOffset, long totalBytes)
        {
            if (totalBytes <= 0) return;

            _accelerator.DefaultStream.Synchronize();

            lock (_lock)
            {
                FlushPending_Locked();

                var handle = GCHandle.Alloc(target, GCHandleType.Pinned);
                try
                {
                    IntPtr baseDst = handle.AddrOfPinnedObject();
                    long done = 0;
                    while (done < totalBytes)
                    {
                        int chunk = (int)Math.Min(totalBytes - done, _slotBytes);
                        var slot = NextSlot_Locked();
                        slot.Fence();

                        sourceView.SubView(sourceOffset + done, chunk)
                            .CopyTo(slot.Stream, slot.PinnedArray.ArrayView.SubView(0, chunk));
                        slot.Stream.Synchronize();

                        unsafe
                        {
                            slot.PinnedArray.Span.Slice(0, chunk)
                                .CopyTo(new Span<byte>((byte*)baseDst + targetOffset + done, chunk));
                        }

                        _session.AddStagedBytes(chunk);
                        _session.AddDownloadedBytes(chunk);
                        done += chunk;
                    }
                }
                finally
                {
                    handle.Free();
                }
            }
        }

        /// <summary>Waits for in-flight uploads. Call before any default-stream kernel reads uploaded data.</summary>
        public void FlushPending()
        {
            lock (_lock)
            {
                FlushPending_Locked();
            }
        }

        private void FlushPending_Locked()
        {
            for (int i = 0; i < _slots.Length; i++)
                _slots[i].Fence();
        }

        private StagingSlot NextSlot_Locked()
        {
            var slot = _slots[_currentSlot];
            _currentSlot = (_currentSlot + 1) % _slots.Length;
            return slot;
        }

        public void SynchronizeAll()
        {
            lock (_lock)
            {
                for (int i = 0; i < _slots.Length; i++)
                {
                    _slots[i].Stream.Synchronize();
                    _slots[i].Pending = false;
                }
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_isDisposed) return;
                _isDisposed = true;

                for (int i = 0; i < _slots.Length; i++)
                {
                    _slots[i]?.Dispose();
                }
            }
        }
    }
}
