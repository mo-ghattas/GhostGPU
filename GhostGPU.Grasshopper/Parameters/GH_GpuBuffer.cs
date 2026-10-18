using System;
using GH_IO.Serialization;
using Grasshopper;
using Grasshopper.Kernel.Types;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Session;

namespace GhostGPU.Grasshopper.Parameters
{
    /// <summary>
    /// Typed Grasshopper Goo wrapping an immutable, resident GPU memory buffer lease.
    /// Duplication retains an independent lease. Handles are ephemeral across files.
    /// Includes finalizer to guarantee release of device leases across repeated Grasshopper solves.
    /// </summary>
    public class GH_GpuBuffer : GH_Goo<GpuBuffer>, IDisposable
    {
        private bool _disposed;

        public GH_GpuBuffer()
        {
        }

        public GH_GpuBuffer(GpuBuffer buffer)
        {
            m_value = buffer;
        }

        public GH_GpuBuffer(GH_GpuBuffer other)
        {
            m_value = other?.Value?.Retain();
        }

        ~GH_GpuBuffer()
        {
            Dispose(false);
        }

        public override bool IsValid => m_value != null && !m_value.IsDisposed;

        public override string IsValidWhyNot
        {
            get
            {
                if (m_value == null) return "GPU buffer reference is null.";
                if (m_value.IsDisposed) return "GPU buffer lease has been disposed.";
                return string.Empty;
            }
        }

        public override string TypeName => "GpuBuffer";
        public override string TypeDescription => "Resident GPU memory buffer handle";

        public override IGH_Goo Duplicate()
        {
            var retained = m_value?.Retain();
            if (retained == null) return new GH_GpuBuffer((GpuBuffer)null);
            var goo = new GH_GpuBuffer(retained);
            try
            {
                var doc = Instances.ActiveCanvas?.Document;
                if (doc != null)
                {
                    GpuSessionManager.RegisterTransientLease(doc, goo);
                }
            }
            catch
            {
                // Headless or background canvas without UI
            }
            return goo;
        }

        public override string ToString()
        {
            if (m_value == null) return "Null GPU Buffer";
            if (m_value.IsDisposed) return $"Disposed GPU Buffer [Kind={m_value.Kind}]";
            return $"GpuBuffer[{m_value.Kind}, Count={m_value.Count}, Id={m_value.Id}]";
        }

        public override bool CastTo<T>(ref T target)
        {
            if (typeof(T).IsAssignableFrom(typeof(GpuBuffer)))
            {
                target = (T)(object)m_value;
                return true;
            }
            if (typeof(T) == typeof(GH_GpuBuffer))
            {
                target = (T)(object)this;
                return true;
            }
            // Explicit boundary: do not silently download geometry without explicit Download component.
            return false;
        }

        public override bool CastFrom(object source)
        {
            if (source is GpuBuffer buffer)
            {
                var replacement = buffer.Retain();
                m_value?.Dispose();
                m_value = replacement;
                return true;
            }
            if (source is GH_GpuBuffer goo)
            {
                var replacement = goo.Value?.Retain();
                m_value?.Dispose();
                m_value = replacement;
                return true;
            }
            return false;
        }

        public override bool Write(GH_IWriter writer)
        {
            // Ephemeral handle contract: record metadata for inspection, but do not serialize device pointers.
            if (m_value != null)
            {
                writer.SetInt32("Kind", (int)m_value.Kind);
                writer.SetInt32("Count", m_value.Count);
                writer.SetInt64("BufferId", m_value.Id);
                writer.SetGuid("SessionId", m_value.SessionId);
            }
            return true;
        }

        public override bool Read(GH_IReader reader)
        {
            // Release any existing value before clearing
            m_value?.Dispose();
            m_value = null;
            return true;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                _disposed = true;
                try
                {
                    m_value?.Dispose();
                }
                catch
                {
                    // Ignore exceptions during finalization/cleanup
                }
                m_value = null;
            }
        }
    }
}
