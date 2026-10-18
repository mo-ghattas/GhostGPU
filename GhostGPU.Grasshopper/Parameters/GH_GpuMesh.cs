using System;
using GH_IO.Serialization;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GhostGPU.Core;

namespace GhostGPU.Grasshopper.Parameters
{
    /// <summary>
    /// Holds the host-side immutable topology template alongside resident GPU vertex position and optional colour leases.
    /// Preserves original quad/ngon topology, face normals, and texture coordinates without modification.
    /// Owns and deterministically disposes its host mesh template and device leases.
    /// </summary>
    public sealed class GpuMeshPayload : IDisposable
    {
        private bool _disposed;

        public Mesh TopologyTemplate { get; private set; }
        public GpuBuffer VertexPositions { get; private set; }
        public GpuBuffer VertexColours { get; private set; }

        public GpuMeshPayload(Mesh topologyTemplate, GpuBuffer vertexPositions, GpuBuffer vertexColours = null)
        {
            if (topologyTemplate == null) throw new ArgumentNullException(nameof(topologyTemplate));
            // Ensure double precision vertex storage
            topologyTemplate.Vertices.UseDoublePrecisionVertices = true;
            TopologyTemplate = topologyTemplate;

            VertexPositions = vertexPositions ?? throw new ArgumentNullException(nameof(vertexPositions));

            if (vertexPositions.Kind != BufferKind.Points || vertexPositions.IsDisposed)
                throw new ArgumentException("Vertex positions must be a live Points buffer.", nameof(vertexPositions));
            if (vertexPositions.Count != topologyTemplate.Vertices.Count)
                throw new ArgumentException("Position count must equal the topology vertex count.", nameof(vertexPositions));

            if (vertexColours != null && vertexColours.Kind != BufferKind.Colours)
                throw new ArgumentException("Vertex colours buffer must be Colours kind.", nameof(vertexColours));
            if (vertexColours != null && (vertexColours.IsDisposed || vertexColours.Count != vertexPositions.Count || vertexColours.SessionId != vertexPositions.SessionId))
                throw new ArgumentException("Vertex colours must match vertex count and session.", nameof(vertexColours));

            VertexColours = vertexColours;
        }

        ~GpuMeshPayload()
        {
            Dispose(false);
        }

        public GpuMeshPayload Retain()
        {
            return new GpuMeshPayload(TopologyTemplate.DuplicateMesh(), VertexPositions.Retain(), VertexColours?.Retain());
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                _disposed = true;
                try
                {
                    VertexPositions?.Dispose();
                    VertexPositions = null;

                    VertexColours?.Dispose();
                    VertexColours = null;

                    TopologyTemplate?.Dispose();
                    TopologyTemplate = null;
                }
                catch { }
            }
        }
    }

    /// <summary>
    /// Typed Grasshopper Goo wrapping a GpuMeshPayload.
    /// Includes finalizer to guarantee release of device leases across repeated Grasshopper solves.
    /// </summary>
    public class GH_GpuMesh : GH_Goo<GpuMeshPayload>, IDisposable
    {
        private bool _disposed;

        public GH_GpuMesh()
        {
        }

        public GH_GpuMesh(GpuMeshPayload payload)
        {
            m_value = payload;
        }

        public GH_GpuMesh(GH_GpuMesh other)
        {
            m_value = other?.Value?.Retain();
        }

        ~GH_GpuMesh()
        {
            Dispose(false);
        }

        public override bool IsValid => m_value != null && m_value.VertexPositions != null && !m_value.VertexPositions.IsDisposed;

        public override string IsValidWhyNot
        {
            get
            {
                if (m_value == null) return "GPU mesh payload is null.";
                if (m_value.TopologyTemplate == null) return "Mesh topology template is null.";
                if (m_value.VertexPositions == null || m_value.VertexPositions.IsDisposed) return "Vertex positions buffer is disposed.";
                return string.Empty;
            }
        }

        public override string TypeName => "GpuMesh";
        public override string TypeDescription => "Resident GPU mesh with host topology template";

        public override IGH_Goo Duplicate()
        {
            return new GH_GpuMesh(m_value?.Retain());
        }

        public override string ToString()
        {
            if (m_value == null) return "Null GPU Mesh";
            int vCount = m_value.TopologyTemplate?.Vertices.Count ?? 0;
            int fCount = m_value.TopologyTemplate?.Faces.Count ?? 0;
            bool hasCol = m_value.VertexColours != null && !m_value.VertexColours.IsDisposed;
            return $"GpuMesh[Vertices={vCount}, Faces={fCount}, Colours={(hasCol ? "Resident" : "None")}]";
        }

        public override bool CastTo<T>(ref T target)
        {
            if (typeof(T).IsAssignableFrom(typeof(GpuMeshPayload)))
            {
                target = (T)(object)m_value;
                return true;
            }
            if (typeof(T) == typeof(GH_GpuMesh))
            {
                target = (T)(object)this;
                return true;
            }
            // Explicit boundary: do not silently download mesh without explicit Download node.
            return false;
        }

        public override bool CastFrom(object source)
        {
            if (source is GpuMeshPayload payload)
            {
                var replacement = payload.Retain();
                m_value?.Dispose();
                m_value = replacement;
                return true;
            }
            if (source is GH_GpuMesh goo)
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
            if (m_value != null && m_value.TopologyTemplate != null)
            {
                writer.SetInt32("VertexCount", m_value.TopologyTemplate.Vertices.Count);
                writer.SetInt32("FaceCount", m_value.TopologyTemplate.Faces.Count);
                writer.SetBoolean("HasColours", m_value.VertexColours != null);
            }
            return true;
        }

        public override bool Read(GH_IReader reader)
        {
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
                catch { }
                m_value = null;
            }
        }
    }
}
