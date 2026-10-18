using System;
using System.Drawing;
using GhostGPU.Core;
using Rhino.Geometry;

namespace GhostGPU.Grasshopper.Parameters
{
    internal static class MeshReadback
    {
        public static Mesh Materialize(IGpuSession session, GpuMeshPayload payload)
        {
            if (payload?.TopologyTemplate == null || payload.VertexPositions == null)
                throw new ArgumentException("GPU mesh payload has been disposed.");
            var vertices = session.DownloadPoints(payload.VertexPositions);
            var count = payload.TopologyTemplate.Vertices.Count;
            if (vertices.Length != count)
                throw new InvalidOperationException("Position count does not match mesh topology.");
            uint[] colours = null;
            if (payload.VertexColours != null)
            {
                colours = session.DownloadColours(payload.VertexColours);
                if (colours.Length != count)
                    throw new InvalidOperationException("Colour count does not match mesh topology.");
            }
            var mesh = payload.TopologyTemplate.DuplicateMesh();
            try
            {
                mesh.Vertices.UseDoublePrecisionVertices = true;
                for (var i = 0; i < count; i++)
                    mesh.Vertices.SetVertex(i, vertices[i].X, vertices[i].Y, vertices[i].Z);
                if (colours != null)
                {
                    mesh.VertexColors.Clear();
                    foreach (var argb in colours)
                        mesh.VertexColors.Add(Color.FromArgb(unchecked((int)argb)));
                }
                mesh.Normals.ComputeNormals();
                return mesh;
            }
            catch
            {
                mesh.Dispose();
                throw;
            }
        }
    }
}
