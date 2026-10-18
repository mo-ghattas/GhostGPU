using System;
using GhostGPU.Core;
using RhinoMesh = Rhino.Geometry.Mesh;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    /// <summary>
    /// Converts Rhino host mesh topology into immutable GhostGPU descriptors and CSR adjacency.
    /// Operates in O(V + F) time without dynamic heap allocations per vertex.
    /// </summary>
    public static class MeshTopologyAdapter
    {
        public static MeshTopologyDescriptor BuildTopologyDescriptor(RhinoMesh mesh)
        {
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));
            int vCount = mesh.Vertices.Count;
            int fCount = mesh.Faces.Count;

            var faces = new MeshFaceIndices[fCount];
            for (int i = 0; i < fCount; i++)
            {
                var f = mesh.Faces[i];
                faces[i] = f.IsTriangle
                    ? new MeshFaceIndices(f.A, f.B, f.C)
                    : new MeshFaceIndices(f.A, f.B, f.C, f.D);
            }

            var adjacency = BuildCSRAdjacency(vCount, faces);
            return MeshTopologyDescriptor.CreateValidated(vCount, faces, adjacency);
        }

        public static MeshVertexAdjacency BuildCSRAdjacency(int vertexCount, MeshFaceIndices[] faces)
        {
            if (vertexCount < 0) throw new ArgumentOutOfRangeException(nameof(vertexCount));
            if (faces == null) throw new ArgumentNullException(nameof(faces));

            if (vertexCount == 0 || faces.Length == 0)
            {
                return new MeshVertexAdjacency(vertexCount, new int[vertexCount + 1], Array.Empty<int>());
            }

            int[] counts = new int[vertexCount];
            for (int f = 0; f < faces.Length; f++)
            {
                var face = faces[f];
                if (face.A >= 0 && face.A < vertexCount) counts[face.A]++;
                if (face.B >= 0 && face.B < vertexCount) counts[face.B]++;
                if (face.C >= 0 && face.C < vertexCount) counts[face.C]++;
                if (face.IsQuad && face.D >= 0 && face.D < vertexCount) counts[face.D]++;
            }

            int[] offsets = new int[vertexCount + 1];
            offsets[0] = 0;
            for (int i = 0; i < vertexCount; i++)
            {
                offsets[i + 1] = offsets[i] + counts[i];
            }

            int total = offsets[vertexCount];
            int[] adjacentFaces = new int[total];
            int[] current = (int[])offsets.Clone();

            for (int f = 0; f < faces.Length; f++)
            {
                var face = faces[f];
                if (face.A >= 0 && face.A < vertexCount) adjacentFaces[current[face.A]++] = f;
                if (face.B >= 0 && face.B < vertexCount) adjacentFaces[current[face.B]++] = f;
                if (face.C >= 0 && face.C < vertexCount) adjacentFaces[current[face.C]++] = f;
                if (face.IsQuad && face.D >= 0 && face.D < vertexCount) adjacentFaces[current[face.D]++] = f;
            }

            return new MeshVertexAdjacency(vertexCount, offsets, adjacentFaces);
        }
    }
}
