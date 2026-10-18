using System;
using System.Collections.Generic;
using RhinoMesh = Rhino.Geometry.Mesh;
using GhostGPU.Core;
using GhostGPU.Booleans;

namespace GhostGPU.Grasshopper.Components.Booleans
{
    public static class MeshBooleanAdapter
    {
        public static TriMesh ToTriMesh(RhinoMesh mesh)
        {
            if (mesh == null) return null;
            var cloned = mesh.DuplicateMesh();
            cloned.Faces.ConvertQuadsToTriangles();
            cloned.Compact();

            int vCount = cloned.Vertices.Count;
            int fCount = cloned.Faces.Count;
            var vertices = new V3[vCount];
            for (int i = 0; i < vCount; i++)
            {
                var pt = cloned.Vertices[i];
                vertices[i] = new V3(pt.X, pt.Y, pt.Z);
            }

            var triangles = new int[fCount * 3];
            for (int i = 0; i < fCount; i++)
            {
                var f = cloned.Faces[i];
                triangles[i * 3] = f.A;
                triangles[i * 3 + 1] = f.B;
                triangles[i * 3 + 2] = f.C;
            }

            return new TriMesh
            {
                Vertices = vertices,
                Triangles = triangles
            };
        }

        public static RhinoMesh ToRhinoMesh(TriMesh tri)
        {
            if (tri == null || tri.Vertices == null || tri.Triangles == null) return null;
            var mesh = new RhinoMesh();
            // Float vertices collapse sub-float slivers the exact kernel produces.
            mesh.Vertices.UseDoublePrecisionVertices = true;
            for (int i = 0; i < tri.Vertices.Length; i++)
            {
                var v = tri.Vertices[i];
                mesh.Vertices.Add(v.X, v.Y, v.Z);
            }
            for (int i = 0; i < tri.Triangles.Length; i += 3)
            {
                mesh.Faces.AddFace(tri.Triangles[i], tri.Triangles[i + 1], tri.Triangles[i + 2]);
            }
            mesh.Normals.ComputeNormals();
            mesh.Compact();
            return mesh;
        }

        /// <summary>One Rhino mesh per connected piece, like native mesh booleans.</summary>
        public static List<RhinoMesh> ToRhinoMeshes(MeshBooleanResult result)
        {
            var list = new List<RhinoMesh>();
            var pieces = result.Pieces != null && result.Pieces.Length > 0 ? result.Pieces : new[] { result.Mesh };
            foreach (var p in pieces)
            {
                var m = ToRhinoMesh(p);
                if (m != null && m.Vertices.Count > 0) list.Add(m);
            }
            return list;
        }
    }
}
