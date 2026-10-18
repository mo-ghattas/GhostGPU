using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhMeshMirror : GhostGpuComponentBase
    {
        public GhMeshMirror()
            : base("Mesh Mirror GPU", "MMirr",
                   "Mirrors resident GPU mesh vertices across a plane (defined by origin point and normal vector) " +
                   "without altering topology. Preserves vertex indexing (note: spatial winding is reflected).",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B108-7E23-4211-9A31-92F6F03F8008");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to mirror", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Origin", "O", "Plane origin point buffer (count=1)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Normal", "N", "Plane normal vector buffer (count=1)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Mirrored GPU mesh with updated vertex positions", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            GH_GpuBuffer oGoo = null;
            GH_GpuBuffer nGoo = null;

            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;
            if (!DA.GetData(1, ref oGoo) || oGoo == null || oGoo.Value == null) return;
            if (!DA.GetData(2, ref nGoo) || nGoo == null || nGoo.Value == null) return;

            var payload = mGoo.Value;
            var originBuf = oGoo.Value;
            var normalBuf = nGoo.Value;

            if (originBuf.Kind != BufferKind.Points)
                throw new ArgumentException("Plane origin input must be a Points buffer.", nameof(oGoo));
            if (normalBuf.Kind != BufferKind.Vectors)
                throw new ArgumentException("Plane normal input must be a Vectors buffer.", nameof(nGoo));

            int vCount = payload.TopologyTemplate.Vertices.Count;
            if (vCount == 0)
            {
                SetGpuMesh(DA, 0, payload.Retain());
                return;
            }

            if (originBuf.Count != 1 && originBuf.Count != vCount)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Mirror requires plane origin count ({originBuf.Count}) to equal 1 or mesh vertex count ({vCount}). Arbitrary lengths are rejected to preserve fixed topology.");
            }
            if (normalBuf.Count != 1 && normalBuf.Count != vCount)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Mirror requires plane normal count ({normalBuf.Count}) to equal 1 or mesh vertex count ({vCount}). Arbitrary lengths are rejected to preserve fixed topology.");
            }

            var session = GetGpuSessionConcrete();
            var mirroredVertices = session.MeshMirror(payload.VertexPositions, originBuf, normalBuf);

            var dupMesh = payload.TopologyTemplate.DuplicateMesh();
            try
            {
                dupMesh.FaceNormals.Clear();
                dupMesh.Normals.Clear();
            }
            catch { }

            var newPayload = new GpuMeshPayload(
                dupMesh,
                mirroredVertices,
                payload.VertexColours?.Retain()
            );

            SetGpuMesh(DA, 0, newPayload);
        }
    }
}
