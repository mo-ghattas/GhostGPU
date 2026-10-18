using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhMeshScale : GhostGpuComponentBase
    {
        public GhMeshScale()
            : base("Mesh Scale GPU", "MScale",
                   "Scales resident GPU mesh vertices around a center point without altering topology. " +
                   "Supports uniform scalar scaling or nonuniform 3D vector scaling (XYZ factors).",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B106-7E23-4211-9A31-92F6F03F8006");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to scale", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Center", "C", "Center of scaling (point buffer, count=1 or vertex count)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Factor", "F", "Scale factor (scalar buffer for uniform scaling, or vector buffer for nonuniform XYZ scaling)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Scaled GPU mesh with updated vertex positions", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            GH_GpuBuffer cGoo = null;
            GH_GpuBuffer fGoo = null;

            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;
            if (!DA.GetData(1, ref cGoo) || cGoo == null || cGoo.Value == null) return;
            if (!DA.GetData(2, ref fGoo) || fGoo == null || fGoo.Value == null) return;

            var payload = mGoo.Value;
            var centerBuf = cGoo.Value;
            var factorBuf = fGoo.Value;

            if (centerBuf.Kind != BufferKind.Points)
                throw new ArgumentException("Scale center input must be a Points buffer.", nameof(cGoo));
            if (factorBuf.Kind != BufferKind.Scalars && factorBuf.Kind != BufferKind.Vectors)
                throw new ArgumentException("Scale factor input must be a Scalars or Vectors buffer.", nameof(fGoo));

            int vCount = payload.TopologyTemplate.Vertices.Count;
            if (vCount == 0)
            {
                SetGpuMesh(DA, 0, payload.Retain());
                return;
            }

            if (centerBuf.Count != 1 && centerBuf.Count != vCount)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Scale requires center count ({centerBuf.Count}) to equal 1 or mesh vertex count ({vCount}). Arbitrary lengths are rejected to preserve fixed topology.");
            }
            if (factorBuf.Count != 1 && factorBuf.Count != vCount)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Scale requires factor count ({factorBuf.Count}) to equal 1 or mesh vertex count ({vCount}). Arbitrary lengths are rejected to preserve fixed topology.");
            }

            var session = GetGpuSessionConcrete();
            var scaledVertices = session.MeshScale(payload.VertexPositions, centerBuf, factorBuf);

            var dupMesh = payload.TopologyTemplate.DuplicateMesh();
            // Invalidate stale cached host normals
            try
            {
                dupMesh.FaceNormals.Clear();
                dupMesh.Normals.Clear();
            }
            catch { }

            var newPayload = new GpuMeshPayload(
                dupMesh,
                scaledVertices,
                payload.VertexColours?.Retain()
            );

            SetGpuMesh(DA, 0, newPayload);
        }
    }
}
