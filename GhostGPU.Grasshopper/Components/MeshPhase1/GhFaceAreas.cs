using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhFaceAreas : GhostGpuComponentBase
    {
        public GhFaceAreas()
            : base("Face Areas GPU", "FArea",
                   "Calculates the surface area of each face in a resident GPU mesh. " +
                   "Triangulated surface area (ABC + ACD) for nonplanar quads; returns zero for degenerate faces.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B102-7E23-4211-9A31-92F6F03F8002");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to evaluate", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Areas", "A", "Resident GPU scalars buffer of face surface areas", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;

            var payload = mGoo.Value;
            var session = GetGpuSessionConcrete();

            var topoDesc = MeshTopologyAdapter.BuildTopologyDescriptor(payload.TopologyTemplate);
            if (topoDesc.FaceCount == 0 || payload.VertexPositions.Count == 0)
            {
                SetGpuBuffer(DA, 0, session.UploadScalars(Array.Empty<double>()));
                return;
            }

            using (var faceIndicesBuffer = session.UploadMeshFaceIndices(topoDesc.GetFacesSnapshot()))
            {
                var areasBuffer = session.MeshFaceAreas(payload.VertexPositions, faceIndicesBuffer);
                SetGpuBuffer(DA, 0, areasBuffer);
            }
        }
    }
}
