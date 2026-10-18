using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Mesh
{
    public class GhMeshVertices : GhostGpuComponentBase
    {
        public GhMeshVertices()
            : base("Mesh Vertices GPU", "MVerts",
                   "Extracts the resident vertex positions buffer from a GPU mesh. Zero-download retained view.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("24BC97D1-85A3-4DC1-8314-2C6F158BA104");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vertices", "V", "Resident GPU point buffer of vertex positions", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;

            var vertexBuffer = mGoo.Value.VertexPositions.Retain();
            SetGpuBuffer(DA, 0, vertexBuffer);
        }
    }
}
