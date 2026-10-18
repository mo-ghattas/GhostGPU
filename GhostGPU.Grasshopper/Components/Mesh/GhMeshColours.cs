using System;
using Grasshopper.Kernel;
using RhinoMesh = Rhino.Geometry.Mesh;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Mesh
{
    public class GhMeshColours : GhostGpuComponentBase
    {
        public GhMeshColours()
            : base("Mesh Colours GPU", "MCol",
                   "Applies an ARGB colour pattern buffer to a GPU mesh. Follows native Grasshopper fidelity by cyclically repeating the colour pattern across mesh vertices (WrapInteger) and preserving quad topology.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("9846E062-8BC4-4C5D-913C-5586617A2116");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "GPU mesh to colour", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Colours", "C", "Colour pattern buffer (resident GPU colours buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Coloured GPU mesh with resident vertex colours", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            GH_GpuBuffer cGoo = null;

            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;
            if (!DA.GetData(1, ref cGoo) || cGoo == null || cGoo.Value == null) return;

            var payload = mGoo.Value;
            int vertexCount = payload.TopologyTemplate.Vertices.Count;

            var session = GetGpuSession();
            var repeatedColours = session.RepeatColours(cGoo.Value, vertexCount);

            var newPayload = new GpuMeshPayload(
                payload.TopologyTemplate.DuplicateMesh(),
                payload.VertexPositions.Retain(),
                repeatedColours
            );

            SetGpuMesh(DA, 0, newPayload);
        }
    }
}
