using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorCross : GhostGpuComponentBase
    {
        public GhVectorCross()
            : base("Vector Cross Product GPU", "Cross",
                   "Computes the vector cross product (A x B) of two resident GPU vector buffers with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("F91DC962-DF50-4F5A-8A46-BE0D9E9D75AD");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector A", "A", "First vector buffer (GPU vector buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Vector B", "B", "Second vector buffer (GPU vector buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Cross Product", "C", "Cross product vector (A x B)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;

            if (!DA.GetData(0, ref aGoo) || aGoo?.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorCrossProduct(aGoo.Value, bGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
