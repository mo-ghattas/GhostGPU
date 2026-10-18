using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorDot : GhostGpuComponentBase
    {
        public GhVectorDot()
            : base("Vector Dot Product GPU", "Dot",
                   "Computes the scalar dot product of two resident GPU vector buffers with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("3EE0929F-49C3-4886-A060-61A601E6959C");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector A", "A", "First vector buffer (GPU vector buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Vector B", "B", "Second vector buffer (GPU vector buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Dot Product", "D", "Scalar dot product values (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;

            if (!DA.GetData(0, ref aGoo) || aGoo?.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorDotProduct(aGoo.Value, bGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
