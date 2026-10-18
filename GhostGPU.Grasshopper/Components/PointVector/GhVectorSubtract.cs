using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorSubtract : GhostGpuComponentBase
    {
        public GhVectorSubtract()
            : base("Vector Subtract GPU", "VSub",
                   "Subtracts resident GPU vector buffer B from A (A - B) with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("86D5A250-9852-4740-97FE-4458EE858658");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector A", "A", "First vector buffer (GPU vector buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Vector B", "B", "Vector buffer to subtract (GPU vector buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Result", "V", "Vector subtraction result (A - B)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;

            if (!DA.GetData(0, ref aGoo) || aGoo?.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorSubtract(aGoo.Value, bGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
