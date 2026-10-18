using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorReverse : GhostGpuComponentBase
    {
        public GhVectorReverse()
            : base("Vector Reverse GPU", "VRev",
                   "Reverses resident GPU vectors (-V).",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("342BA8B5-9BE2-4363-9562-F2E515286169");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Resident GPU vectors to reverse", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Result", "V", "Reversed vectors (-V)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            if (!DA.GetData(0, ref vGoo) || vGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorReverse(vGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
