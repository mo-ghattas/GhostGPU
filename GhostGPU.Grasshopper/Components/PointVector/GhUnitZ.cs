using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhUnitZ : GhostGpuComponentBase
    {
        public GhUnitZ()
            : base("Unit Z GPU", "UnitZ",
                   "Creates unit Z vectors (0, 0, 1) optionally scaled by factor scalars.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("4E347A22-3168-45F4-9844-428CE2B7B333");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Factor", "F", "Scale factor (GPU scalar buffer, optional; default 1.0)", GH_ParamAccess.item);
            pManager[0].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Unit Z", "Z", "Unit Z resident vectors", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer fGoo = null;
            DA.GetData(0, ref fGoo);

            var session = GetGpuSessionConcrete();
            var result = session.UnitZ(fGoo?.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
