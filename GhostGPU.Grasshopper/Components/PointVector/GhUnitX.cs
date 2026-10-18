using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhUnitX : GhostGpuComponentBase
    {
        public GhUnitX()
            : base("Unit X GPU", "UnitX",
                   "Creates unit X vectors (1, 0, 0) optionally scaled by factor scalars.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("17D01FA3-75AC-44C1-8B39-86B5A10A2B11");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Factor", "F", "Scale factor (GPU scalar buffer, optional; default 1.0)", GH_ParamAccess.item);
            pManager[0].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Unit X", "X", "Unit X resident vectors", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer fGoo = null;
            DA.GetData(0, ref fGoo);

            var session = GetGpuSessionConcrete();
            var result = session.UnitX(fGoo?.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
