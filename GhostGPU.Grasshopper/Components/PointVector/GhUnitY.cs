using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhUnitY : GhostGpuComponentBase
    {
        public GhUnitY()
            : base("Unit Y GPU", "UnitY",
                   "Creates unit Y vectors (0, 1, 0) optionally scaled by factor scalars.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("8B5D4E12-F559-4D33-A2A9-3A7F11C8E522");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Factor", "F", "Scale factor (GPU scalar buffer, optional; default 1.0)", GH_ParamAccess.item);
            pManager[0].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Unit Y", "Y", "Unit Y resident vectors", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer fGoo = null;
            DA.GetData(0, ref fGoo);

            var session = GetGpuSessionConcrete();
            var result = session.UnitY(fGoo?.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
