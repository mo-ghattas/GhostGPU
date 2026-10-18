using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Vector
{
    public class GhUnitVector : GhostGpuComponentBase
    {
        public GhUnitVector()
            : base("Unit Vector GPU", "UnitV",
                   "Normalizes vectors to unit length on GPU. Zero-length vectors remain zero vectors without error.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("C750798C-D3F1-4F2D-BCBF-DF06D8DFB108");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector", "V", "Vectors to unitize (resident GPU buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Unit Vector", "U", "Unit vectors (resident GPU buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            if (!DA.GetData(0, ref vGoo) || vGoo == null || vGoo.Value == null) return;

            var session = GetGpuSession();
            var result = session.Unitize(vGoo.Value);

            SetGpuBuffer(DA, 0, result);
        }
    }
}
