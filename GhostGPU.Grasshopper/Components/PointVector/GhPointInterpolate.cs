using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhPointInterpolate : GhostGpuComponentBase
    {
        public GhPointInterpolate()
            : base("Point Interpolate GPU", "PtInterp",
                   "Linearly interpolates between resident GPU points A and B by parameter T (P = (1-T)*A + T*B) with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("B7E84B05-C02A-4D39-A1B2-7935216EDECF");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Point A", "A", "First point buffer (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Point B", "B", "Second point buffer (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Parameter", "T", "Interpolation parameter (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "Pt", "Interpolated resident GPU points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;
            GH_GpuBuffer tGoo = null;

            if (!DA.GetData(0, ref aGoo) || aGoo?.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo?.Value == null) return;
            if (!DA.GetData(2, ref tGoo) || tGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.PointInterpolate(aGoo.Value, bGoo.Value, tGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
