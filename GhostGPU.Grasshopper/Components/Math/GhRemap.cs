using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Math
{
    public class GhRemap : GhostGpuComponentBase
    {
        public GhRemap()
            : base("Remap Numbers GPU", "Remap",
                   "Remaps numbers from resident source bounds to a target interval on GPU. Strictly matches native Grasshopper fidelity: collapsed source bounds map to target midpoint (targetMin+targetMax)*0.5, and supports reversed target intervals without host readback. (Outputs Mapped values for v0.1).",
                   SubCategoryMath)
        {
        }

        public override Guid ComponentGuid => new Guid("68BD017C-BE9C-4CD5-827B-CE74F34A8113");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Values", "V", "Scalar values to remap (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Source Bounds", "S", "Source domain bounds [min, max] (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddNumberParameter("Target Min", "T0", "Target interval start", GH_ParamAccess.item, 0.0);
            pManager.AddNumberParameter("Target Max", "T1", "Target interval end", GH_ParamAccess.item, 1.0);
            pManager[2].Optional = true;
            pManager[3].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Mapped", "M", "Remapped values (resident GPU buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            GH_GpuBuffer sGoo = null;
            double t0 = 0.0;
            double t1 = 1.0;

            if (!DA.GetData(0, ref vGoo) || vGoo == null || vGoo.Value == null) return;
            if (!DA.GetData(1, ref sGoo) || sGoo == null || sGoo.Value == null) return;
            DA.GetData(2, ref t0);
            DA.GetData(3, ref t1);

            var session = GetGpuSession();
            var unclipped = session.Remap(vGoo.Value, sGoo.Value, t0, t1, false);

            SetGpuBuffer(DA, 0, unclipped);
        }
    }
}
