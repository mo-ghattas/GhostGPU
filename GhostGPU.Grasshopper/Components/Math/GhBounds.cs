using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Math
{
    public class GhBounds : GhostGpuComponentBase
    {
        public GhBounds()
            : base("Bounds GPU", "Bnd",
                   "Computes the minimum and maximum of a GPU scalar buffer. Returns a resident 2-element GPU buffer [min, max] without host readback.",
                   SubCategoryMath)
        {
        }

        public override Guid ComponentGuid => new Guid("318C7E77-FB9D-40C6-9EE3-EF0F5EE73112");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Numbers", "N", "Scalar buffer to compute bounds for (resident GPU buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Bounds", "B", "Resident 2-element GPU scalar buffer [min, max]", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer nGoo = null;
            if (!DA.GetData(0, ref nGoo) || nGoo == null || nGoo.Value == null) return;

            var session = GetGpuSession();
            var result = session.Bounds(nGoo.Value);

            SetGpuBuffer(DA, 0, result);
        }
    }
}
