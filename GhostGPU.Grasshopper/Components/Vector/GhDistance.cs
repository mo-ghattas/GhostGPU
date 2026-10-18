using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Vector
{
    public class GhDistance : GhostGpuComponentBase
    {
        public GhDistance()
            : base("Distance GPU", "Dist",
                   "Computes pairwise Euclidean distances between point/vector buffers on GPU. Repeats the last element of a shorter non-empty buffer to match longest list.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("F0D9BC05-3948-4394-B6AA-EF78A9C22106");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Point A", "A", "First resident GPU point/vector buffer", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Point B", "B", "Second resident GPU point/vector buffer", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Distance", "D", "Pairwise distances (resident GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;

            if (!DA.GetData(0, ref aGoo) || aGoo == null || aGoo.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo == null || bGoo.Value == null) return;

            var session = GetGpuSession();
            var result = session.Distance(aGoo.Value, bGoo.Value);

            SetGpuBuffer(DA, 0, result);
        }
    }
}
