using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Vector
{
    public class GhClosestPoint : GhostGpuComponentBase
    {
        public GhClosestPoint()
            : base("Closest Point GPU", "ClsPt",
                   "Finds the closest target point for each query point on GPU. Preserves query order and chooses lowest target index for exact distance ties.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("71708AC6-B7E0-4BA8-868C-D9B97F7A6105");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Queries", "Q", "Resident GPU point buffer of query points", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Targets", "T", "Resident GPU point buffer of target/cloud points", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Nearest target points (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Indices", "i", "Indices of nearest target points in target buffer", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Distances", "D", "Distances to nearest target points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer qGoo = null;
            GH_GpuBuffer tGoo = null;

            if (!DA.GetData(0, ref qGoo) || qGoo == null || qGoo.Value == null) return;
            if (!DA.GetData(1, ref tGoo) || tGoo == null || tGoo.Value == null) return;

            var session = GetGpuSession();
            var result = session.ClosestPoints(qGoo.Value, tGoo.Value);

            SetGpuBuffer(DA, 0, result.Points);
            SetGpuBuffer(DA, 1, result.Indices);
            SetGpuBuffer(DA, 2, result.Distances);
        }
    }
}
