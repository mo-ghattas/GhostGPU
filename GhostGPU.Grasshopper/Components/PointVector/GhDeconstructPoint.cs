using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhDeconstructPoint : GhostGpuComponentBase
    {
        public GhDeconstructPoint()
            : base("Deconstruct Point GPU", "DePt",
                   "Deconstructs resident GPU points into X, Y, and Z coordinate scalar buffers.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("E623A849-0DF1-4835-B4C2-930B48002712");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "Pt", "Resident GPU points to deconstruct", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "X", "X", "X coordinates (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Y", "Y", "Y coordinates (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Z", "Z", "Z coordinates (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer ptGoo = null;
            if (!DA.GetData(0, ref ptGoo) || ptGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            session.DeconstructPoints(ptGoo.Value, out var x, out var y, out var z);
            SetGpuBuffer(DA, 0, x);
            SetGpuBuffer(DA, 1, y);
            SetGpuBuffer(DA, 2, z);
        }
    }
}
