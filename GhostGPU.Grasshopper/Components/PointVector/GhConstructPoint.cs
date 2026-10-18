using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhConstructPoint : GhostGpuComponentBase
    {
        public GhConstructPoint()
            : base("Construct Point GPU", "Pt",
                   "Constructs resident GPU points from X, Y, and Z coordinate scalar buffers with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("6B0A8211-7F45-4B9D-9457-302E1528BA01");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "X", "X", "X coordinate (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Y", "Y", "Y coordinate (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Z", "Z", "Z coordinate (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "Pt", "Constructed resident GPU points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer xGoo = null;
            GH_GpuBuffer yGoo = null;
            GH_GpuBuffer zGoo = null;

            if (!DA.GetData(0, ref xGoo) || xGoo?.Value == null) return;
            if (!DA.GetData(1, ref yGoo) || yGoo?.Value == null) return;
            if (!DA.GetData(2, ref zGoo) || zGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.ConstructPoints(xGoo.Value, yGoo.Value, zGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
