using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorAngle : GhostGpuComponentBase
    {
        public GhVectorAngle()
            : base("Vector Angle GPU", "VAngle",
                   "Computes the angle in radians between two resident GPU vector buffers in range [0, π].",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("ACCE1DF2-8B18-4EE4-B974-9C7802D33BBE");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector A", "A", "First vector buffer (GPU vector buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Vector B", "B", "Second vector buffer (GPU vector buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Angle", "A", "Angle in radians between vectors (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;

            if (!DA.GetData(0, ref aGoo) || aGoo?.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorAngle(aGoo.Value, bGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
