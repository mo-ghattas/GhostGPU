using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorDivide : GhostGpuComponentBase
    {
        public GhVectorDivide()
            : base("Vector Divide GPU", "VDiv",
                   "Divides resident GPU vectors by scalar factors (V / D) with repeat-last matching. Zero divisors or non-finite inputs return zero vectors (0, 0, 0).",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("3D573DF1-0985-4D3B-A816-B0E329437FE5");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Resident GPU vectors to divide", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Divisor", "D", "Scalar divisors (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Result", "V", "Divided resident GPU vectors", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            GH_GpuBuffer dGoo = null;

            if (!DA.GetData(0, ref vGoo) || vGoo?.Value == null) return;
            if (!DA.GetData(1, ref dGoo) || dGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorDivide(vGoo.Value, dGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
