using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorScale : GhostGpuComponentBase
    {
        public GhVectorScale()
            : base("Vector Scale GPU", "VScale",
                   "Multiplies resident GPU vectors by scalar factors with repeat-last matching (direct component scaling without normalization).",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("88939BC8-F648-433C-8472-E7A4A28BE28B");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Resident GPU vectors to scale", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Factor", "F", "Scalar multiplication factors (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Result", "V", "Scaled resident GPU vectors", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            GH_GpuBuffer fGoo = null;

            if (!DA.GetData(0, ref vGoo) || vGoo?.Value == null) return;
            if (!DA.GetData(1, ref fGoo) || fGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorScale(vGoo.Value, fGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
