using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhPointScale : GhostGpuComponentBase
    {
        public GhPointScale()
            : base("Scale Point GPU", "ScalePt",
                   "Scales resident GPU points relative to center points by scalar factors with repeat-last matching (P' = C + (P - C) * F).",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("11E1BF3E-C246-4E27-B6DF-D502B0F4E002");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Resident points to scale (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Center", "C", "Scale center points (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Factor", "F", "Scale factors (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Scaled resident GPU points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer pGoo = null;
            GH_GpuBuffer cGoo = null;
            GH_GpuBuffer fGoo = null;

            if (!DA.GetData(0, ref pGoo) || pGoo?.Value == null) return;
            if (!DA.GetData(1, ref cGoo) || cGoo?.Value == null) return;
            if (!DA.GetData(2, ref fGoo) || fGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.PointScale(pGoo.Value, cGoo.Value, fGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
