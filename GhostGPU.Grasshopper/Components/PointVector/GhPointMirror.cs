using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhPointMirror : GhostGpuComponentBase
    {
        public GhPointMirror()
            : base("Mirror Point GPU", "MirPt",
                   "Reflects resident GPU points across mirror planes defined by origin points and normal vectors with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("EBEFF8DF-7C0F-4D2A-B9C0-1C6DE7CFFD24");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Resident points to reflect (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Origin", "O", "Mirror plane origin points (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Normal", "N", "Mirror plane normal vectors (GPU vector buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Reflected resident GPU points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer pGoo = null;
            GH_GpuBuffer oGoo = null;
            GH_GpuBuffer nGoo = null;

            if (!DA.GetData(0, ref pGoo) || pGoo?.Value == null) return;
            if (!DA.GetData(1, ref oGoo) || oGoo?.Value == null) return;
            if (!DA.GetData(2, ref nGoo) || nGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.PointMirror(pGoo.Value, oGoo.Value, nGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
