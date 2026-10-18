using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhPointProject : GhostGpuComponentBase
    {
        public GhPointProject()
            : base("Project Point GPU", "ProjPt",
                   "Projects resident GPU points onto planes defined by origin points and normal vectors with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("91B33B88-C488-46E1-8BD5-5EF092A7A444");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Resident points to project (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Origin", "O", "Plane origin points (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Normal", "N", "Plane normal vectors (GPU vector buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Projected resident GPU points", GH_ParamAccess.item);
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
            var result = session.PointProject(pGoo.Value, oGoo.Value, nGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
