using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhProjectVectorToPlane : GhostGpuComponentBase
    {
        public GhProjectVectorToPlane()
            : base("Project Vector GPU", "ProjVec",
                   "Projects resident GPU vectors onto planes defined by normal vectors with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("C4F02DB5-F329-4112-BF9B-A5C9B39B2AE0");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Resident GPU vectors to project", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Normal", "N", "Plane normal vectors (GPU vector buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Projected resident GPU vectors", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            GH_GpuBuffer nGoo = null;

            if (!DA.GetData(0, ref vGoo) || vGoo?.Value == null) return;
            if (!DA.GetData(1, ref nGoo) || nGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.ProjectVectorToPlane(vGoo.Value, nGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
