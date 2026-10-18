using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhPointRotateAxis : GhostGpuComponentBase
    {
        public GhPointRotateAxis()
            : base("Rotate Point GPU", "RotPt",
                   "Rotates resident GPU points around an axis defined by center points and direction vectors by angles in radians with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("77FE7C7A-A1BA-4AE1-931B-EFEA420FE913");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Resident points to rotate (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Center", "C", "Center points on rotation axis (GPU point buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Axis", "X", "Rotation axis direction vectors (GPU vector buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Angle", "A", "Rotation angle in radians (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Rotated resident GPU points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer pGoo = null;
            GH_GpuBuffer cGoo = null;
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer tGoo = null;

            if (!DA.GetData(0, ref pGoo) || pGoo?.Value == null) return;
            if (!DA.GetData(1, ref cGoo) || cGoo?.Value == null) return;
            if (!DA.GetData(2, ref aGoo) || aGoo?.Value == null) return;
            if (!DA.GetData(3, ref tGoo) || tGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.PointRotateAxis(pGoo.Value, cGoo.Value, aGoo.Value, tGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
