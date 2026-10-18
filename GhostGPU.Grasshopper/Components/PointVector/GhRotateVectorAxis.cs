using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhRotateVectorAxis : GhostGpuComponentBase
    {
        public GhRotateVectorAxis()
            : base("Rotate Vector GPU", "RotVec",
                   "Rotates resident GPU vectors around an axis vector by an angle in radians using Rodrigues rotation formula with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("6986C8CE-2E24-42DE-8C56-DA15697669F1");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Resident GPU vectors to rotate", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Axis", "X", "Rotation axis vector (GPU vector buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Angle", "A", "Rotation angle in radians (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Rotated resident GPU vectors", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer tGoo = null;

            if (!DA.GetData(0, ref vGoo) || vGoo?.Value == null) return;
            if (!DA.GetData(1, ref aGoo) || aGoo?.Value == null) return;
            if (!DA.GetData(2, ref tGoo) || tGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.RotateVectorAxis(vGoo.Value, aGoo.Value, tGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
