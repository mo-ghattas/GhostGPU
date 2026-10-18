using System;
using Grasshopper.Kernel;
using RhinoMesh = Rhino.Geometry.Mesh;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Vector
{
    public class GhMove : GhostGpuComponentBase
    {
        public GhMove()
            : base("Move GPU", "Move",
                   "Translates resident GPU points with repeat-last matching. This first version returns points only; it does not return a native transform or move arbitrary Rhino geometry.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("0994B67E-B8B8-43F1-9BE4-332EB7AEB110");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Resident points to translate", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Translation", "T", "Translation vector(s) (resident GPU buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "P", "Translated resident points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer ptsGoo = null;
            GH_GpuBuffer tGoo = null;

            if (!DA.GetData(0, ref ptsGoo) || ptsGoo?.Value == null) return;
            if (!DA.GetData(1, ref tGoo) || tGoo == null || tGoo.Value == null) return;

            var session = GetGpuSession();

            var moved = session.Move(ptsGoo.Value, tGoo.Value);
            SetGpuBuffer(DA, 0, moved);
        }
    }
}
