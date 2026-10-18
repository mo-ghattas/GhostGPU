using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorAdd : GhostGpuComponentBase
    {
        public GhVectorAdd()
            : base("Vector Add GPU", "VAdd",
                   "Adds two resident GPU vector buffers (A + B) with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("2867F5B1-8C56-424D-B5DF-C51B6BE48C47");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector A", "A", "First vector buffer (GPU vector buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Vector B", "B", "Second vector buffer (GPU vector buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Result", "V", "Vector addition result (A + B)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;

            if (!DA.GetData(0, ref aGoo) || aGoo?.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorAdd(aGoo.Value, bGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
