using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhConstructVector : GhostGpuComponentBase
    {
        public GhConstructVector()
            : base("Construct Vector GPU", "Vec",
                   "Constructs resident GPU vectors from X, Y, and Z coordinate scalar buffers with repeat-last matching.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("9C465228-44A5-4BC5-88AA-E2E109605333");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "X", "X", "X component (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Y", "Y", "Y component (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Z", "Z", "Z component (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Constructed resident GPU vectors", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer xGoo = null;
            GH_GpuBuffer yGoo = null;
            GH_GpuBuffer zGoo = null;

            if (!DA.GetData(0, ref xGoo) || xGoo?.Value == null) return;
            if (!DA.GetData(1, ref yGoo) || yGoo?.Value == null) return;
            if (!DA.GetData(2, ref zGoo) || zGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.ConstructVectors(xGoo.Value, yGoo.Value, zGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
