using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhVectorLength : GhostGpuComponentBase
    {
        public GhVectorLength()
            : base("Vector Length GPU", "VLen",
                   "Computes the Euclidean length of resident GPU vectors.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("4ACFF388-1647-493B-B051-CA18FE5D627A");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Resident GPU vectors to measure", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Length", "L", "Vector lengths (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            if (!DA.GetData(0, ref vGoo) || vGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.VectorLength(vGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
