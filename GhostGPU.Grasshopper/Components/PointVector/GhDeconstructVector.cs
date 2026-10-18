using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhDeconstructVector : GhostGpuComponentBase
    {
        public GhDeconstructVector()
            : base("Deconstruct Vector GPU", "DeVec",
                   "Deconstructs resident GPU vectors into X, Y, and Z coordinate scalar buffers.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("79C1D067-EB8D-476E-8BDD-8820B1333B14");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Resident GPU vectors to deconstruct", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "X", "X", "X components (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Y", "Y", "Y components (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Z", "Z", "Z components (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            if (!DA.GetData(0, ref vGoo) || vGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            session.DeconstructVectors(vGoo.Value, out var x, out var y, out var z);
            SetGpuBuffer(DA, 0, x);
            SetGpuBuffer(DA, 1, y);
            SetGpuBuffer(DA, 2, z);
        }
    }
}
