using System;
using Grasshopper.Kernel;

namespace GhostGPU.Grasshopper.Parameters
{
    public class Param_GpuMesh : GH_Param<GH_GpuMesh>
    {
        public Param_GpuMesh()
            : base("GPU Mesh", "GMsh", "Resident GPU mesh with host topology template.", "GhostGPU", "00 | IO", GH_ParamAccess.item)
        {
        }

        public override Guid ComponentGuid => new Guid("9E45B821-2E11-4712-B835-A1098679F123");

        public override GH_Exposure Exposure => GH_Exposure.primary;

        protected override System.Drawing.Bitmap Icon => null;
    }
}
