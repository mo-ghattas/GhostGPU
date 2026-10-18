using System;
using Grasshopper.Kernel;

namespace GhostGPU.Grasshopper.Parameters
{
    public class Param_GpuBuffer : GH_Param<GH_GpuBuffer>
    {
        public Param_GpuBuffer()
            : base("GPU Buffer", "GBuf", "Handle to a resident GPU memory buffer.", "GhostGPU", "00 | IO", GH_ParamAccess.item)
        {
        }

        public override Guid ComponentGuid => new Guid("8A40E538-4F15-4D56-8A31-4F2A13327661");

        public override GH_Exposure Exposure => GH_Exposure.primary;

        protected override System.Drawing.Bitmap Icon => Icons.GhostGpuIconProvider.GetFallbackIcon("BUF");
    }
}
