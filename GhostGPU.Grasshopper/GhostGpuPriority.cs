using Grasshopper;
using Grasshopper.Kernel;

namespace GhostGPU.Grasshopper
{
    public class GhostGpuPriority : GH_AssemblyPriority
    {
        public override GH_LoadingInstruction PriorityLoad()
        {
            Instances.ComponentServer.AddCategoryShortName("GhostGPU", "Ghost");
            Instances.ComponentServer.AddCategorySymbolName("GhostGPU", 'G');
            return GH_LoadingInstruction.Proceed;
        }
    }
}
