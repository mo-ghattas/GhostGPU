using System;
using System.Reflection;
using System.Runtime.InteropServices;
using Grasshopper.Kernel;

[assembly: AssemblyTitle("GhostGPU.Grasshopper")]
[assembly: AssemblyDescription("CUDA-accelerated Grasshopper node pack powered by ILGPU")]
[assembly: AssemblyConfiguration("")]
[assembly: AssemblyCompany("GhostGPU contributors")]
[assembly: AssemblyProduct("GhostGPU")]
[assembly: AssemblyCopyright("Copyright © 2026")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]
[assembly: ComVisible(false)]
[assembly: Guid("4C519965-FDE5-46D3-BC30-A21FE8AE5167")]

namespace GhostGPU.Grasshopper
{
    public class GhostGpuAssemblyInfo : GH_AssemblyInfo
    {
        public override string Name => "GhostGPU";
        public override System.Drawing.Bitmap Icon => null;
        public override string Description => "CUDA-accelerated Grasshopper node pack powered by ILGPU";
        public override Guid Id => new Guid("4C519965-FDE5-46D3-BC30-A21FE8AE5167");
        public override string AuthorName => "GhostGPU contributors";
        public override string AuthorContact => string.Empty;
        public override string Version => "0.2.0";
    }
}
