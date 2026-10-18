using System;
using System.Drawing;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Math
{
    public class GhColourRamp : GhostGpuComponentBase
    {
        public GhColourRamp()
            : base("Colour Ramp GPU", "Ramp",
                   "Evaluates a two-colour gradient ramp on GPU from resident scalar values. Clamps to [0,1], linearly interpolates ARGB bytes, and outputs a resident GPU colours buffer.",
                   SubCategoryMath)
        {
        }

        public override Guid ComponentGuid => new Guid("B687C804-A759-4E6D-B27F-7B27C1748117");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Scalars", "S", "Scalar values to map into colours (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddColourParameter("Low Colour", "C0", "Start colour of the gradient ramp (at value 0.0)", GH_ParamAccess.item, Color.FromArgb(0, 0, 255));
            pManager.AddColourParameter("High Colour", "C1", "End colour of the gradient ramp (at value 1.0)", GH_ParamAccess.item, Color.FromArgb(255, 0, 0));
            pManager[1].Optional = true;
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Colours", "C", "Evaluated ARGB colours (resident GPU colours buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer sGoo = null;
            Color lowColor = Color.FromArgb(0, 0, 255);
            Color highColor = Color.FromArgb(255, 0, 0);

            if (!DA.GetData(0, ref sGoo) || sGoo == null || sGoo.Value == null) return;
            DA.GetData(1, ref lowColor);
            DA.GetData(2, ref highColor);

            var session = GetGpuSession();
            uint lowArgb = (uint)lowColor.ToArgb();
            uint highArgb = (uint)highColor.ToArgb();

            var colours = session.ColourRamp(sGoo.Value, lowArgb, highArgb);

            SetGpuBuffer(DA, 0, colours);
        }
    }
}
