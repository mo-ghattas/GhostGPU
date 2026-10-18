using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Vector
{
    public class GhVector2Pt : GhostGpuComponentBase
    {
        public GhVector2Pt()
            : base("Vector 2Pt GPU", "V2Pt",
                   "Creates vectors between pairs of points (B - A) on GPU. Matches native Grasshopper longest-list repetition and provides optional unitization.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("5E035987-9DE8-48D5-BC40-EBCBE0114107");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Point A", "A", "Base points (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Point B", "B", "Tip points (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddBooleanParameter("Unitize", "U", "Unitize resulting vectors to length 1.0", GH_ParamAccess.item, false);
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector", "V", "Vectors from A to B (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Length", "L", "Vector lengths (resident GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;
            bool unitize = false;

            if (!DA.GetData(0, ref aGoo) || aGoo == null || aGoo.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo == null || bGoo.Value == null) return;
            DA.GetData(2, ref unitize);

            var session = GetGpuSession();
            var vecBuf = session.VectorBetween(aGoo.Value, bGoo.Value);
            RegisterSolutionLease(vecBuf);
            var lenBuf = session.Distance(aGoo.Value, bGoo.Value);

            if (unitize)
            {
                var uBuf = session.Unitize(vecBuf);
                vecBuf.Dispose();
                vecBuf = uBuf;
            }

            SetGpuBuffer(DA, 0, vecBuf);
            SetGpuBuffer(DA, 1, lenBuf);
        }
    }
}
