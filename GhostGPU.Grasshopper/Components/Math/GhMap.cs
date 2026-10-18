using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Math
{
    public class GhMap : GhostGpuComponentBase
    {
        public GhMap()
            : base("Function Map GPU", "Map",
                   "Applies mathematical shaping functions (Linear, SmoothStep, Sine) to a resident GPU scalar buffer. Evaluates entirely on GPU.",
                   SubCategoryMath)
        {
        }

        public override Guid ComponentGuid => new Guid("1251E14F-C00D-477D-9730-C7BAAC88F114");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Values", "V", "Scalar values to map (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Function", "F", "Mapping function (0: Linear, 1: SmoothStep, 2: Sine)", GH_ParamAccess.item, 1);
            pManager[1].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Result", "R", "Mapped values (resident GPU buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            int fnIndex = 1;

            if (!DA.GetData(0, ref vGoo) || vGoo == null || vGoo.Value == null) return;
            DA.GetData(1, ref fnIndex);

            if (fnIndex < 0 || fnIndex > 2)
            {
                throw new ArgumentOutOfRangeException(nameof(fnIndex), "Function index must be between 0 and 2 (0: Linear, 1: SmoothStep, 2: Sine).");
            }

            var session = GetGpuSession();
            var fn = (MapFunction)fnIndex;
            var result = session.Map(vGoo.Value, fn);

            SetGpuBuffer(DA, 0, result);
        }
    }
}
