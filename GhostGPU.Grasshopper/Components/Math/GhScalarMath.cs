using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Math
{
    public class GhScalarMath : GhostGpuComponentBase
    {
        public GhScalarMath()
            : base("Scalar Math GPU", "Math",
                   "Performs pairwise arithmetic operations (Add, Subtract, Multiply, Divide) on resident GPU scalar buffers. Matches longest-list repetition and returns IEEE floating-point results.",
                   SubCategoryMath)
        {
        }

        public override Guid ComponentGuid => new Guid("8688D0F1-5246-4B89-AE94-4F81BF6AA111");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Buffer A", "A", "First scalar buffer (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Buffer B", "B", "Second scalar buffer (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Operation", "Op", "Arithmetic operation (0: Add, 1: Subtract, 2: Multiply, 3: Divide)", GH_ParamAccess.item, 0);
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Result", "R", "Calculated scalar result (resident GPU buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer bGoo = null;
            int opCode = 0;

            if (!DA.GetData(0, ref aGoo) || aGoo == null || aGoo.Value == null) return;
            if (!DA.GetData(1, ref bGoo) || bGoo == null || bGoo.Value == null) return;
            DA.GetData(2, ref opCode);

            if (opCode < 0 || opCode > 3)
            {
                throw new ArgumentOutOfRangeException(nameof(opCode), "Operation must be between 0 and 3 (0: Add, 1: Subtract, 2: Multiply, 3: Divide).");
            }

            var session = GetGpuSession();
            var op = (ScalarOperation)opCode;
            var result = session.ScalarMath(aGoo.Value, bGoo.Value, op);

            SetGpuBuffer(DA, 0, result);
        }
    }
}
