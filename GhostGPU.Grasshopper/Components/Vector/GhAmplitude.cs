using System;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Vector
{
    public class GhAmplitude : GhostGpuComponentBase
    {
        public GhAmplitude()
            : base("Amplitude GPU", "Amp",
                   "Sets vector amplitude on GPU. Follows native Grasshopper workflow: unitizes vectors then scales by magnitude, supporting negative amplitudes. Zero vectors return zero vectors.",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("23000673-E2E1-4FFD-8B42-4D907CFB2109");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector", "V", "Vectors to scale (resident GPU buffer)", GH_ParamAccess.item);
            pManager.AddGenericParameter("Amplitude", "A", "Target magnitude(s) as resident GPU scalar buffer or numeric value", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vector", "V", "Scaled vectors (resident GPU buffer)", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer vGoo = null;
            if (!DA.GetData(0, ref vGoo) || vGoo == null || vGoo.Value == null) return;

            object ampObj = null;
            if (!DA.GetData(1, ref ampObj) || ampObj == null) return;

            var session = GetGpuSession();
            GpuBuffer ampBuf = null;
            bool disposeAmpBuf = false;

            try
            {
                if (ampObj is GH_GpuBuffer ampGpu && ampGpu.Value != null)
                {
                    ampBuf = ampGpu.Value;
                }
                else if (ampObj is GH_Number ghNum)
                {
                    ampBuf = session.UploadScalars(new[] { ghNum.Value });
                    disposeAmpBuf = true;
                }
                else if (ampObj is double d)
                {
                    ampBuf = session.UploadScalars(new[] { d });
                    disposeAmpBuf = true;
                }
                else if (ampObj is int i)
                {
                    ampBuf = session.UploadScalars(new double[] { i });
                    disposeAmpBuf = true;
                }
                else
                {
                    throw new ArgumentException($"Cannot convert amplitude input '{ampObj.GetType().Name}' to a scalar magnitude.");
                }

                var result = session.Amplitude(vGoo.Value, ampBuf);
                SetGpuBuffer(DA, 0, result);
            }
            finally
            {
                if (disposeAmpBuf && ampBuf != null)
                {
                    ampBuf.Dispose();
                }
            }
        }
    }
}
