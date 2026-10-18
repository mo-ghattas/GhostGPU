using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GhostGPU.Grasshopper.BrepPipeline;

namespace GhostGPU.Grasshopper.Components.Booleans
{
    public class GhSolidUnion : GhostGpuComponentBase
    {
        public GhSolidUnion()
            : base("Solid Union GPU", "GSUnion",
                   "Perform a GPU-accelerated hybrid solid union on a set of Breps.",
                   SubCategoryBooleans)
        {
        }

        public override Guid ComponentGuid => new Guid("627AD9C1-0D3C-4BCC-BE6F-B8BBBB23C6A1");

        public override GH_Exposure Exposure => GH_Exposure.primary;

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter("Breps", "B", "Breps to union", GH_ParamAccess.list);
            pManager.AddTextParameter("Kernel", "K", "Exact Brep kernel for intersect/split/join: \"RhinoCommon\" (default) or \"GhostGPU\" (falls back to RhinoCommon per pair with a warning).", GH_ParamAccess.item, nameof(BrepKernel.RhinoCommon));
            pManager[1].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddBrepParameter("Result", "R", "Union result", GH_ParamAccess.list);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            var list = new List<GH_Brep>();
            if (!DA.GetDataList(0, list))
            {
                return;
            }

            string kernelText = nameof(BrepKernel.RhinoCommon);
            DA.GetData(1, ref kernelText);
            if (!BrepBooleanPipeline.TryParseKernel(kernelText, out var kernel))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Unknown Kernel \"{kernelText}\": use \"RhinoCommon\" or \"GhostGPU\".");
                return;
            }

            double angleTol = DocumentAngleTolerance();
            double docTol = DocumentTolerance();

            var prepped = new List<Brep>(list.Count);
            for (int i = 0; i < list.Count; i++)
            {
                var item = list[i];
                if (item == null || item.Value == null)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Input contains a null item at index {i}.");
                    continue;
                }
                var b = BrepBooleanPipeline.EnsureCorrectBrep(item.Value, angleTol);
                if (b != null) prepped.Add(b);
            }

            if (prepped.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No Brep geometry could be reconstructed");
                return;
            }

            if (prepped.Count == 1)
            {
                DA.SetDataList(0, prepped);
                return;
            }

            var session = GetGpuSessionConcrete();
            var result = BrepBooleanPipeline.BooleanUnion(
                session,
                prepped,
                docTol,
                msg => AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, msg),
                kernel);

            if (result == null || result.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Boolean Union set is empty.");
            }

            DA.SetDataList(0, result);
        }
    }
}
