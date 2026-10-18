using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GhostGPU.Grasshopper.BrepPipeline;

namespace GhostGPU.Grasshopper.Components.Booleans
{
    public class GhSolidIntersection : GhostGpuComponentBase
    {
        public GhSolidIntersection()
            : base("Solid Intersection GPU", "GSInt",
                   "Perform a GPU-accelerated hybrid solid intersection on two Brep sets.",
                   SubCategoryBooleans)
        {
        }

        public override Guid ComponentGuid => new Guid("16E286EE-2659-41B0-9625-E19A0E2F059F");

        public override GH_Exposure Exposure => GH_Exposure.primary;

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter("Breps A", "A", "First Brep set", GH_ParamAccess.list);
            pManager.AddBrepParameter("Breps B", "B", "Second Brep set", GH_ParamAccess.list);
            pManager.AddTextParameter("Kernel", "K", "Exact Brep kernel for intersect/split/join: \"RhinoCommon\" (default) or \"GhostGPU\" (falls back to RhinoCommon per pair with a warning).", GH_ParamAccess.item, nameof(BrepKernel.RhinoCommon));
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddBrepParameter("Result", "R", "Intersection result", GH_ParamAccess.list);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            var listA = new List<GH_Brep>();
            var listB = new List<GH_Brep>();
            if (!DA.GetDataList(0, listA) || !DA.GetDataList(1, listB))
            {
                return;
            }

            string kernelText = nameof(BrepKernel.RhinoCommon);
            DA.GetData(2, ref kernelText);
            if (!BrepBooleanPipeline.TryParseKernel(kernelText, out var kernel))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Unknown Kernel \"{kernelText}\": use \"RhinoCommon\" or \"GhostGPU\".");
                return;
            }

            double angleTol = DocumentAngleTolerance();
            double docTol = DocumentTolerance();

            var preppedA = new List<Brep>(listA.Count);
            for (int i = 0; i < listA.Count; i++)
            {
                var item = listA[i];
                if (item == null || item.Value == null)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Input A contains a null item at index {i}.");
                    continue;
                }
                var b = BrepBooleanPipeline.EnsureCorrectBrep(item.Value, angleTol);
                if (b != null) preppedA.Add(b);
            }

            var preppedB = new List<Brep>(listB.Count);
            for (int i = 0; i < listB.Count; i++)
            {
                var item = listB[i];
                if (item == null || item.Value == null)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Input B contains a null item at index {i}.");
                    continue;
                }
                var b = BrepBooleanPipeline.EnsureCorrectBrep(item.Value, angleTol);
                if (b != null) preppedB.Add(b);
            }

            if (preppedA.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "First Brep set is empty");
                return;
            }
            if (preppedB.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Second Brep set is empty");
                return;
            }

            var session = GetGpuSessionConcrete();
            var result = BrepBooleanPipeline.BooleanIntersection(
                session,
                preppedA,
                preppedB,
                docTol,
                msg => AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, msg),
                kernel);

            if (result == null || result.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Boolean Intersection set is empty");
            }

            DA.SetDataList(0, result);
        }
    }
}
