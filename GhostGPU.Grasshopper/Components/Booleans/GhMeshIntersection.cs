using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using RhinoMesh = Rhino.Geometry.Mesh;
using GhostGPU.Booleans;

namespace GhostGPU.Grasshopper.Components.Booleans
{
    public class GhMeshIntersection : GhostGpuComponentBase
    {
        public GhMeshIntersection()
            : base("Mesh Intersection GPU", "GMInt",
                   "Perform a GPU-accelerated mesh intersection on two mesh sets.",
                   SubCategoryBooleans)
        {
        }

        public override Guid ComponentGuid => new Guid("A1372AA7-6986-4982-A5B2-05069149B0E5");

        public override GH_Exposure Exposure => GH_Exposure.secondary;

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddMeshParameter("Meshes A", "A", "First mesh set", GH_ParamAccess.list);
            pManager.AddMeshParameter("Meshes B", "B", "Second mesh set", GH_ParamAccess.list);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Result", "R", "Intersection mesh result", GH_ParamAccess.list);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            var listA = new List<GH_Mesh>();
            var listB = new List<GH_Mesh>();
            if (!DA.GetDataList(0, listA) || !DA.GetDataList(1, listB))
            {
                return;
            }

            var triA = new List<TriMesh>(listA.Count);
            for (int i = 0; i < listA.Count; i++)
            {
                var item = listA[i];
                if (item == null || item.Value == null)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Input A contains a null item at index {i}.");
                    continue;
                }
                var tm = MeshBooleanAdapter.ToTriMesh(item.Value);
                if (tm != null) triA.Add(tm);
            }

            var triB = new List<TriMesh>(listB.Count);
            for (int i = 0; i < listB.Count; i++)
            {
                var item = listB[i];
                if (item == null || item.Value == null)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Input B contains a null item at index {i}.");
                    continue;
                }
                var tm = MeshBooleanAdapter.ToTriMesh(item.Value);
                if (tm != null) triB.Add(tm);
            }

            if (triA.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "First mesh set is empty");
                return;
            }
            if (triB.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Second mesh set is empty");
                return;
            }

            var session = GetGpuSessionConcrete();
            double tol = DocumentTolerance();

            var result = MeshBoolean.Compute(session, triA, triB, BooleanOperation.Intersection, tol);

            if (result != null && result.Warnings != null)
            {
                foreach (var warn in result.Warnings)
                {
                    if (!string.IsNullOrEmpty(warn))
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, warn);
                    }
                }
            }

            if (result == null || result.Mesh == null || result.Mesh.Vertices == null || result.Mesh.Vertices.Length == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh Intersection result is empty");
                return;
            }

            DA.SetDataList(0, MeshBooleanAdapter.ToRhinoMeshes(result));
        }
    }
}
