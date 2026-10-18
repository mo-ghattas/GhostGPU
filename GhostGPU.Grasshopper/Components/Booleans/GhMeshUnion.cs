using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using RhinoMesh = Rhino.Geometry.Mesh;
using GhostGPU.Booleans;

namespace GhostGPU.Grasshopper.Components.Booleans
{
    public class GhMeshUnion : GhostGpuComponentBase
    {
        public GhMeshUnion()
            : base("Mesh Union GPU", "GMUnion",
                   "Perform a GPU-accelerated solid union on a set of meshes.",
                   SubCategoryBooleans)
        {
        }

        public override Guid ComponentGuid => new Guid("9FD9BFE8-4ACF-472E-8DA4-076B55524059");

        public override GH_Exposure Exposure => GH_Exposure.secondary;

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddMeshParameter("Meshes", "M", "Meshes to union", GH_ParamAccess.list);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Result", "R", "Mesh solid union result", GH_ParamAccess.list);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            var list = new List<GH_Mesh>();
            if (!DA.GetDataList(0, list))
            {
                return;
            }

            var triMeshes = new List<TriMesh>(list.Count);
            for (int i = 0; i < list.Count; i++)
            {
                var item = list[i];
                if (item == null || item.Value == null)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Input contains a null item at index {i}.");
                    continue;
                }
                var tm = MeshBooleanAdapter.ToTriMesh(item.Value);
                if (tm != null) triMeshes.Add(tm);
            }

            if (triMeshes.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh set is empty");
                return;
            }

            if (triMeshes.Count == 1)
            {
                var singleMesh = MeshBooleanAdapter.ToRhinoMesh(triMeshes[0]);
                DA.SetDataList(0, new[] { singleMesh });
                return;
            }

            var session = GetGpuSessionConcrete();
            double tol = DocumentTolerance();

            var listA = new List<TriMesh> { triMeshes[0] };
            var listB = new List<TriMesh>(triMeshes.Count - 1);
            for (int i = 1; i < triMeshes.Count; i++) listB.Add(triMeshes[i]);

            var result = MeshBoolean.Compute(session, listA, listB, BooleanOperation.Union, tol);

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
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh Union failed");
                return;
            }

            DA.SetDataList(0, MeshBooleanAdapter.ToRhinoMeshes(result));
        }
    }
}
