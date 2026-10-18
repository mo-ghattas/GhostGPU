using System;
using Grasshopper.Kernel;
using Rhino.Geometry;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhMeshTransform : GhostGpuComponentBase
    {
        public GhMeshTransform()
            : base("Mesh Transform GPU", "MXform",
                   "Applies an affine 4x4 transformation matrix directly to resident GPU mesh vertices without altering topology.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B109-7E23-4211-9A31-92F6F03F8009");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to transform", GH_ParamAccess.item);
            pManager.AddTransformParameter("Transform", "X", "4x4 affine transformation matrix", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Transformed GPU mesh with updated vertex positions", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            Transform xform = Transform.Identity;

            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;
            if (!DA.GetData(1, ref xform)) return;

            var payload = mGoo.Value;
            if (payload.TopologyTemplate.Vertices.Count == 0)
            {
                SetGpuMesh(DA, 0, payload.Retain());
                return;
            }

            double[] matrix = new double[16]
            {
                xform.M00, xform.M01, xform.M02, xform.M03,
                xform.M10, xform.M11, xform.M12, xform.M13,
                xform.M20, xform.M21, xform.M22, xform.M23,
                xform.M30, xform.M31, xform.M32, xform.M33
            };

            var session = GetGpuSessionConcrete();
            var transformedVertices = session.MeshTransform(payload.VertexPositions, matrix);

            var dupMesh = payload.TopologyTemplate.DuplicateMesh();
            try
            {
                dupMesh.FaceNormals.Clear();
                dupMesh.Normals.Clear();
            }
            catch { }

            var newPayload = new GpuMeshPayload(
                dupMesh,
                transformedVertices,
                payload.VertexColours?.Retain()
            );

            SetGpuMesh(DA, 0, newPayload);
        }
    }
}
