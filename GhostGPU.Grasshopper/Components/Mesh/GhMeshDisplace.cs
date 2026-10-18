using System;
using Grasshopper.Kernel;
using RhinoMesh = Rhino.Geometry.Mesh;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Mesh
{
    public class GhMeshDisplace : GhostGpuComponentBase
    {
        public GhMeshDisplace()
            : base("Mesh Displace GPU", "MDisp",
                   "Displaces GPU mesh vertices using resident displacement vectors. Strictly preserves mesh vertex cardinality (vector count must equal vertex count or 1 for uniform translation). Preserves host topology, faces, and texture coordinates.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("8DA3A7BD-BC84-48FE-8B4F-A16DC19A6115");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "GPU mesh to displace", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Vectors", "V", "Per-vertex displacement vectors (resident GPU buffer, count must equal vertex count or 1)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Displaced GPU mesh with updated vertex positions", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            GH_GpuBuffer vGoo = null;

            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;
            if (!DA.GetData(1, ref vGoo) || vGoo == null || vGoo.Value == null) return;

            var payload = mGoo.Value;
            int vCount = payload.TopologyTemplate.Vertices.Count;
            int vecCount = vGoo.Value.Count;

            // Preserve the public displacement cardinality contract.
            if (vCount == 0 && vecCount == 0)
            {
                SetGpuMesh(DA, 0, payload.Retain());
                return;
            }

            if (vecCount != vCount && vecCount != 1)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Displace requires displacement vector count ({vecCount}) to equal mesh vertex count ({vCount}) " +
                    "or 1 (for uniform translation). Arbitrary list lengths that add or drop vertices are rejected to preserve topology.");
            }

            var session = GetGpuSession();
            var displacedVertices = session.Move(payload.VertexPositions, vGoo.Value);

            var newPayload = new GpuMeshPayload(
                payload.TopologyTemplate.DuplicateMesh(),
                displacedVertices,
                payload.VertexColours?.Retain()
            );

            SetGpuMesh(DA, 0, newPayload);
        }
    }
}
