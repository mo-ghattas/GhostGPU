using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhVertexNormals : GhostGpuComponentBase
    {
        public GhVertexNormals()
            : base("Vertex Normals GPU", "VNorm",
                   "Calculates vertex normal vectors on GPU using precomputed host CSR adjacency. " +
                   "Supports both area-weighted and unweighted adjacent face normal averaging; " +
                   "zero vector for unreferenced or degenerate vertices.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B103-7E23-4211-9A31-92F6F03F8003");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to evaluate", GH_ParamAccess.item);
            pManager.AddBooleanParameter("AreaWeighted", "W", "Weight face contributions by face area (true) or unweighted (false)", GH_ParamAccess.item, true);
            pManager[1].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Normals", "N", "Resident GPU vectors buffer of vertex unit normals", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;

            bool areaWeighted = true;
            DA.GetData(1, ref areaWeighted);

            var payload = mGoo.Value;
            var session = GetGpuSessionConcrete();

            var topoDesc = MeshTopologyAdapter.BuildTopologyDescriptor(payload.TopologyTemplate);
            if (payload.VertexPositions.Count == 0)
            {
                SetGpuBuffer(DA, 0, session.UploadVectors(Array.Empty<V3>()));
                return;
            }

            var weighting = areaWeighted ? MeshNormalWeighting.AreaWeighted : MeshNormalWeighting.Unweighted;

            using (var faceIndicesBuffer = session.UploadMeshFaceIndices(topoDesc.GetFacesSnapshot()))
            using (var offsetsBuffer = session.UploadIndices(topoDesc.Adjacency.GetOffsetsSnapshot()))
            using (var adjFacesBuffer = session.UploadIndices(topoDesc.Adjacency.GetFacesSnapshot()))
            {
                var normals = session.MeshVertexNormals(
                    payload.VertexPositions,
                    faceIndicesBuffer,
                    offsetsBuffer,
                    adjFacesBuffer,
                    weighting);

                SetGpuBuffer(DA, 0, normals);
            }
        }
    }
}
