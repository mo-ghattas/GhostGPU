using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhDeconstructMesh : GhostGpuComponentBase
    {
        public GhDeconstructMesh()
            : base("Deconstruct Mesh GPU", "DeMesh",
                   "Deconstructs a resident GPU mesh into constituent resident GPU buffers: " +
                   "vertices, face indices (quad/triangle index descriptors), vertex colours (if present), and vertex normals.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B104-7E23-4211-9A31-92F6F03F8004");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to deconstruct", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Vertices", "V", "Resident GPU point buffer of vertex positions", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Faces", "F", "Resident GPU indices buffer of face vertex connectivity (4 indices per face)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Colours", "C", "Resident GPU colour buffer of vertex colours (empty if uncoloured)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Normals", "N", "Resident GPU vector buffer of vertex normals", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;

            var payload = mGoo.Value;
            var session = GetGpuSessionConcrete();

            // Output 0: Vertices (retained zero-download lease)
            SetGpuBuffer(DA, 0, payload.VertexPositions.Retain());

            // Build topology and faces buffer
            var topoDesc = MeshTopologyAdapter.BuildTopologyDescriptor(payload.TopologyTemplate);
            var faceIndicesBuffer = session.UploadMeshFaceIndices(topoDesc.GetFacesSnapshot());
            SetGpuBuffer(DA, 1, faceIndicesBuffer);

            // Output 2: Colours
            if (payload.VertexColours != null && !payload.VertexColours.IsDisposed)
            {
                SetGpuBuffer(DA, 2, payload.VertexColours.Retain());
            }
            else
            {
                SetGpuBuffer(DA, 2, session.UploadColours(Array.Empty<uint>()));
            }

            // Output 3: Normals
            if (payload.VertexPositions.Count == 0)
            {
                SetGpuBuffer(DA, 3, session.UploadVectors(Array.Empty<V3>()));
            }
            else
            {
                using (var offsetsBuffer = session.UploadIndices(topoDesc.Adjacency.GetOffsetsSnapshot()))
                using (var adjFacesBuffer = session.UploadIndices(topoDesc.Adjacency.GetFacesSnapshot()))
                {
                    var normals = session.MeshVertexNormals(
                        payload.VertexPositions,
                        faceIndicesBuffer,
                        offsetsBuffer,
                        adjFacesBuffer,
                        MeshNormalWeighting.AreaWeighted);

                    SetGpuBuffer(DA, 3, normals);
                }
            }
        }
    }
}
