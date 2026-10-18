using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhFaceNormals : GhostGpuComponentBase
    {
        public GhFaceNormals()
            : base("Face Normals GPU", "FNorm",
                   "Calculates center points and normal vectors for all faces of a resident GPU mesh. " +
                   "Triangulation-independent Newell normal and centroid formulation for nonplanar quads; " +
                   "zero vector for degenerate faces.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B101-7E23-4211-9A31-92F6F03F8001");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to evaluate", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Centers", "C", "Resident GPU points buffer of face centers", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Normals", "N", "Resident GPU vectors buffer of face unit normals", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;

            var payload = mGoo.Value;
            var session = GetGpuSessionConcrete();

            var topoDesc = MeshTopologyAdapter.BuildTopologyDescriptor(payload.TopologyTemplate);
            if (topoDesc.FaceCount == 0 || payload.VertexPositions.Count == 0)
            {
                SetGpuBuffer(DA, 0, session.UploadPoints(Array.Empty<V3>()));
                SetGpuBuffer(DA, 1, session.UploadVectors(Array.Empty<V3>()));
                return;
            }

            using (var faceIndicesBuffer = session.UploadMeshFaceIndices(topoDesc.GetFacesSnapshot()))
            using (var analysis = session.AnalyzeMeshFaces(payload.VertexPositions, faceIndicesBuffer))
            {
                SetGpuBuffer(DA, 0, analysis.Centers.Retain());
                SetGpuBuffer(DA, 1, analysis.Normals.Retain());
            }
        }
    }
}
