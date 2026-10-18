using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhMeshTranslate : GhostGpuComponentBase
    {
        public GhMeshTranslate()
            : base("Mesh Translate GPU", "MTrans",
                   "Translates resident GPU mesh vertices by translation vector(s) without altering topology. " +
                   "Accepts uniform translation (vector count=1) or per-vertex displacement (count=vertex count).",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B105-7E23-4211-9A31-92F6F03F8005");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to translate", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Vector", "V", "Translation vector buffer (count=1 for uniform, or count=vertex count)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Translated GPU mesh with updated vertex positions", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            GH_GpuBuffer vGoo = null;

            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;
            if (!DA.GetData(1, ref vGoo) || vGoo == null || vGoo.Value == null) return;

            var payload = mGoo.Value;
            var vecBuf = vGoo.Value;

            if (vecBuf.Kind != BufferKind.Vectors)
                throw new ArgumentException("Translation input must be a Vectors buffer.", nameof(vGoo));

            int vCount = payload.TopologyTemplate.Vertices.Count;
            int vecCount = vecBuf.Count;

            if (vCount == 0)
            {
                SetGpuMesh(DA, 0, payload.Retain());
                return;
            }

            if (vecCount != vCount && vecCount != 1)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Translate requires vector count ({vecCount}) to equal 1 (uniform translation) " +
                    $"or mesh vertex count ({vCount}). Arbitrary lengths are rejected to preserve fixed topology.");
            }

            var session = GetGpuSession();
            var translatedVertices = session.Move(payload.VertexPositions, vecBuf);

            var dupMesh = payload.TopologyTemplate.DuplicateMesh();
            var newPayload = new GpuMeshPayload(
                dupMesh,
                translatedVertices,
                payload.VertexColours?.Retain()
            );

            SetGpuMesh(DA, 0, newPayload);
        }
    }
}
