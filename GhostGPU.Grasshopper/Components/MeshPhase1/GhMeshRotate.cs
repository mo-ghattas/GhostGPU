using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.MeshPhase1
{
    public class GhMeshRotate : GhostGpuComponentBase
    {
        public GhMeshRotate()
            : base("Mesh Rotate GPU", "MRot",
                   "Rotates resident GPU mesh vertices around an arbitrary 3D axis by an angle in radians without altering topology.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("A194B107-7E23-4211-9A31-92F6F03F8007");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh to rotate", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Origin", "O", "Axis origin point buffer (count=1)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Axis", "A", "Axis direction vector buffer (count=1)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Angle", "R", "Rotation angle in radians (scalar buffer, count=1)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Rotated GPU mesh with updated vertex positions", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuMesh mGoo = null;
            GH_GpuBuffer oGoo = null;
            GH_GpuBuffer aGoo = null;
            GH_GpuBuffer rGoo = null;

            if (!DA.GetData(0, ref mGoo) || mGoo == null || mGoo.Value == null) return;
            if (!DA.GetData(1, ref oGoo) || oGoo == null || oGoo.Value == null) return;
            if (!DA.GetData(2, ref aGoo) || aGoo == null || aGoo.Value == null) return;
            if (!DA.GetData(3, ref rGoo) || rGoo == null || rGoo.Value == null) return;

            var payload = mGoo.Value;
            var originBuf = oGoo.Value;
            var axisBuf = aGoo.Value;
            var angleBuf = rGoo.Value;

            if (originBuf.Kind != BufferKind.Points)
                throw new ArgumentException("Axis origin input must be a Points buffer.", nameof(oGoo));
            if (axisBuf.Kind != BufferKind.Vectors)
                throw new ArgumentException("Axis direction input must be a Vectors buffer.", nameof(aGoo));
            if (angleBuf.Kind != BufferKind.Scalars)
                throw new ArgumentException("Angle input must be a Scalars buffer (radians).", nameof(rGoo));

            int vCount = payload.TopologyTemplate.Vertices.Count;
            if (vCount == 0)
            {
                SetGpuMesh(DA, 0, payload.Retain());
                return;
            }

            if (originBuf.Count != 1 && originBuf.Count != vCount)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Rotate requires axis origin count ({originBuf.Count}) to equal 1 or mesh vertex count ({vCount}). Arbitrary lengths are rejected to preserve fixed topology.");
            }
            if (axisBuf.Count != 1 && axisBuf.Count != vCount)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Rotate requires axis direction count ({axisBuf.Count}) to equal 1 or mesh vertex count ({vCount}). Arbitrary lengths are rejected to preserve fixed topology.");
            }
            if (angleBuf.Count != 1 && angleBuf.Count != vCount)
            {
                throw new InvalidOperationException(
                    $"[GhostGPU] Mesh Rotate requires angle count ({angleBuf.Count}) to equal 1 or mesh vertex count ({vCount}). Arbitrary lengths are rejected to preserve fixed topology.");
            }

            var session = GetGpuSessionConcrete();
            var rotatedVertices = session.MeshRotate(payload.VertexPositions, originBuf, axisBuf, angleBuf);

            var dupMesh = payload.TopologyTemplate.DuplicateMesh();
            try
            {
                dupMesh.FaceNormals.Clear();
                dupMesh.Normals.Clear();
            }
            catch { }

            var newPayload = new GpuMeshPayload(
                dupMesh,
                rotatedVertices,
                payload.VertexColours?.Retain()
            );

            SetGpuMesh(DA, 0, newPayload);
        }
    }
}
