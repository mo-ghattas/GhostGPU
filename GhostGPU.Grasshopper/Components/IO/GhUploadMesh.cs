using System;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using RhinoMesh = Rhino.Geometry.Mesh;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.IO
{
    public class GhUploadMesh : GhostGpuComponentBase
    {
        public GhUploadMesh()
            : base("Upload Mesh", "UpMsh",
                   "Uploads Grasshopper meshes to GPU memory. Retains host-side topology template (faces, normals, quad topology) and uploads double-precision vertices and optional vertex colours to resident GPU buffers.",
                   SubCategoryMesh)
        {
        }

        public override Guid ComponentGuid => new Guid("54B55E5D-47C0-463E-B19E-6C0D9B9DF103");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Meshes to upload to GPU memory", GH_ParamAccess.tree);
            pManager.AddTextParameter("Format", "Fmt", "Optional storage format for vertex positions: 'V3D' or 'V3F'. Defaults to V3D.", GH_ParamAccess.item);
            pManager.AddTextParameter("Residency", "Res", "Optional storage residency: 'Auto', 'Device', or 'Host'. Defaults to Auto.", GH_ParamAccess.item);
            pManager[1].Optional = true;
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuMesh(), "GPU Mesh", "GM", "Resident GPU meshes with immutable host topology", GH_ParamAccess.tree);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            if (!DA.GetDataTree(0, out GH_Structure<GH_Mesh> tree)) return;

            string fmtStr = null;
            DA.GetData(1, ref fmtStr);
            string resStr = null;
            DA.GetData(2, ref resStr);

            var session = GetGpuSessionConcrete();

            ElementType format;
            if (string.IsNullOrWhiteSpace(fmtStr))
            {
                format = session.Options.DefaultVectorFormat;
            }
            else if (string.Equals(fmtStr, "V3D", StringComparison.OrdinalIgnoreCase))
            {
                format = ElementType.V3D;
            }
            else if (string.Equals(fmtStr, "V3F", StringComparison.OrdinalIgnoreCase))
            {
                format = ElementType.V3F;
            }
            else
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Unknown vector format '{fmtStr}'. Expected 'V3D' or 'V3F'.");
                return;
            }

            Residency residency;
            if (string.IsNullOrWhiteSpace(resStr) || string.Equals(resStr, "Auto", StringComparison.OrdinalIgnoreCase))
            {
                residency = Residency.Auto;
            }
            else if (string.Equals(resStr, "Device", StringComparison.OrdinalIgnoreCase))
            {
                residency = Residency.Device;
            }
            else if (string.Equals(resStr, "Host", StringComparison.OrdinalIgnoreCase))
            {
                residency = Residency.Host;
            }
            else
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Unknown residency '{resStr}'. Expected 'Auto', 'Device', or 'Host'.");
                return;
            }
            var outTree = new GH_Structure<GH_GpuMesh>();

            foreach (GH_Path path in tree.Paths)
            {
                var branch = tree[path];
                if (branch == null || branch.Count == 0)
                {
                    outTree.EnsurePath(path);
                    continue;
                }

                for (int mIdx = 0; mIdx < branch.Count; mIdx++)
                {
                    var item = branch[mIdx];
                    if (item == null || item.Value == null)
                    {
                        throw new InvalidOperationException($"Branch {path} contains a null mesh at index {mIdx}. Null mesh upload is not permitted.");
                    }

                    RhinoMesh hostMesh = item.Value;
                    int vCount = hostMesh.Vertices.Count;

                    var v3s = new V3[vCount];
                    for (int i = 0; i < vCount; i++)
                    {
                        var pt = hostMesh.Vertices.Point3dAt(i);
                        if (!Check.Finite(pt.X) || !Check.Finite(pt.Y) || !Check.Finite(pt.Z))
                        {
                            throw new InvalidOperationException($"Mesh in branch {path} contains non-finite vertex coordinates at index {i}: ({pt.X}, {pt.Y}, {pt.Z}).");
                        }
                        v3s[i] = new V3(pt.X, pt.Y, pt.Z);
                    }

                    var vBuf = session.UploadPoints(v3s, format, residency);
                    RegisterSolutionLease(vBuf);

                    GpuBuffer cBuf = null;
                    if (hostMesh.VertexColors != null && hostMesh.VertexColors.Count == vCount)
                    {
                        var argbs = new uint[vCount];
                        for (int i = 0; i < vCount; i++)
                        {
                            argbs[i] = (uint)hostMesh.VertexColors[i].ToArgb();
                        }
                        cBuf = session.UploadColours(argbs);
                        RegisterSolutionLease(cBuf);
                    }

                    // Store an owned clone of the topology template
                    var payload = new GpuMeshPayload(hostMesh.DuplicateMesh(), vBuf, cBuf);
                    RegisterSolutionLease(payload);
                    outTree.Append(new GH_GpuMesh(payload), path);
                }
            }

            SetGpuMeshTree(DA, 0, outTree);
        }
    }
}
