using System;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.IO
{
    public class GhUploadVectors : GhostGpuComponentBase
    {
        public GhUploadVectors()
            : base("Upload Vectors", "UpVec",
                   "Uploads Grasshopper vectors into GPU memory. Packs each data tree branch into one resident GPU vector buffer handle, preserving paths, order, and empty branches.",
                   SubCategoryIO)
        {
        }

        public override Guid ComponentGuid => new Guid("77626F74-B831-4F42-999D-E3F9210B2120");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddVectorParameter("Vectors", "V", "Vectors to upload to GPU memory", GH_ParamAccess.tree);
            pManager.AddTextParameter("Format", "Fmt", "Optional storage format: 'V3D' (double) or 'V3F' (compact offset float). Defaults to V3D.", GH_ParamAccess.item);
            pManager.AddTextParameter("Residency", "Res", "Optional storage residency: 'Auto', 'Device', or 'Host'. Defaults to Auto.", GH_ParamAccess.item);
            pManager[1].Optional = true;
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Buffer", "B", "Resident GPU vector buffer per branch", GH_ParamAccess.tree);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            if (!DA.GetDataTree(0, out GH_Structure<GH_Vector> tree)) return;

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
            var outTree = new GH_Structure<GH_GpuBuffer>();

            foreach (GH_Path path in tree.Paths)
            {
                var branch = tree[path];
                if (branch == null || branch.Count == 0)
                {
                    var emptyBuf = session.UploadVectors(new V3[0], format, residency);
                    RegisterSolutionLease(emptyBuf);
                    outTree.Append(new GH_GpuBuffer(emptyBuf), path);
                    continue;
                }

                var v3s = new V3[branch.Count];
                for (int i = 0; i < branch.Count; i++)
                {
                    var item = branch[i];
                    if (item == null)
                    {
                        throw new InvalidOperationException($"Branch {path} contains a null vector at index {i}. Null inputs are not permitted in GPU uploads.");
                    }
                    var vec = item.Value;
                    if (!Check.Finite(vec.X) || !Check.Finite(vec.Y) || !Check.Finite(vec.Z))
                    {
                        throw new InvalidOperationException($"Branch {path} contains a non-finite vector at index {i}: ({vec.X}, {vec.Y}, {vec.Z}).");
                    }
                    v3s[i] = new V3(vec.X, vec.Y, vec.Z);
                }

                var buf = session.UploadVectors(v3s, format, residency);
                RegisterSolutionLease(buf);
                outTree.Append(new GH_GpuBuffer(buf), path);
            }

            SetGpuBufferTree(DA, 0, outTree);
        }
    }
}
