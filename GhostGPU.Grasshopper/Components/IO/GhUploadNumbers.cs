using System;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.IO
{
    public class GhUploadNumbers : GhostGpuComponentBase
    {
        public GhUploadNumbers()
            : base("Upload Numbers", "UpNum",
                   "Uploads Grasshopper numbers into GPU memory. Packs each data tree branch into one resident GPU scalar buffer handle, preserving paths, order, and empty branches.",
                   SubCategoryIO)
        {
        }

        public override Guid ComponentGuid => new Guid("C475A952-B4D2-48F4-A04B-7F01358CE102");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddNumberParameter("Numbers", "N", "Numbers to upload to GPU memory", GH_ParamAccess.tree);
            pManager.AddTextParameter("Format", "Fmt", "Optional storage format: 'F64' (double) or 'F32' (float). Defaults to F64.", GH_ParamAccess.item);
            pManager.AddTextParameter("Residency", "Res", "Optional storage residency: 'Auto', 'Device', or 'Host'. Defaults to Auto.", GH_ParamAccess.item);
            pManager[1].Optional = true;
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Buffer", "B", "Resident GPU scalar buffer per branch", GH_ParamAccess.tree);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            if (!DA.GetDataTree(0, out GH_Structure<GH_Number> tree)) return;

            string fmtStr = null;
            DA.GetData(1, ref fmtStr);
            string resStr = null;
            DA.GetData(2, ref resStr);

            var session = GetGpuSessionConcrete();

            ElementType format;
            if (string.IsNullOrWhiteSpace(fmtStr))
            {
                format = session.Options.DefaultScalarFormat;
            }
            else if (string.Equals(fmtStr, "F64", StringComparison.OrdinalIgnoreCase))
            {
                format = ElementType.F64;
            }
            else if (string.Equals(fmtStr, "F32", StringComparison.OrdinalIgnoreCase))
            {
                format = ElementType.F32;
            }
            else
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Unknown scalar format '{fmtStr}'. Expected 'F64' or 'F32'.");
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
                    var emptyBuf = session.UploadScalars(new double[0], format, residency);
                    RegisterSolutionLease(emptyBuf);
                    outTree.Append(new GH_GpuBuffer(emptyBuf), path);
                    continue;
                }

                var scalars = new double[branch.Count];
                for (int i = 0; i < branch.Count; i++)
                {
                    var item = branch[i];
                    if (item == null)
                    {
                        throw new InvalidOperationException($"Branch {path} contains a null number at index {i}. Null inputs are not permitted in GPU uploads.");
                    }
                    double val = item.Value;
                    if (!Check.Finite(val))
                    {
                        throw new InvalidOperationException($"Branch {path} contains a non-finite number at index {i}: {val}.");
                    }
                    scalars[i] = val;
                }

                var buf = session.UploadScalars(scalars, format, residency);
                RegisterSolutionLease(buf);
                outTree.Append(new GH_GpuBuffer(buf), path);
            }

            SetGpuBufferTree(DA, 0, outTree);
        }
    }
}
