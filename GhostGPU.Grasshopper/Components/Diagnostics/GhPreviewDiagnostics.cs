using System;
using System.Drawing;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Rhino.Display;
using Rhino.Geometry;
using RhinoMesh = Rhino.Geometry.Mesh;
using SystemMath = System.Math;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.Diagnostics
{
    public class GhPreviewDiagnostics : GhostGpuComponentBase
    {
        private readonly List<RhinoMesh> _cachedPreviewMeshes = new List<RhinoMesh>();
        private readonly List<Point3d> _cachedPreviewPoints = new List<Point3d>();
        private BoundingBox _clippingBox = BoundingBox.Empty;

        public GhPreviewDiagnostics()
            : base("GPU Preview & Diagnostics", "GpuDiag",
                   "Provides Grasshopper viewport preview and real-time device memory diagnostics. Explicitly labelled readback preview boundary cached per solve.",
                   SubCategoryDiagnostics)
        {
        }

        public override Guid ComponentGuid => new Guid("99AA8FDC-EAF4-482B-89E9-183F25A0F119");

        public override bool IsPreviewCapable => true;

        public override BoundingBox ClippingBox => _clippingBox;

        protected override void BeforeSolveInstance()
        {
            ClearPreview();
            base.BeforeSolveInstance();
        }

        private void ClearPreview()
        {
            foreach (var mesh in _cachedPreviewMeshes) mesh.Dispose();
            _cachedPreviewMeshes.Clear();
            _cachedPreviewPoints.Clear();
            _clippingBox = BoundingBox.Empty;
        }

        public override void ClearData()
        {
            ClearPreview();
            base.ClearData();
        }

        public override void RemovedFromDocument(GH_Document document)
        {
            ClearPreview();
            base.RemovedFromDocument(document);
        }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Geometry", "G", "GPU resident geometry to preview (GPU buffer or GPU mesh)", GH_ParamAccess.item);
            pManager.AddBooleanParameter("Enable Preview", "P", "Enable viewport display of GPU geometry", GH_ParamAccess.item, true);
            pManager[0].Optional = true;
            pManager[1].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Diagnostics", "D", "Snapshot metrics and session diagnostics", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Active Bytes", "AB", "Current active GPU resident bytes", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Live Allocations", "LA", "Count of currently allocated resident GPU buffers", GH_ParamAccess.item);
            pManager.AddTextParameter("Device Name", "Dev", "Active GPU hardware accelerator name", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            var session = GetGpuSession();

            bool enablePreview = true;
            DA.GetData(1, ref enablePreview);

            object geomObj = null;
            DA.GetData(0, ref geomObj);

            if (enablePreview && geomObj != null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    "Notice: Viewport preview downloads geometry from GPU memory to host CPU for display. Not zero-copy graphics interop.");

                if (geomObj is GH_GpuMesh meshGoo && meshGoo.Value != null)
                {
                    var mesh = MeshReadback.Materialize(session, meshGoo.Value);
                    _cachedPreviewMeshes.Add(mesh);
                    _clippingBox.Union(mesh.GetBoundingBox(true));
                }
                else if (geomObj is GH_GpuBuffer bufGoo && bufGoo.Value != null && bufGoo.Value.Kind == BufferKind.Points)
                {
                    var v3s = session.DownloadPoints(bufGoo.Value);
                    var pts = new Point3d[v3s.Length];
                    var bbox = BoundingBox.Empty;

                    for (int i = 0; i < v3s.Length; i++)
                    {
                        pts[i] = new Point3d(v3s[i].X, v3s[i].Y, v3s[i].Z);
                        bbox.Union(pts[i]);
                    }

                    _cachedPreviewPoints.AddRange(pts);
                    _clippingBox.Union(bbox);
                }
                else throw new ArgumentException("Preview requires a resident point buffer or mesh.");
            }

            var metrics = session.Metrics;
            string diagText =
                $"=== GhostGPU Session Diagnostics ===\r\n" +
                $"Device: {session.DeviceName} (Hardware Accelerated: {session.IsHardwareAccelerated})\r\n" +
                $"Active Resident Memory: {metrics.ActiveBytes / (1024.0 * 1024.0):F2} MB ({metrics.ActiveBytes:N0} bytes)\r\n" +
                $"Peak Resident Memory: {metrics.PeakBytes / (1024.0 * 1024.0):F2} MB ({metrics.PeakBytes:N0} bytes)\r\n" +
                $"Live Buffer Allocations: {metrics.LiveAllocations}\r\n" +
                $"Total Uploaded: {metrics.UploadedBytes / 1024.0:F2} KB\r\n" +
                $"Total Downloaded: {metrics.DownloadedBytes / 1024.0:F2} KB\r\n" +
                $"Kernel Dispatches: {metrics.KernelDispatches}\r\n" +
                $"Host Dispatches: {metrics.HostDispatches}\r\n" +
                $"Device Synchronizations: {metrics.Synchronizations}\r\n" +
                $"Pooled Device Memory: {metrics.PooledBytes / (1024.0 * 1024.0):F2} MB ({metrics.PooledBytes:N0} bytes)\r\n" +
                $"Pool Hits / Misses: {metrics.PoolHits} / {metrics.PoolMisses}\r\n" +
                $"Host Resident Memory: {metrics.HostResidentBytes / (1024.0 * 1024.0):F2} MB ({metrics.HostResidentBytes:N0} bytes)\r\n" +
                $"Spilled Bytes: {metrics.SpilledBytes / 1024.0:F2} KB ({metrics.SpilledBytes:N0} bytes)\r\n" +
                $"Staged Transfer Bytes: {metrics.StagedBytes / 1024.0:F2} KB ({metrics.StagedBytes:N0} bytes)\r\n" +
                $"Preview Readback Boundary: Cached per solve (Explicit CPU download)";

            DA.SetData(0, diagText);
            DA.SetData(1, (int)SystemMath.Min(metrics.ActiveBytes, int.MaxValue));
            DA.SetData(2, (int)SystemMath.Min(metrics.LiveAllocations, int.MaxValue));
            DA.SetData(3, session.DeviceName);
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            if (_cachedPreviewPoints != null)
            {
                foreach (var pt in _cachedPreviewPoints)
                {
                    args.Display.DrawPoint(pt, PointStyle.RoundSimple, 3, Color.FromArgb(0, 180, 255));
                }
            }

            foreach (var mesh in _cachedPreviewMeshes)
            {
                args.Display.DrawMeshWires(mesh, Color.FromArgb(80, 0, 0, 0));
            }
        }

        public override void DrawViewportMeshes(IGH_PreviewArgs args)
        {
            foreach (var mesh in _cachedPreviewMeshes)
            {
                if (mesh.VertexColors.Count == mesh.Vertices.Count)
                {
                    args.Display.DrawMeshFalseColors(mesh);
                }
                else
                {
                    using (var material = new DisplayMaterial(Color.FromArgb(200, 220, 240)))
                        args.Display.DrawMeshShaded(mesh, material);
                }
            }
        }
    }
}
