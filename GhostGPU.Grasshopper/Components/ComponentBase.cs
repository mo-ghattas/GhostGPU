using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using GhostGPU.Core;
using GhostGPU.Runtime;
using GhostGPU.Grasshopper.Parameters;
using GhostGPU.Grasshopper.Session;

namespace GhostGPU.Grasshopper.Components
{
    /// <summary>
    /// Base class for all GhostGPU Grasshopper components.
    /// Manages per-document session resolution, category constants, and solution-level GPU lease cleanup.
    /// Cleanups occur once per solution pass (BeforeSolveInstance / ClearData / RemovedFromDocument)
    /// to safely support multi-branch item iterations without premature lease disposal.
    /// </summary>
    public abstract class GhostGpuComponentBase : GH_Component
    {
        public const string CategoryName = "GhostGPU";
        public const string SubCategoryIO = "00 | IO";
        public const string SubCategoryVector = "01 | Vector";
        public const string SubCategoryMath = "02 | Math";
        public const string SubCategoryMesh = "03 | Mesh";
        public const string SubCategoryDiagnostics = "04 | Diagnostics";
        public const string SubCategoryBooleans = "05 | Booleans";

        private readonly List<IDisposable> _solutionLeases = new List<IDisposable>();
        private readonly object _leaseLock = new object();

        protected GhostGpuComponentBase(string name, string nickname, string description, string subCategory)
            : base(name, nickname, description, CategoryName, subCategory)
        {
        }

        protected IGpuSession GetGpuSession()
        {
            var doc = OnPingDocument();
            return GpuSessionManager.GetSession(doc);
        }

        protected GpuSession GetGpuSessionConcrete()
        {
            return (GpuSession)GetGpuSession();
        }

        /// <summary>
        /// Registers a GPU lease or payload produced during the current solution pass
        /// so it is deterministically disposed at the start of the next solution.
        /// </summary>
        protected void RegisterSolutionLease(IDisposable lease)
        {
            if (lease == null) return;
            lock (_leaseLock)
            {
                _solutionLeases.Add(lease);
            }
        }

        protected void SetGpuBuffer(IGH_DataAccess DA, int paramIndex, GpuBuffer buffer)
        {
            if (buffer != null)
            {
                var goo = new GH_GpuBuffer(buffer);
                RegisterSolutionLease(goo);
                DA.SetData(paramIndex, goo);
            }
        }

        protected void SetGpuMesh(IGH_DataAccess DA, int paramIndex, GpuMeshPayload payload)
        {
            if (payload != null)
            {
                var goo = new GH_GpuMesh(payload);
                RegisterSolutionLease(goo);
                DA.SetData(paramIndex, goo);
            }
        }

        protected void SetGpuBufferTree(IGH_DataAccess DA, int paramIndex, GH_Structure<GH_GpuBuffer> tree)
        {
            if (tree != null)
            {
                foreach (GH_Path path in tree.Paths)
                {
                    var branch = tree[path];
                    if (branch != null)
                    {
                        foreach (GH_GpuBuffer goo in branch)
                        {
                            if (goo != null)
                            {
                                RegisterSolutionLease(goo);
                            }
                        }
                    }
                }
                DA.SetDataTree(paramIndex, tree);
            }
        }

        protected void SetGpuMeshTree(IGH_DataAccess DA, int paramIndex, GH_Structure<GH_GpuMesh> tree)
        {
            if (tree != null)
            {
                foreach (GH_Path path in tree.Paths)
                {
                    var branch = tree[path];
                    if (branch != null)
                    {
                        foreach (GH_GpuMesh goo in branch)
                        {
                            if (goo != null)
                            {
                                RegisterSolutionLease(goo);
                            }
                        }
                    }
                }
                DA.SetDataTree(paramIndex, tree);
            }
        }

        private void CleanupSolutionLeases()
        {
            lock (_leaseLock)
            {
                foreach (var lease in _solutionLeases)
                {
                    try { lease.Dispose(); } catch { }
                }
                _solutionLeases.Clear();
            }
        }

        protected override void BeforeSolveInstance()
        {
            CleanupSolutionLeases();
            base.BeforeSolveInstance();
        }

        public override void ClearData()
        {
            CleanupSolutionLeases();
            base.ClearData();
        }

        public override void RemovedFromDocument(GH_Document document)
        {
            CleanupSolutionLeases();
            base.RemovedFromDocument(document);
        }

        protected sealed override void SolveInstance(IGH_DataAccess DA)
        {
            try
            {
                Solve(DA);
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
            }
        }

        protected abstract void Solve(IGH_DataAccess DA);

        protected override System.Drawing.Bitmap Icon => Icons.GhostGpuIconProvider.GetIcon(this);
    }
}
