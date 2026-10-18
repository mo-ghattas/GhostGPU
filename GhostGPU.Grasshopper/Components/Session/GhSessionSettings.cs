using System;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Session;

namespace GhostGPU.Grasshopper.Components.Session
{
    public class GhSessionSettings : GhostGpuComponentBase
    {
        public GhSessionSettings()
            : base("GPU Session Settings", "GpuSettings",
                   "Configures GPU SessionOptions (memory budget, pooling, staging ring, host execution threshold, spill policy, and default data formats) and recreates the active document session.",
                   SubCategoryDiagnostics)
        {
        }

        public override Guid ComponentGuid => new Guid("340D18FE-7F4F-4DEE-B587-C1D7E38A3013");

        public override GH_Exposure Exposure => GH_Exposure.primary;

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddIntegerParameter("Memory Limit (MB)", "MemMB", "Maximum GPU device memory budget in megabytes (default 512 MB)", GH_ParamAccess.item, 512);
            pManager.AddIntegerParameter("Pool Cache (MB)", "PoolMB", "Device memory pool cache limit in megabytes (default 128 MB)", GH_ParamAccess.item, 128);
            pManager.AddIntegerParameter("Staging Slots", "Slots", "Pinned host staging ring slot count (default 4)", GH_ParamAccess.item, 4);
            pManager.AddIntegerParameter("Staging Slot (MB)", "SlotMB", "Pinned host staging ring slot size in megabytes (default 16 MB)", GH_ParamAccess.item, 16);
            pManager.AddIntegerParameter("Host Threshold", "HostThresh", "Element count threshold below which host execution runs (default 65536)", GH_ParamAccess.item, 65536);
            pManager.AddBooleanParameter("Spill to Host", "Spill", "If true, spills LRU device buffers to host on budget pressure; if false, throws on OOM (default true)", GH_ParamAccess.item, true);
            pManager.AddTextParameter("Default Vector Format", "VecFmt", "Default format for uploaded vectors/points: 'V3D' or 'V3F' (default 'V3D')", GH_ParamAccess.item, "V3D");
            pManager.AddTextParameter("Default Scalar Format", "ScalarFmt", "Default format for uploaded scalars: 'F64' or 'F32' (default 'F64')", GH_ParamAccess.item, "F64");
            pManager.AddTextParameter("Backend", "Backend", "GPU Backend: 'CudaOnly' or 'CpuDebug' (default 'CudaOnly')", GH_ParamAccess.item, "CudaOnly");

            for (int i = 0; i < pManager.ParamCount; i++)
            {
                pManager[i].Optional = true;
            }
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Session Info", "Info", "Active GPU session configuration details", GH_ParamAccess.item);
            pManager.AddTextParameter("Session ID", "ID", "Active document GPU session GUID", GH_ParamAccess.item);
        }

        private SessionOptions _lastOptions;
        private Guid _lastDocId = Guid.Empty;

        private static bool OptionsEqual(SessionOptions a, SessionOptions b)
        {
            if (a == null || b == null) return false;
            return a.MemoryLimitBytes == b.MemoryLimitBytes &&
                   a.PoolCacheBytes == b.PoolCacheBytes &&
                   a.StagingSlots == b.StagingSlots &&
                   a.StagingSlotBytes == b.StagingSlotBytes &&
                   a.HostWorkThreshold == b.HostWorkThreshold &&
                   a.Spill == b.Spill &&
                   a.DefaultVectorFormat == b.DefaultVectorFormat &&
                   a.DefaultScalarFormat == b.DefaultScalarFormat &&
                   a.Backend == b.Backend;
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            int memMb = 512;
            int poolMb = 128;
            int slots = 4;
            int slotMb = 16;
            int hostThresh = 65536;
            bool spillToHost = true;
            string vecFmtStr = "V3D";
            string scalarFmtStr = "F64";
            string backendStr = "CudaOnly";

            DA.GetData(0, ref memMb);
            DA.GetData(1, ref poolMb);
            DA.GetData(2, ref slots);
            DA.GetData(3, ref slotMb);
            DA.GetData(4, ref hostThresh);
            DA.GetData(5, ref spillToHost);
            DA.GetData(6, ref vecFmtStr);
            DA.GetData(7, ref scalarFmtStr);
            DA.GetData(8, ref backendStr);

            var vecFmt = string.Equals(vecFmtStr, "V3F", StringComparison.OrdinalIgnoreCase)
                ? ElementType.V3F
                : ElementType.V3D;

            var scalarFmt = string.Equals(scalarFmtStr, "F32", StringComparison.OrdinalIgnoreCase)
                ? ElementType.F32
                : ElementType.F64;

            var backend = string.Equals(backendStr, "CpuDebug", StringComparison.OrdinalIgnoreCase)
                ? GpuBackend.CpuDebug
                : GpuBackend.CudaOnly;

            var options = new SessionOptions
            {
                MemoryLimitBytes = System.Math.Max(16L * 1024 * 1024, (long)memMb * 1024 * 1024),
                PoolCacheBytes = System.Math.Max(0L, (long)poolMb * 1024 * 1024),
                StagingSlots = System.Math.Max(1, slots),
                StagingSlotBytes = System.Math.Max(1L * 1024 * 1024, (long)slotMb * 1024 * 1024),
                HostWorkThreshold = System.Math.Max(0, hostThresh),
                Spill = spillToHost ? SpillPolicy.SpillToHost : SpillPolicy.Fail,
                DefaultVectorFormat = vecFmt,
                DefaultScalarFormat = scalarFmt,
                Backend = backend
            };

            var doc = OnPingDocument();
            IGpuSession session;

            bool optionsChanged = _lastOptions == null ||
                                  doc == null ||
                                  doc.DocumentID != _lastDocId ||
                                  !OptionsEqual(_lastOptions, options);

            if (optionsChanged)
            {
                _lastOptions = options;
                _lastDocId = doc != null ? doc.DocumentID : Guid.Empty;
                session = GpuSessionManager.RecreateSession(doc, options);

                if (doc != null)
                {
                    doc.ScheduleSolution(1, d =>
                    {
                        foreach (var obj in d.Objects)
                        {
                            if (obj is GhostGpuComponentBase comp && comp != this)
                            {
                                comp.ExpireSolution(false);
                            }
                        }
                    });
                }
            }
            else
            {
                session = GpuSessionManager.GetSession(doc, options);
            }

            string info =
                $"GhostGPU Document Session: {session.SessionId}\r\n" +
                $"Device: {session.DeviceName} (Hardware Accelerated: {session.IsHardwareAccelerated})\r\n" +
                $"Backend: {options.Backend}\r\n" +
                $"Memory Budget: {options.MemoryLimitBytes / (1024.0 * 1024.0):F0} MB\r\n" +
                $"Pool Cache: {options.PoolCacheBytes / (1024.0 * 1024.0):F0} MB\r\n" +
                $"Staging Ring: {options.StagingSlots} slots x {options.StagingSlotBytes / (1024.0 * 1024.0):F0} MB\r\n" +
                $"Host Work Threshold: {options.HostWorkThreshold} elements\r\n" +
                $"Spill Policy: {options.Spill}\r\n" +
                $"Default Formats: Vector={options.DefaultVectorFormat}, Scalar={options.DefaultScalarFormat}";

            DA.SetData(0, info);
            DA.SetData(1, session.SessionId.ToString());
        }
    }
}
