using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhSphericalPoint : GhostGpuComponentBase
    {
        public GhSphericalPoint()
            : base("Spherical Point GPU", "SphPt",
                   "Constructs resident GPU points from spherical coordinates (radius, azimuth angle in radians in XY plane, elevation angle in radians from XY plane).",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("19CD16A7-7D9A-4BA4-8A87-A1C9369C9736");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Radius", "R", "Radial distance (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Azimuth", "θ", "Azimuth angle in radians in XY plane from +X (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Elevation", "φ", "Elevation angle in radians from XY plane (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "Pt", "Constructed resident GPU points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer rGoo = null;
            GH_GpuBuffer tGoo = null;
            GH_GpuBuffer pGoo = null;

            if (!DA.GetData(0, ref rGoo) || rGoo?.Value == null) return;
            if (!DA.GetData(1, ref tGoo) || tGoo?.Value == null) return;
            if (!DA.GetData(2, ref pGoo) || pGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.ConstructSphericalPoints(rGoo.Value, tGoo.Value, pGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
