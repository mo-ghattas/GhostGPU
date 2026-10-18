using System;
using Grasshopper.Kernel;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.PointVector
{
    public class GhCylindricalPoint : GhostGpuComponentBase
    {
        public GhCylindricalPoint()
            : base("Cylindrical Point GPU", "CylPt",
                   "Constructs resident GPU points from cylindrical coordinates (radius, angle in radians, elevation z).",
                   SubCategoryVector)
        {
        }

        public override Guid ComponentGuid => new Guid("54B27DF9-635E-4B02-B792-506B8F8A2825");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Radius", "R", "Radial coordinate distance (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Angle", "θ", "Azimuth angle in radians in XY plane (GPU scalar buffer)", GH_ParamAccess.item);
            pManager.AddParameter(new Param_GpuBuffer(), "Elevation", "Z", "Height coordinate along Z (GPU scalar buffer)", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Points", "Pt", "Constructed resident GPU points", GH_ParamAccess.item);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            GH_GpuBuffer rGoo = null;
            GH_GpuBuffer tGoo = null;
            GH_GpuBuffer zGoo = null;

            if (!DA.GetData(0, ref rGoo) || rGoo?.Value == null) return;
            if (!DA.GetData(1, ref tGoo) || tGoo?.Value == null) return;
            if (!DA.GetData(2, ref zGoo) || zGoo?.Value == null) return;

            var session = GetGpuSessionConcrete();
            var result = session.ConstructCylindricalPoints(rGoo.Value, tGoo.Value, zGoo.Value);
            SetGpuBuffer(DA, 0, result);
        }
    }
}
