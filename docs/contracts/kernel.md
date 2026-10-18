# Geometry kernel contract

GhostGPU.Kernel implements experimental NURBS evaluation, surface intersection, face splitting and joining. It targets net48/net8.0, depends on Core and Runtime, and does not reference RhinoCommon. Shared model conventions: full clamped knot vectors, homogeneous control points, and ON_Brep-style topology. These are public interface goals; see ../KNOWN-ISSUES.md for current gaps.

## M1 API (namespace GhostGPU.Kernel)
```csharp
public static class NurbsEval {
  // returns point and partial derivatives up to `order` (order 0..2): ders[k][l] = d^(k+l)S / du^k dv^l
  public static V3[,] SurfaceDerivatives(NurbsSurface s, double u, double v, int order);
  public static V3 SurfacePoint(NurbsSurface s, double u, double v);
  public static V3 SurfaceNormal(NurbsSurface s, double u, double v);          // unit, zero if degenerate
  public static V3[] CurveDerivatives(NurbsCurve c, double t, int order);      // 2D curves: Z = 0
  public static V3 CurvePoint(NurbsCurve c, double t);
  public static int FindSpan(double[] knots, int degree, int cvCount, double t);
}
public sealed class BezierPatch { public int SourceSurface; public Interval U, V; public int DegreeU, DegreeV;
  public double[] Cvs; /* homogeneous, (DegreeU+1)*(DegreeV+1)*4 */ public V3 BoxMin, BoxMax; }
public static class BezierDecomposition {
  public static List<BezierPatch> Decompose(NurbsSurface s, int sourceIndex);  // knot insertion to full multiplicity
  public static void Subdivide(BezierPatch p, out BezierPatch a, out BezierPatch b, bool splitU); // de Casteljau halves
}
public static class GpuEval {  // batched, uses GpuSession extension API (Rent/Adopt/GetKernel/Pin)
  public static GpuBuffer EvaluateSurfacePoints(GpuSession s, NurbsSurface srf, GpuBuffer uv /*Points: x=u,y=v*/);
  public static GpuBuffer PatchBoxes(GpuSession s, IReadOnlyList<BezierPatch> patches); // 2*n points min,max
}
```
Accuracy: match RhinoCommon `Surface.Evaluate`/`Curve.DerivativeAt` to 1e-12 relative on golden data.

## M2 API
`public static class Ssi { public static SsiResult Intersect(KBrep a, KBrep b, double tolerance, GpuSession session = null); }`
Pipeline: Bezier patches per face, GPU patch-pair box culling with recursive subdivision to seed boxes,
GPU Newton seeding, marching with a pinned-coordinate constrained Newton corrector (largest |du_i| pinned)
with curvature-based step control, branch tracing to face trim boundaries (respect trims, not only the
surface domain), loop closure, duplicate-branch merge, then a cubic non-rational NURBS fit of the 3D curve and
both UV pullbacks within tolerance. Validate against live `Intersection.BrepBrep`: same branch count, and the
max distance between curves < 2*tol, on box/sphere/cylinder/freeform/Steinmetz/tangent-near-miss cases.

## M3 API
`public static class FaceSplit { public static List<KBrep> Split(KBrep brep, IReadOnlyList<SsiBranch> branches, bool branchesOnA, double tolerance); }`
UV arrangement per face (existing loops + pullback curves, as polylines within tolerance), DCEL, cycle
tracing, outer/inner containment, periodic seam handling, new faces with exact trims (the 2D pullback
curves) and shared new edges (the 3D SSI curves). Validate against live `Brep.Split`: piece count, face
counts and areas (rel 1e-6).

## M4 API
`public static class BrepJoin { public static List<KBrep> Join(IReadOnlyList<KBrep> pieces, double tolerance); }`
Vertex clustering, T-junction edge splitting (vertex within tol of another naked edge interior), edge
matching by sampled max deviation <= tol, trim orientation reconciliation, mated trim types, edge
tolerance update. Validate against live `JoinBreps`: count, IsSolid, naked edge count, including the
tolerance step (offset 0.9*tol joins, 1.1*tol does not).

## Golden data
`tests/fixtures/golden/` holds JSON dumped from live Rhino by a RhinoCommon exporter script
(`scripts/ExportGolden.csx`, run inside a licensed Rhino host): surfaces (CVs,
knots), evaluations with derivatives, Intersection.BrepBrep curves, Split piece stats, JoinBreps stats.
