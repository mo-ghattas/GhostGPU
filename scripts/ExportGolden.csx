// GhostGPU Golden Data Exporter for Rhino's RhinoCommon C# script runner.
// Globals: doc (RhinoDoc), output (StringBuilder)
// Pre-imported: System, System.Linq, Rhino, Rhino.Geometry
// Note: beware 'Environment' is ambiguous: use System.Environment.

using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using Rhino;
using Rhino.Geometry;
using Rhino.Geometry.Collections;
using Rhino.Geometry.Intersect;

output.AppendLine("GhostGPU Golden Exporter starting at " + DateTime.UtcNow.ToString("o"));

string repoRoot = System.Environment.GetEnvironmentVariable("GHOSTGPU_REPO_ROOT");
if (string.IsNullOrWhiteSpace(repoRoot)) throw new InvalidOperationException("Set GHOSTGPU_REPO_ROOT to the repository directory before running this script.");
string goldenDir = Path.Combine(repoRoot, "tests", "fixtures", "golden");
Directory.CreateDirectory(goldenDir);

var jsonOptions = new JsonSerializerOptions
{
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.Never
};

// =========================================================================
// 1. SURFACES.JSON
// =========================================================================
output.AppendLine("Building surfaces...");
var surfaceCases = new List<SurfaceCaseDto>();
var rnd = new Random(42);

// Case 1: Plane
var planeSrf = new PlaneSurface(Plane.WorldXY, new Interval(-10, 10), new Interval(-5, 15)).ToNurbsSurface();
surfaceCases.Add(BuildSurfaceCase("plane", planeSrf));

// Case 2: Rational Cylinder
var cylSrf = new Cylinder(new Circle(Plane.WorldXY, 4.0), 12.0).ToNurbsSurface();
surfaceCases.Add(BuildSurfaceCase("cylinder_rational", cylSrf));

// Case 3: Rational Sphere
var sphSrf = new Sphere(Point3d.Origin, 6.0).ToNurbsSurface();
surfaceCases.Add(BuildSurfaceCase("sphere_rational", sphSrf));

// Case 4: Rational Torus
var torSrf = new Torus(Plane.WorldXY, 10.0, 3.0).ToNurbsSurface();
surfaceCases.Add(BuildSurfaceCase("torus_rational", torSrf));

// Case 5: Rational Cone
var coneSrf = new Cone(Plane.WorldXY, 10.0, 4.0).ToNurbsSurface();
surfaceCases.Add(BuildSurfaceCase("cone_rational", coneSrf));

// Case 6: Random Bicubic
var bicubicPts = new Point3d[36];
for (int i = 0; i < 6; i++)
{
    for (int j = 0; j < 6; j++)
    {
        double x = i * 4.0 + (rnd.NextDouble() - 0.5);
        double y = j * 4.0 + (rnd.NextDouble() - 0.5);
        double z = Math.Sin(i * 0.8) * Math.Cos(j * 0.8) * 3.0;
        bicubicPts[i * 6 + j] = new Point3d(x, y, z);
    }
}
var bicubicSrf = NurbsSurface.CreateFromPoints(bicubicPts, 6, 6, 3, 3);
surfaceCases.Add(BuildSurfaceCase("random_bicubic", bicubicSrf));

// Case 7: Degree (2, 5)
var pts25 = new Point3d[4 * 8];
for (int i = 0; i < 4; i++)
{
    for (int j = 0; j < 8; j++)
    {
        double x = i * 5.0 + (rnd.NextDouble() - 0.5);
        double y = j * 3.0 + (rnd.NextDouble() - 0.5);
        double z = Math.Sin(j * 0.9) * 2.5;
        pts25[i * 8 + j] = new Point3d(x, y, z);
    }
}
var srf25 = NurbsSurface.CreateFromPoints(pts25, 4, 8, 2, 5);
surfaceCases.Add(BuildSurfaceCase("degree_2_5", srf25));

// Case 8: Periodic Revolved Surface
var circleCurve = new Circle(new Plane(new Point3d(8, 0, 0), Vector3d.XAxis, Vector3d.ZAxis), 2.5);
var revSrf = RevSurface.Create(circleCurve.ToNurbsCurve(), new Line(Point3d.Origin, new Point3d(0, 0, 10)));
var periodicSrf = revSrf.ToNurbsSurface();
surfaceCases.Add(BuildSurfaceCase("periodic_revolve", periodicSrf));

string surfacesJsonPath = Path.Combine(goldenDir, "surfaces.json");
File.WriteAllText(surfacesJsonPath, JsonSerializer.Serialize(surfaceCases, jsonOptions));
output.AppendLine($"Exported {surfaceCases.Count} surfaces to {surfacesJsonPath}");

// =========================================================================
// 2. CURVES.JSON
// =========================================================================
output.AppendLine("Building curves...");
var curveCases = new List<CurveCaseDto>();

// Case 1: 3D Line Degree 1
var lineCrv = new Line(new Point3d(0, 0, 0), new Point3d(12, 8, 5)).ToNurbsCurve();
curveCases.Add(BuildCurveCase("line_degree1", lineCrv, 3));

// Case 2: 3D Parabola Degree 2
var parabolaCrv = NurbsCurve.Create(false, 2, new Point3d[] {
    new Point3d(0, 0, 0),
    new Point3d(5, 12, 3),
    new Point3d(10, 0, 6)
});
curveCases.Add(BuildCurveCase("parabola_degree2", parabolaCrv, 3));

// Case 3: 3D Cubic Spline Degree 3
var cubicCrv = NurbsCurve.Create(false, 3, new Point3d[] {
    new Point3d(0, 0, 0),
    new Point3d(2, 6, 4),
    new Point3d(6, -3, 2),
    new Point3d(9, 8, -1),
    new Point3d(12, 1, 5)
});
curveCases.Add(BuildCurveCase("cubic_spline_degree3", cubicCrv, 3));

// Case 4: 3D Rational Arc
var arcCrv = new Arc(new Circle(Plane.WorldXY, 6.0), Math.PI * 1.25).ToNurbsCurve();
curveCases.Add(BuildCurveCase("rational_arc_degree2", arcCrv, 3));

// Case 5: 3D Periodic Circle
var circleCrv = new Circle(Plane.WorldXY, 8.0).ToNurbsCurve();
curveCases.Add(BuildCurveCase("periodic_circle", circleCrv, 3));

// Case 6: 3D High Degree 5
var crvDeg5 = NurbsCurve.Create(false, 5, new Point3d[] {
    new Point3d(0, 0, 0),
    new Point3d(2, 4, 1),
    new Point3d(4, 1, 5),
    new Point3d(7, -2, 2),
    new Point3d(9, 5, -2),
    new Point3d(12, 3, 4),
    new Point3d(15, 0, 0)
});
curveCases.Add(BuildCurveCase("high_degree5", crvDeg5, 3));

// Case 7: 2D Trim Line
var trimLine = new Line(new Point3d(0.1, 0.1, 0), new Point3d(0.9, 0.8, 0)).ToNurbsCurve();
curveCases.Add(BuildCurveCase("trim_2d_line", trimLine, 2));

// Case 8: 2D Trim Cubic Spline
var trimCubic = NurbsCurve.Create(false, 3, new Point3d[] {
    new Point3d(0.05, 0.1, 0),
    new Point3d(0.2, 0.9, 0),
    new Point3d(0.8, 0.1, 0),
    new Point3d(0.95, 0.85, 0)
});
curveCases.Add(BuildCurveCase("trim_2d_cubic", trimCubic, 2));

// Case 9: 2D Trim Rational Arc
var trimArc = new Arc(new Circle(Plane.WorldXY, 0.4), new Interval(0.1, Math.PI * 0.8)).ToNurbsCurve();
curveCases.Add(BuildCurveCase("trim_2d_arc_rational", trimArc, 2));

string curvesJsonPath = Path.Combine(goldenDir, "curves.json");
File.WriteAllText(curvesJsonPath, JsonSerializer.Serialize(curveCases, jsonOptions));
output.AppendLine($"Exported {curveCases.Count} curves to {curvesJsonPath}");

// =========================================================================
// 3. BREP PAIRS FOR SSI.JSON AND SPLIT.JSON
// =========================================================================
output.AppendLine("Building 10 Brep pairs for SSI & Split...");

var pairs = new List<(string name, Brep brepA, Brep brepB, double tol)>();
double defaultTol = 0.001;

// Pair 1: box / box
var boxA1 = new BoundingBox(-5, -5, -5, 5, 5, 5).ToBrep();
var boxB1 = new BoundingBox(0, -3, -3, 10, 3, 3).ToBrep();
pairs.Add(("box_box", boxA1, boxB1, defaultTol));

// Pair 2: box / sphere
var boxA2 = new BoundingBox(-5, -5, -5, 5, 5, 5).ToBrep();
var sphB2 = new Sphere(new Point3d(4, 0, 0), 4.0).ToBrep();
pairs.Add(("box_sphere", boxA2, sphB2, defaultTol));

// Pair 3: box / cylinder capped
var boxA3 = new BoundingBox(-5, -5, -5, 5, 5, 5).ToBrep();
var cyl3 = new Cylinder(new Circle(new Plane(new Point3d(0, 0, -8), Vector3d.ZAxis), 3.5), 16.0);
var cylB3 = cyl3.ToBrep(true, true);
pairs.Add(("box_cylinder_capped", boxA3, cylB3, defaultTol));

// Pair 4: Steinmetz cylinders (equal radius R=10 perpendicular capped cylinders)
var cylA4 = new Cylinder(new Circle(new Plane(new Point3d(-15, 0, 0), Vector3d.XAxis), 10.0), 30.0).ToBrep(true, true);
var cylB4 = new Cylinder(new Circle(new Plane(new Point3d(0, -15, 0), Vector3d.YAxis), 10.0), 30.0).ToBrep(true, true);
pairs.Add(("steinmetz_cylinders", cylA4, cylB4, defaultTol));

// Pair 5: sphere / sphere
var sphA5 = new Sphere(new Point3d(-3, 0, 0), 5.0).ToBrep();
var sphB5 = new Sphere(new Point3d(3, 0, 0), 5.0).ToBrep();
pairs.Add(("sphere_sphere", sphA5, sphB5, defaultTol));

// Pair 6: two freeform surfaces
var ffPtsA = new Point3d[36];
var ffPtsB = new Point3d[36];
for (int i = 0; i < 6; i++)
{
    for (int j = 0; j < 6; j++)
    {
        double u = i * 4.0 - 10.0;
        double v = j * 4.0 - 10.0;
        ffPtsA[i * 6 + j] = new Point3d(u, v, Math.Sin(u * 0.3) * Math.Cos(v * 0.3) * 3.0);
        ffPtsB[i * 6 + j] = new Point3d(u, Math.Cos(u * 0.3) * 3.0, v);
    }
}
var ffSrfA = NurbsSurface.CreateFromPoints(ffPtsA, 6, 6, 3, 3);
var ffSrfB = NurbsSurface.CreateFromPoints(ffPtsB, 6, 6, 3, 3);
var ffBrepA = Brep.CreateFromSurface(ffSrfA);
var ffBrepB = Brep.CreateFromSurface(ffSrfB);
pairs.Add(("two_freeform_surfaces", ffBrepA, ffBrepB, defaultTol));

// Pair 7: torus / plane
var tor7 = new Torus(Plane.WorldXY, 10.0, 3.0).ToRevSurface().ToBrep();
var pl7 = new PlaneSurface(new Plane(new Point3d(0, 0, 1.0), new Vector3d(0, 0.5, 1)), new Interval(-15, 15), new Interval(-15, 15)).ToBrep();
pairs.Add(("torus_plane", tor7, pl7, defaultTol));

// Pair 8: near-tangent spheres (distance 9.998, radii 5.0 + 5.0 = 10.0, tiny intersection circle)
var sphA8 = new Sphere(new Point3d(0, 0, 0), 5.0).ToBrep();
var sphB8 = new Sphere(new Point3d(9.998, 0, 0), 5.0).ToBrep();
pairs.Add(("near_tangent_spheres", sphA8, sphB8, defaultTol));

// Pair 9: coincident-face boxes
var boxA9 = new BoundingBox(0, 0, 0, 10, 10, 10).ToBrep();
var boxB9 = new BoundingBox(10, 0, 0, 20, 10, 10).ToBrep();
pairs.Add(("coincident_face_boxes", boxA9, boxB9, defaultTol));

// Pair 10: cylinder / sphere
var cylA10 = new Cylinder(new Circle(Plane.WorldXY, 4.0), 10.0).ToBrep(true, true);
var sphB10 = new Sphere(new Point3d(3, 0, 5), 3.5).ToBrep();
pairs.Add(("cylinder_sphere", cylA10, sphB10, defaultTol));

// --- 3.1 Write SSI.JSON ---
var ssiCases = new List<SsiCaseDto>();
foreach (var p in pairs)
{
    Intersection.BrepBrep(p.brepA, p.brepB, p.tol, out Curve[] intersectionCurves, out Point3d[] intersectionPoints);
    var ssiCurves = new List<SsiCurveDto>();
    if (intersectionCurves != null)
    {
        foreach (var c in intersectionCurves)
        {
            var pts = new List<double[]>();
            double t0 = c.Domain.T0;
            double t1 = c.Domain.T1;
            int numSamples = 200;
            for (int k = 0; k < numSamples; k++)
            {
                double s = (double)k / (numSamples - 1);
                double t = t0 + s * (t1 - t0);
                Point3d pt = c.PointAt(t);
                pts.Add(new double[] { pt.X, pt.Y, pt.Z });
            }
            ssiCurves.Add(new SsiCurveDto
            {
                Degree = c.Degree,
                IsClosed = c.IsClosed,
                PointCount = pts.Count,
                Points = pts
            });
        }
    }

    ssiCases.Add(new SsiCaseDto
    {
        Name = p.name,
        Tolerance = p.tol,
        BrepA = ExportKBrep(p.brepA),
        BrepB = ExportKBrep(p.brepB),
        CurveCount = ssiCurves.Count,
        Curves = ssiCurves
    });
}
string ssiJsonPath = Path.Combine(goldenDir, "ssi.json");
File.WriteAllText(ssiJsonPath, JsonSerializer.Serialize(ssiCases, jsonOptions));
output.AppendLine($"Exported {ssiCases.Count} SSI cases to {ssiJsonPath}");

// --- 3.2 Write SPLIT.JSON ---
var splitCases = new List<SplitCaseDto>();
foreach (var p in pairs)
{
    var piecesA = p.brepA.Split(p.brepB, p.tol);
    var piecesB = p.brepB.Split(p.brepA, p.tol);

    splitCases.Add(new SplitCaseDto
    {
        Name = p.name,
        Tolerance = p.tol,
        SplitAByB = BuildSplitResult(piecesA),
        SplitBByA = BuildSplitResult(piecesB)
    });
}
string splitJsonPath = Path.Combine(goldenDir, "split.json");
File.WriteAllText(splitJsonPath, JsonSerializer.Serialize(splitCases, jsonOptions));
output.AppendLine($"Exported {splitCases.Count} Split cases to {splitJsonPath}");

// =========================================================================
// 4. JOIN.JSON
// =========================================================================
output.AppendLine("Building JoinBreps cases...");
var joinCases = new List<JoinCaseDto>();
double joinTol = 0.001;

// Base open box: 5 faces of a 10x10x10 box (bottom + 4 sides)
// Face 0: bottom (z=0, [0..10]x[0..10])
// Face 1: front (y=0, [0..10]x[0..10])
// Face 2: back (y=10, [0..10]x[0..10])
// Face 3: left (x=0, [0..10]x[0..10])
// Face 4: right (x=10, [0..10]x[0..10])
var fullBox = new BoundingBox(0, 0, 0, 10, 10, 10).ToBrep();
// Identify top face (where normal points +Z or Z is 10)
int topFaceIdx = -1;
for (int fi = 0; fi < fullBox.Faces.Count; fi++)
{
    var bbox = fullBox.Faces[fi].GetBoundingBox(true);
    if (Math.Abs(bbox.Min.Z - 10.0) < 1e-4 && Math.Abs(bbox.Max.Z - 10.0) < 1e-4)
    {
        topFaceIdx = fi;
        break;
    }
}

var openBoxFaces = new List<Brep>();
for (int fi = 0; fi < fullBox.Faces.Count; fi++)
{
    if (fi != topFaceIdx)
    {
        openBoxFaces.Add(fullBox.Faces[fi].DuplicateFace(false));
    }
}
var openBox = Brep.JoinBreps(openBoxFaces, 0.0001)[0];
var baseTopFace = fullBox.Faces[topFaceIdx].DuplicateFace(false);

// Tolerance step cases: offset top face along Z by factor * joinTol
double[] offsetFactors = new double[] { 0.5, 0.9, 1.1, 2.0 };
foreach (var factor in offsetFactors)
{
    double offsetZ = factor * joinTol;
    var shiftedTop = baseTopFace.DuplicateBrep();
    shiftedTop.Transform(Transform.Translation(0, 0, offsetZ));

    var inputs = new Brep[] { openBox.DuplicateBrep(), shiftedTop };
    var joined = Brep.JoinBreps(inputs, joinTol);

    string caseName = $"box_offset_{factor.ToString("0.0", CultureInfo.InvariantCulture).Replace('.', '_')}";
    joinCases.Add(BuildJoinCase(caseName, joinTol, inputs, joined));
}

// Case 5: Shifted Shells (meeting along edge, shifted by 0.5*tol vs 1.5*tol)
var shell1 = new PlaneSurface(Plane.WorldXY, new Interval(-10, 0), new Interval(0, 10)).ToBrep();
var shell2SubTol = new PlaneSurface(Plane.WorldXY, new Interval(0, 10), new Interval(0.5 * joinTol, 10 + 0.5 * joinTol)).ToBrep();
var joinedShellsSub = Brep.JoinBreps(new Brep[] { shell1.DuplicateBrep(), shell2SubTol }, joinTol);
joinCases.Add(BuildJoinCase("shifted_shells_0_5", joinTol, new Brep[] { shell1.DuplicateBrep(), shell2SubTol }, joinedShellsSub));

var shell2SuperTol = new PlaneSurface(Plane.WorldXY, new Interval(0, 10), new Interval(1.5 * joinTol, 10 + 1.5 * joinTol)).ToBrep();
var joinedShellsSuper = Brep.JoinBreps(new Brep[] { shell1.DuplicateBrep(), shell2SuperTol }, joinTol);
joinCases.Add(BuildJoinCase("shifted_shells_1_5", joinTol, new Brep[] { shell1.DuplicateBrep(), shell2SuperTol }, joinedShellsSuper));

// Case 6: T-Junction cases
// Face 1: Large base face [0, 20] x [0, 10]
var tBase = new PlaneSurface(Plane.WorldXY, new Interval(0, 20), new Interval(0, 10)).ToBrep();
// Face 2: Left square [0, 10] x [10, 20]
var tLeft = new PlaneSurface(Plane.WorldXY, new Interval(0, 10), new Interval(10, 20)).ToBrep();
// Face 3: Right square [10, 20] x [10, 20]
var tRight = new PlaneSurface(Plane.WorldXY, new Interval(10, 20), new Interval(10, 20)).ToBrep();

var tInputs = new Brep[] { tBase, tLeft, tRight };
var tJoined = Brep.JoinBreps(tInputs, joinTol);
joinCases.Add(BuildJoinCase("t_junction", joinTol, tInputs, tJoined));

string joinJsonPath = Path.Combine(goldenDir, "join.json");
File.WriteAllText(joinJsonPath, JsonSerializer.Serialize(joinCases, jsonOptions));
output.AppendLine($"Exported {joinCases.Count} Join cases to {joinJsonPath}");

output.AppendLine("Export completed successfully.");

// =========================================================================
// HELPER METHODS
// =========================================================================

SurfaceCaseDto BuildSurfaceCase(string name, NurbsSurface srf)
{
    var dto = new SurfaceCaseDto
    {
        Name = name,
        DegreeU = srf.Degree(0),
        DegreeV = srf.Degree(1),
        CvCountU = srf.Points.CountU,
        CvCountV = srf.Points.CountV,
        IsRational = srf.IsRational,
        IsClosedU = srf.IsClosed(0),
        IsClosedV = srf.IsClosed(1),
        KnotsU = GetFullKnots(srf.KnotsU),
        KnotsV = GetFullKnots(srf.KnotsV),
        Cvs = GetSurfaceCvs(srf),
        Samples = new List<SurfaceSampleDto>()
    };

    Interval uDom = srf.Domain(0);
    Interval vDom = srf.Domain(1);
    int numU = 8, numV = 7;
    for (int i = 0; i < numU; i++)
    {
        double u = uDom.ParameterAt((i + 0.5) / numU);
        for (int j = 0; j < numV; j++)
        {
            double v = vDom.ParameterAt((j + 0.5) / numV);
            srf.Evaluate(u, v, 2, out Point3d pt, out Vector3d[] ders);
            Vector3d norm = srf.NormalAt(u, v);
            norm.Unitize();

            dto.Samples.Add(new SurfaceSampleDto
            {
                U = u,
                V = v,
                Point = new double[] { pt.X, pt.Y, pt.Z },
                Su = ders != null && ders.Length > 0 ? new double[] { ders[0].X, ders[0].Y, ders[0].Z } : new double[3],
                Sv = ders != null && ders.Length > 1 ? new double[] { ders[1].X, ders[1].Y, ders[1].Z } : new double[3],
                Suu = ders != null && ders.Length > 2 ? new double[] { ders[2].X, ders[2].Y, ders[2].Z } : new double[3],
                Suv = ders != null && ders.Length > 3 ? new double[] { ders[3].X, ders[3].Y, ders[3].Z } : new double[3],
                Svv = ders != null && ders.Length > 4 ? new double[] { ders[4].X, ders[4].Y, ders[4].Z } : new double[3],
                Normal = new double[] { norm.X, norm.Y, norm.Z }
            });
        }
    }

    return dto;
}

CurveCaseDto BuildCurveCase(string name, NurbsCurve crv, int dimension)
{
    var dto = new CurveCaseDto
    {
        Name = name,
        Dimension = dimension,
        Degree = crv.Degree,
        CvCount = crv.Points.Count,
        IsRational = crv.IsRational,
        Knots = GetFullKnots(crv.Knots),
        Cvs = GetCurveCvs(crv, dimension),
        Samples = new List<CurveSampleDto>()
    };

    Interval dom = crv.Domain;
    int numSamples = 50;
    for (int k = 0; k < numSamples; k++)
    {
        double t = dom.ParameterAt((double)k / (numSamples - 1));
        Point3d pt = crv.PointAt(t);
        Vector3d[] ders = crv.DerivativeAt(t, 2);

        dto.Samples.Add(new CurveSampleDto
        {
            T = t,
            Point = new double[] { pt.X, pt.Y, dimension == 2 ? 0.0 : pt.Z },
            D1 = ders != null && ders.Length > 1 ? new double[] { ders[1].X, ders[1].Y, dimension == 2 ? 0.0 : ders[1].Z } : new double[3],
            D2 = ders != null && ders.Length > 2 ? new double[] { ders[2].X, ders[2].Y, dimension == 2 ? 0.0 : ders[2].Z } : new double[3]
        });
    }

    return dto;
}

double[] GetFullKnots(NurbsSurfaceKnotList knots)
{
    double[] full = new double[knots.Count + 2];
    full[0] = knots.SuperfluousKnot(true);
    for (int i = 0; i < knots.Count; i++) full[i + 1] = knots[i];
    full[full.Length - 1] = knots.SuperfluousKnot(false);
    return full;
}

double[] GetFullKnots(NurbsCurveKnotList knots)
{
    double[] full = new double[knots.Count + 2];
    full[0] = knots.SuperfluousKnot(true);
    for (int i = 0; i < knots.Count; i++) full[i + 1] = knots[i];
    full[full.Length - 1] = knots.SuperfluousKnot(false);
    return full;
}

double[] GetSurfaceCvs(NurbsSurface srf)
{
    int countU = srf.Points.CountU;
    int countV = srf.Points.CountV;
    bool isRat = srf.IsRational;
    double[] cvs = new double[countU * countV * 4];
    int ptr = 0;
    for (int i = 0; i < countU; i++)
    {
        for (int j = 0; j < countV; j++)
        {
            ControlPoint cp = srf.Points.GetControlPoint(i, j);
            cvs[ptr++] = cp.X;
            cvs[ptr++] = cp.Y;
            cvs[ptr++] = cp.Z;
            cvs[ptr++] = cp.Weight;
        }
    }
    return cvs;
}

double[] GetCurveCvs(NurbsCurve crv, int dimension)
{
    int count = crv.Points.Count;
    double[] cvs = new double[count * 4];
    int ptr = 0;
    for (int i = 0; i < count; i++)
    {
        ControlPoint cp = crv.Points[i];
        cvs[ptr++] = cp.X;
        cvs[ptr++] = cp.Y;
        cvs[ptr++] = dimension == 2 ? 0.0 : cp.Z;
        cvs[ptr++] = cp.Weight;
    }
    return cvs;
}

KBrepDto ExportKBrep(Brep brep)
{
    var dto = new KBrepDto();

    // Surfaces
    for (int i = 0; i < brep.Surfaces.Count; i++)
    {
        var s = brep.Surfaces[i].ToNurbsSurface();
        dto.Surfaces.Add(new KNurbsSurfaceDto
        {
            DegreeU = s.Degree(0),
            DegreeV = s.Degree(1),
            CvCountU = s.Points.CountU,
            CvCountV = s.Points.CountV,
            IsRational = s.IsRational,
            IsClosedU = s.IsClosed(0),
            IsClosedV = s.IsClosed(1),
            KnotsU = GetFullKnots(s.KnotsU),
            KnotsV = GetFullKnots(s.KnotsV),
            Cvs = GetSurfaceCvs(s)
        });
    }

    // Curves3D
    for (int i = 0; i < brep.Curves3D.Count; i++)
    {
        var c = brep.Curves3D[i].ToNurbsCurve();
        dto.Curves3.Add(new KNurbsCurveDto
        {
            Dimension = 3,
            Degree = c.Degree,
            CvCount = c.Points.Count,
            IsRational = c.IsRational,
            Knots = GetFullKnots(c.Knots),
            Cvs = GetCurveCvs(c, 3)
        });
    }

    // Curves2D
    for (int i = 0; i < brep.Curves2D.Count; i++)
    {
        var c = brep.Curves2D[i].ToNurbsCurve();
        dto.Curves2.Add(new KNurbsCurveDto
        {
            Dimension = 2,
            Degree = c.Degree,
            CvCount = c.Points.Count,
            IsRational = c.IsRational,
            Knots = GetFullKnots(c.Knots),
            Cvs = GetCurveCvs(c, 2)
        });
    }

    // Vertices
    for (int i = 0; i < brep.Vertices.Count; i++)
    {
        var v = brep.Vertices[i];
        dto.Vertices.Add(new KVertexDto
        {
            Point = new double[] { v.Location.X, v.Location.Y, v.Location.Z },
            Tolerance = v.Tolerance,
            Edges = v.EdgeIndices() != null ? new List<int>(v.EdgeIndices()) : new List<int>()
        });
    }

    // Edges
    for (int i = 0; i < brep.Edges.Count; i++)
    {
        var e = brep.Edges[i];
        dto.Edges.Add(new KEdgeDto
        {
            Curve3 = e.EdgeCurveIndex,
            Domain = new double[] { e.Domain.T0, e.Domain.T1 },
            V0 = e.StartVertex != null ? e.StartVertex.VertexIndex : -1,
            V1 = e.EndVertex != null ? e.EndVertex.VertexIndex : -1,
            Trims = e.TrimIndices() != null ? new List<int>(e.TrimIndices()) : new List<int>(),
            Tolerance = e.Tolerance
        });
    }

    // Trims
    for (int i = 0; i < brep.Trims.Count; i++)
    {
        var t = brep.Trims[i];
        t.GetTolerances(out double tolU, out double tolV);
        dto.Trims.Add(new KTrimDto
        {
            Curve2 = t.TrimCurveIndex,
            Domain = new double[] { t.Domain.T0, t.Domain.T1 },
            Edge = t.Edge != null ? t.Edge.EdgeIndex : -1,
            Loop = t.Loop != null ? t.Loop.LoopIndex : -1,
            Reversed3d = t.IsReversed(),
            Type = t.TrimType.ToString(),
            TolU = tolU,
            TolV = tolV
        });
    }

    // Loops
    for (int i = 0; i < brep.Loops.Count; i++)
    {
        var l = brep.Loops[i];
        var trimList = new List<int>();
        for (int ti = 0; ti < l.Trims.Count; ti++)
        {
            trimList.Add(l.Trims[ti].TrimIndex);
        }
        dto.Loops.Add(new KLoopDto
        {
            Face = l.Face != null ? l.Face.FaceIndex : -1,
            Type = l.LoopType.ToString(),
            Trims = trimList
        });
    }

    // Faces (outer loop first)
    for (int i = 0; i < brep.Faces.Count; i++)
    {
        var f = brep.Faces[i];
        var loopList = new List<int>();
        if (f.OuterLoop != null)
        {
            loopList.Add(f.OuterLoop.LoopIndex);
        }
        for (int li = 0; li < f.Loops.Count; li++)
        {
            int lIdx = f.Loops[li].LoopIndex;
            if (!loopList.Contains(lIdx))
                loopList.Add(lIdx);
        }
        dto.Faces.Add(new KFaceDto
        {
            Surface = f.SurfaceIndex,
            Reversed = f.OrientationIsReversed,
            Loops = loopList
        });
    }

    return dto;
}

SplitResultDto BuildSplitResult(Brep[] pieces)
{
    var res = new SplitResultDto
    {
        PieceCount = pieces != null ? pieces.Length : 0,
        Pieces = new List<PieceDto>()
    };

    if (pieces != null)
    {
        foreach (var p in pieces)
        {
            var amp = AreaMassProperties.Compute(p);
            res.Pieces.Add(new PieceDto
            {
                FaceCount = p.Faces.Count,
                Area = amp != null ? amp.Area : 0.0,
                IsValid = p.IsValid
            });
        }
    }
    return res;
}

JoinCaseDto BuildJoinCase(string name, double tol, Brep[] inputs, Brep[] joined)
{
    var dto = new JoinCaseDto
    {
        Name = name,
        Tolerance = tol,
        InputCount = inputs != null ? inputs.Length : 0,
        Inputs = inputs != null ? inputs.Select(ExportKBrep).ToList() : new List<KBrepDto>(),
        ResultCount = joined != null ? joined.Length : 0,
        Results = new List<JoinResultItemDto>()
    };

    if (joined != null)
    {
        foreach (var j in joined)
        {
            int nakedCount = 0;
            foreach (var e in j.Edges)
            {
                if (e.Valence == EdgeAdjacency.Naked)
                    nakedCount++;
            }

            dto.Results.Add(new JoinResultItemDto
            {
                IsSolid = j.IsSolid,
                NakedEdgeCount = nakedCount
            });
        }
    }

    return dto;
}

// =========================================================================
// DTO DEFINITIONS
// =========================================================================

public sealed class SurfaceCaseDto
{
    public string Name { get; set; }
    public int DegreeU { get; set; }
    public int DegreeV { get; set; }
    public int CvCountU { get; set; }
    public int CvCountV { get; set; }
    public bool IsRational { get; set; }
    public bool IsClosedU { get; set; }
    public bool IsClosedV { get; set; }
    public double[] KnotsU { get; set; }
    public double[] KnotsV { get; set; }
    public double[] Cvs { get; set; }
    public List<SurfaceSampleDto> Samples { get; set; }
}

public sealed class SurfaceSampleDto
{
    public double U { get; set; }
    public double V { get; set; }
    public double[] Point { get; set; }
    public double[] Su { get; set; }
    public double[] Sv { get; set; }
    public double[] Suu { get; set; }
    public double[] Suv { get; set; }
    public double[] Svv { get; set; }
    public double[] Normal { get; set; }
}

public sealed class CurveCaseDto
{
    public string Name { get; set; }
    public int Dimension { get; set; }
    public int Degree { get; set; }
    public int CvCount { get; set; }
    public bool IsRational { get; set; }
    public double[] Knots { get; set; }
    public double[] Cvs { get; set; }
    public List<CurveSampleDto> Samples { get; set; }
}

public sealed class CurveSampleDto
{
    public double T { get; set; }
    public double[] Point { get; set; }
    public double[] D1 { get; set; }
    public double[] D2 { get; set; }
}

public sealed class SsiCaseDto
{
    public string Name { get; set; }
    public double Tolerance { get; set; }
    public KBrepDto BrepA { get; set; }
    public KBrepDto BrepB { get; set; }
    public int CurveCount { get; set; }
    public List<SsiCurveDto> Curves { get; set; }
}

public sealed class SsiCurveDto
{
    public int Degree { get; set; }
    public bool IsClosed { get; set; }
    public int PointCount { get; set; }
    public List<double[]> Points { get; set; }
}

public sealed class SplitCaseDto
{
    public string Name { get; set; }
    public double Tolerance { get; set; }
    public SplitResultDto SplitAByB { get; set; }
    public SplitResultDto SplitBByA { get; set; }
}

public sealed class SplitResultDto
{
    public int PieceCount { get; set; }
    public List<PieceDto> Pieces { get; set; }
}

public sealed class PieceDto
{
    public int FaceCount { get; set; }
    public double Area { get; set; }
    public bool IsValid { get; set; }
}

public sealed class JoinCaseDto
{
    public string Name { get; set; }
    public double Tolerance { get; set; }
    public int InputCount { get; set; }
    public List<KBrepDto> Inputs { get; set; }
    public int ResultCount { get; set; }
    public List<JoinResultItemDto> Results { get; set; }
}

public sealed class JoinResultItemDto
{
    public bool IsSolid { get; set; }
    public int NakedEdgeCount { get; set; }
}

public sealed class KBrepDto
{
    public List<KNurbsSurfaceDto> Surfaces { get; set; } = new List<KNurbsSurfaceDto>();
    public List<KNurbsCurveDto> Curves3 { get; set; } = new List<KNurbsCurveDto>();
    public List<KNurbsCurveDto> Curves2 { get; set; } = new List<KNurbsCurveDto>();
    public List<KVertexDto> Vertices { get; set; } = new List<KVertexDto>();
    public List<KEdgeDto> Edges { get; set; } = new List<KEdgeDto>();
    public List<KTrimDto> Trims { get; set; } = new List<KTrimDto>();
    public List<KLoopDto> Loops { get; set; } = new List<KLoopDto>();
    public List<KFaceDto> Faces { get; set; } = new List<KFaceDto>();
}

public sealed class KNurbsSurfaceDto
{
    public int DegreeU { get; set; }
    public int DegreeV { get; set; }
    public int CvCountU { get; set; }
    public int CvCountV { get; set; }
    public bool IsRational { get; set; }
    public bool IsClosedU { get; set; }
    public bool IsClosedV { get; set; }
    public double[] KnotsU { get; set; }
    public double[] KnotsV { get; set; }
    public double[] Cvs { get; set; }
}

public sealed class KNurbsCurveDto
{
    public int Dimension { get; set; }
    public int Degree { get; set; }
    public int CvCount { get; set; }
    public bool IsRational { get; set; }
    public double[] Knots { get; set; }
    public double[] Cvs { get; set; }
}

public sealed class KVertexDto
{
    public double[] Point { get; set; }
    public double Tolerance { get; set; }
    public List<int> Edges { get; set; }
}

public sealed class KEdgeDto
{
    public int Curve3 { get; set; }
    public double[] Domain { get; set; }
    public int V0 { get; set; }
    public int V1 { get; set; }
    public List<int> Trims { get; set; }
    public double Tolerance { get; set; }
}

public sealed class KTrimDto
{
    public int Curve2 { get; set; }
    public double[] Domain { get; set; }
    public int Edge { get; set; }
    public int Loop { get; set; }
    public bool Reversed3d { get; set; }
    public string Type { get; set; }
    public double TolU { get; set; }
    public double TolV { get; set; }
}

public sealed class KLoopDto
{
    public int Face { get; set; }
    public string Type { get; set; }
    public List<int> Trims { get; set; }
}

public sealed class KFaceDto
{
    public int Surface { get; set; }
    public bool Reversed { get; set; }
    public List<int> Loops { get; set; }
}
