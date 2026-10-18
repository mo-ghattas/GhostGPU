using System;
using System.Collections.Generic;
using GhostGPU.Core;

namespace GhostGPU.Kernel
{
    // Shared model; public conventions are documented in docs/contracts/kernel.md.
    // Conventions: double precision; knot vectors are FULL (length = cvCount + degree + 1, clamped),
    // i.e. opennurbs' "superfluous" end knots are restored by the Rhino adapter; control points are
    // homogeneous (x*w, y*w, z*w, w) like opennurbs; w = 1 when not rational.

    public readonly struct Interval
    {
        public readonly double T0, T1;
        public Interval(double t0, double t1) { T0 = t0; T1 = t1; }
        public double Length => T1 - T0;
        public double ParameterAt(double s) => T0 + s * (T1 - T0);
    }

    public readonly struct V2
    {
        public readonly double X, Y;
        public V2(double x, double y) { X = x; Y = y; }
    }

    /// <summary>NURBS curve in 2D (trim, Dimension = 2: z of each CV is 0) or 3D (edge, Dimension = 3).</summary>
    public sealed class NurbsCurve
    {
        public int Dimension { get; }
        public int Degree { get; }
        public int CvCount { get; }
        public bool IsRational { get; }
        /// <summary>Full clamped knot vector, length CvCount + Degree + 1.</summary>
        public double[] Knots { get; }
        /// <summary>Homogeneous CVs, 4 doubles per CV: x*w, y*w, z*w, w.</summary>
        public double[] Cvs { get; }

        public NurbsCurve(int dimension, int degree, bool isRational, double[] knots, double[] cvs)
        {
            if (dimension != 2 && dimension != 3) throw new ArgumentOutOfRangeException(nameof(dimension));
            if (degree < 1) throw new ArgumentOutOfRangeException(nameof(degree));
            if (cvs == null || cvs.Length % 4 != 0) throw new ArgumentException("Cvs must hold 4 doubles per CV.", nameof(cvs));
            Dimension = dimension; Degree = degree; IsRational = isRational; CvCount = cvs.Length / 4;
            if (knots == null || knots.Length != CvCount + degree + 1)
                throw new ArgumentException("Knots must be the full clamped vector (CvCount + Degree + 1).", nameof(knots));
            Knots = knots; Cvs = cvs;
        }

        public Interval Domain => new Interval(Knots[Degree], Knots[CvCount]);
    }

    /// <summary>NURBS surface. CVs are row-major over U: index (i, j) = i * CvCountV + j.</summary>
    public sealed class NurbsSurface
    {
        public int DegreeU { get; }
        public int DegreeV { get; }
        public int CvCountU { get; }
        public int CvCountV { get; }
        public bool IsRational { get; }
        public double[] KnotsU { get; }
        public double[] KnotsV { get; }
        /// <summary>Homogeneous CVs, 4 doubles per CV, index (i * CvCountV + j) * 4.</summary>
        public double[] Cvs { get; }
        /// <summary>Closed-periodic flags (seam handling in split); false for clamped open surfaces.</summary>
        public bool IsClosedU { get; set; }
        public bool IsClosedV { get; set; }

        public NurbsSurface(int degreeU, int degreeV, int cvCountU, int cvCountV, bool isRational,
                            double[] knotsU, double[] knotsV, double[] cvs)
        {
            if (cvs == null || cvs.Length != cvCountU * cvCountV * 4) throw new ArgumentException("Cvs size mismatch.", nameof(cvs));
            if (knotsU == null || knotsU.Length != cvCountU + degreeU + 1) throw new ArgumentException("KnotsU must be full.", nameof(knotsU));
            if (knotsV == null || knotsV.Length != cvCountV + degreeV + 1) throw new ArgumentException("KnotsV must be full.", nameof(knotsV));
            DegreeU = degreeU; DegreeV = degreeV; CvCountU = cvCountU; CvCountV = cvCountV;
            IsRational = isRational; KnotsU = knotsU; KnotsV = knotsV; Cvs = cvs;
        }

        public Interval DomainU => new Interval(KnotsU[DegreeU], KnotsU[CvCountU]);
        public Interval DomainV => new Interval(KnotsV[DegreeV], KnotsV[CvCountV]);
    }

    // ---- Topology: mirrors ON_Brep (public opennurbs). Indices into the KBrep lists; -1 = none. ----

    public enum TrimType { Unknown, Boundary, Mated, Seam, Singular, CurveOnSurface, PointOnSurface, Slit }
    public enum LoopType { Unknown, Outer, Inner, Slit, CurveOnSurface, PointOnSurface }

    public sealed class KVertex { public V3 Point; public double Tolerance; public List<int> Edges = new List<int>(); }

    public sealed class KEdge
    {
        public int Curve3 = -1; public Interval Domain; public int V0 = -1, V1 = -1;
        public List<int> Trims = new List<int>(); public double Tolerance;
    }

    public sealed class KTrim
    {
        public int Curve2 = -1; public Interval Domain; public int Edge = -1; public int Loop = -1;
        /// <summary>True when the trim runs opposite to its edge's 3D curve direction.</summary>
        public bool Reversed3d; public TrimType Type; public double TolU, TolV;
    }

    public sealed class KLoop { public int Face = -1; public LoopType Type; public List<int> Trims = new List<int>(); }

    public sealed class KFace
    {
        public int Surface = -1;
        /// <summary>True when the face normal is opposite to the surface normal (ON_BrepFace::m_bRev).</summary>
        public bool Reversed; public List<int> Loops = new List<int>();
    }

    /// <summary>Boundary representation. Outer loop first in each face's Loops; outer CCW, inner CW in UV.</summary>
    public sealed class KBrep
    {
        public List<NurbsSurface> Surfaces = new List<NurbsSurface>();
        public List<NurbsCurve> Curves3 = new List<NurbsCurve>();
        public List<NurbsCurve> Curves2 = new List<NurbsCurve>();
        public List<KVertex> Vertices = new List<KVertex>();
        public List<KEdge> Edges = new List<KEdge>();
        public List<KTrim> Trims = new List<KTrim>();
        public List<KLoop> Loops = new List<KLoop>();
        public List<KFace> Faces = new List<KFace>();
    }

    // ---- Milestone APIs (implementations owned per docs/contracts/kernel.md) ----

    /// <summary>One SSI branch: 3D curve plus its pullbacks on the two faces it lies on.</summary>
    public sealed class SsiBranch
    {
        public int FaceA, FaceB;
        public NurbsCurve Curve3;
        public NurbsCurve UvA, UvB;
        public bool IsClosed;
        /// <summary>Max measured deviation |S_A(uvA(t)) - S_B(uvB(t))| and to Curve3 at sample points.</summary>
        public double MaxDeviation;
    }

    public sealed class SsiResult
    {
        public List<SsiBranch> Branches = new List<SsiBranch>();
        public List<string> Warnings = new List<string>();
    }
}
