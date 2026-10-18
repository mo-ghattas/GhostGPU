using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using GhostGPU.Core;
using GhostGPU.Kernel;

namespace GhostGPU.Kernel.Validation
{
    public sealed class GoldenSurfaceCase
    {
        public string Name { get; set; }
        public NurbsSurface Surface { get; set; }
        public List<GoldenSurfaceSample> Samples { get; set; } = new List<GoldenSurfaceSample>();
    }

    public sealed class GoldenSurfaceSample
    {
        public double U { get; set; }
        public double V { get; set; }
        public V3 Point { get; set; }
        public V3 Su { get; set; }
        public V3 Sv { get; set; }
        public V3 Suu { get; set; }
        public V3 Suv { get; set; }
        public V3 Svv { get; set; }
        public V3 Normal { get; set; }
    }

    public sealed class GoldenCurveCase
    {
        public string Name { get; set; }
        public int Dimension { get; set; }
        public NurbsCurve Curve { get; set; }
        public List<GoldenCurveSample> Samples { get; set; } = new List<GoldenCurveSample>();
    }

    public sealed class GoldenCurveSample
    {
        public double T { get; set; }
        public V3 Point { get; set; }
        public V3 D1 { get; set; }
        public V3 D2 { get; set; }
    }

    public sealed class GoldenSsiCase
    {
        public string Name { get; set; }
        public double Tolerance { get; set; }
        public KBrep BrepA { get; set; }
        public KBrep BrepB { get; set; }
        public int CurveCount { get; set; }
        public List<GoldenSsiCurve> Curves { get; set; } = new List<GoldenSsiCurve>();
    }

    public sealed class GoldenSsiCurve
    {
        public int Degree { get; set; }
        public bool IsClosed { get; set; }
        public int PointCount { get; set; }
        public List<V3> Points { get; set; } = new List<V3>();
    }

    public sealed class GoldenSplitCase
    {
        public string Name { get; set; }
        public double Tolerance { get; set; }
        public GoldenSplitResult SplitAByB { get; set; }
        public GoldenSplitResult SplitBByA { get; set; }
    }

    public sealed class GoldenSplitResult
    {
        public int PieceCount { get; set; }
        public List<GoldenPiece> Pieces { get; set; } = new List<GoldenPiece>();
    }

    public sealed class GoldenPiece
    {
        public int FaceCount { get; set; }
        public double Area { get; set; }
        public bool IsValid { get; set; }
    }

    public sealed class GoldenJoinCase
    {
        public string Name { get; set; }
        public double Tolerance { get; set; }
        public int InputCount { get; set; }
        public List<KBrep> Inputs { get; set; } = new List<KBrep>();
        public int ResultCount { get; set; }
        public List<GoldenJoinResult> Results { get; set; } = new List<GoldenJoinResult>();
    }

    public sealed class GoldenJoinResult
    {
        public bool IsSolid { get; set; }
        public int NakedEdgeCount { get; set; }
    }

    public static class JsonElementExtensions
    {
        public static bool TryGetProp(this JsonElement elem, string name, out JsonElement val)
        {
            if (elem.TryGetProperty(name, out val)) return true;
            if (name.Length > 0)
            {
                string pascal = char.ToUpperInvariant(name[0]) + name.Substring(1);
                if (elem.TryGetProperty(pascal, out val)) return true;
                string lower = char.ToLowerInvariant(name[0]) + name.Substring(1);
                if (elem.TryGetProperty(lower, out val)) return true;
            }
            val = default;
            return false;
        }

        public static JsonElement GetProp(this JsonElement elem, string name)
        {
            if (elem.TryGetProp(name, out var val)) return val;
            throw new KeyNotFoundException($"Property '{name}' not found on JsonElement.");
        }
    }

    public static class GoldenDataLoader
    {
        public static string ResolveGoldenDir(string explicitDir = null)
        {
            if (!string.IsNullOrEmpty(explicitDir) && Directory.Exists(explicitDir))
                return explicitDir;

            string dir = AppContext.BaseDirectory;
            while (!string.IsNullOrEmpty(dir))
            {
                string candidate = Path.Combine(dir, "GhostGPU", "tests", "fixtures", "golden");
                if (Directory.Exists(candidate)) return candidate;

                candidate = Path.Combine(dir, "tests", "fixtures", "golden");
                if (Directory.Exists(candidate)) return candidate;

                dir = Path.GetDirectoryName(dir);
            }

            return Path.Combine(AppContext.BaseDirectory, "golden");
        }

        public static List<GoldenSurfaceCase> LoadSurfaces(string goldenDir)
        {
            string path = Path.Combine(goldenDir, "surfaces.json");
            if (!File.Exists(path)) return null;

            var cases = new List<GoldenSurfaceCase>();
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            foreach (var elem in doc.RootElement.EnumerateArray())
            {
                string name = elem.GetProp("name").GetString();
                int degU = elem.GetProp("degreeU").GetInt32();
                int degV = elem.GetProp("degreeV").GetInt32();
                int countU = elem.GetProp("cvCountU").GetInt32();
                int countV = elem.GetProp("cvCountV").GetInt32();
                bool isRat = elem.GetProp("isRational").GetBoolean();
                bool closedU = elem.TryGetProp("isClosedU", out var pcu) && pcu.GetBoolean();
                bool closedV = elem.TryGetProp("isClosedV", out var pcv) && pcv.GetBoolean();

                double[] knotsU = ReadDoubleArray(elem.GetProp("knotsU"));
                double[] knotsV = ReadDoubleArray(elem.GetProp("knotsV"));
                double[] cvs = ReadDoubleArray(elem.GetProp("cvs"));

                var srf = new NurbsSurface(degU, degV, countU, countV, isRat, knotsU, knotsV, cvs)
                {
                    IsClosedU = closedU,
                    IsClosedV = closedV
                };

                var caseObj = new GoldenSurfaceCase { Name = name, Surface = srf };
                if (elem.TryGetProp("samples", out var samplesElem))
                {
                    foreach (var s in samplesElem.EnumerateArray())
                    {
                        caseObj.Samples.Add(new GoldenSurfaceSample
                        {
                            U = s.GetProp("u").GetDouble(),
                            V = s.GetProp("v").GetDouble(),
                            Point = ReadV3(s.GetProp("point")),
                            Su = ReadV3(s.GetProp("su")),
                            Sv = ReadV3(s.GetProp("sv")),
                            Suu = ReadV3(s.GetProp("suu")),
                            Suv = ReadV3(s.GetProp("suv")),
                            Svv = ReadV3(s.GetProp("svv")),
                            Normal = ReadV3(s.GetProp("normal"))
                        });
                    }
                }
                cases.Add(caseObj);
            }
            return cases;
        }

        public static List<GoldenCurveCase> LoadCurves(string goldenDir)
        {
            string path = Path.Combine(goldenDir, "curves.json");
            if (!File.Exists(path)) return null;

            var cases = new List<GoldenCurveCase>();
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            foreach (var elem in doc.RootElement.EnumerateArray())
            {
                string name = elem.GetProp("name").GetString();
                int dim = elem.GetProp("dimension").GetInt32();
                int deg = elem.GetProp("degree").GetInt32();
                bool isRat = elem.GetProp("isRational").GetBoolean();
                double[] knots = ReadDoubleArray(elem.GetProp("knots"));
                double[] cvs = ReadDoubleArray(elem.GetProp("cvs"));

                var crv = new NurbsCurve(dim, deg, isRat, knots, cvs);
                var caseObj = new GoldenCurveCase { Name = name, Dimension = dim, Curve = crv };

                if (elem.TryGetProp("samples", out var samplesElem))
                {
                    foreach (var s in samplesElem.EnumerateArray())
                    {
                        caseObj.Samples.Add(new GoldenCurveSample
                        {
                            T = s.GetProp("t").GetDouble(),
                            Point = ReadV3(s.GetProp("point")),
                            D1 = ReadV3(s.GetProp("d1")),
                            D2 = ReadV3(s.GetProp("d2"))
                        });
                    }
                }
                cases.Add(caseObj);
            }
            return cases;
        }

        public static List<GoldenSsiCase> LoadSsi(string goldenDir)
        {
            string path = Path.Combine(goldenDir, "ssi.json");
            if (!File.Exists(path)) return null;

            var cases = new List<GoldenSsiCase>();
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            foreach (var elem in doc.RootElement.EnumerateArray())
            {
                string name = elem.GetProp("name").GetString();
                double tol = elem.GetProp("tolerance").GetDouble();
                int curveCount = elem.GetProp("curveCount").GetInt32();
                var brepA = ParseKBrep(elem.GetProp("brepA"));
                var brepB = ParseKBrep(elem.GetProp("brepB"));

                var caseObj = new GoldenSsiCase
                {
                    Name = name,
                    Tolerance = tol,
                    BrepA = brepA,
                    BrepB = brepB,
                    CurveCount = curveCount
                };

                if (elem.TryGetProp("curves", out var curvesElem))
                {
                    foreach (var c in curvesElem.EnumerateArray())
                    {
                        var ssiCrv = new GoldenSsiCurve
                        {
                            Degree = c.GetProp("degree").GetInt32(),
                            IsClosed = c.GetProp("isClosed").GetBoolean(),
                            PointCount = c.GetProp("pointCount").GetInt32()
                        };
                        foreach (var p in c.GetProp("points").EnumerateArray())
                        {
                            ssiCrv.Points.Add(ReadV3(p));
                        }
                        caseObj.Curves.Add(ssiCrv);
                    }
                }
                cases.Add(caseObj);
            }
            return cases;
        }

        public static List<GoldenSplitCase> LoadSplit(string goldenDir)
        {
            string path = Path.Combine(goldenDir, "split.json");
            if (!File.Exists(path)) return null;

            var cases = new List<GoldenSplitCase>();
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            foreach (var elem in doc.RootElement.EnumerateArray())
            {
                cases.Add(new GoldenSplitCase
                {
                    Name = elem.GetProp("name").GetString(),
                    Tolerance = elem.GetProp("tolerance").GetDouble(),
                    SplitAByB = ParseSplitResult(elem.GetProp("splitAByB")),
                    SplitBByA = ParseSplitResult(elem.GetProp("splitBByA"))
                });
            }
            return cases;
        }

        public static List<GoldenJoinCase> LoadJoin(string goldenDir)
        {
            string path = Path.Combine(goldenDir, "join.json");
            if (!File.Exists(path)) return null;

            var cases = new List<GoldenJoinCase>();
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            foreach (var elem in doc.RootElement.EnumerateArray())
            {
                var caseObj = new GoldenJoinCase
                {
                    Name = elem.GetProp("name").GetString(),
                    Tolerance = elem.GetProp("tolerance").GetDouble(),
                    InputCount = elem.GetProp("inputCount").GetInt32(),
                    ResultCount = elem.GetProp("resultCount").GetInt32()
                };

                if (elem.TryGetProp("inputs", out var inputsElem))
                {
                    foreach (var inB in inputsElem.EnumerateArray())
                    {
                        caseObj.Inputs.Add(ParseKBrep(inB));
                    }
                }

                if (elem.TryGetProp("results", out var resElem))
                {
                    foreach (var r in resElem.EnumerateArray())
                    {
                        caseObj.Results.Add(new GoldenJoinResult
                        {
                            IsSolid = r.GetProp("isSolid").GetBoolean(),
                            NakedEdgeCount = r.GetProp("nakedEdgeCount").GetInt32()
                        });
                    }
                }
                cases.Add(caseObj);
            }
            return cases;
        }

        public static KBrep ParseKBrep(JsonElement elem)
        {
            var k = new KBrep();

            if (elem.TryGetProp("surfaces", out var srfElem))
            {
                foreach (var s in srfElem.EnumerateArray())
                {
                    int degU = s.GetProp("degreeU").GetInt32();
                    int degV = s.GetProp("degreeV").GetInt32();
                    int cu = s.GetProp("cvCountU").GetInt32();
                    int cv = s.GetProp("cvCountV").GetInt32();
                    bool isRat = s.GetProp("isRational").GetBoolean();
                    bool clU = s.TryGetProp("isClosedU", out var pu) && pu.GetBoolean();
                    bool clV = s.TryGetProp("isClosedV", out var pv) && pv.GetBoolean();

                    var srf = new NurbsSurface(degU, degV, cu, cv, isRat,
                        ReadDoubleArray(s.GetProp("knotsU")),
                        ReadDoubleArray(s.GetProp("knotsV")),
                        ReadDoubleArray(s.GetProp("cvs")))
                    {
                        IsClosedU = clU,
                        IsClosedV = clV
                    };
                    k.Surfaces.Add(srf);
                }
            }

            if (elem.TryGetProp("curves3", out var c3Elem))
            {
                foreach (var c in c3Elem.EnumerateArray())
                {
                    int dim = c.GetProp("dimension").GetInt32();
                    int deg = c.GetProp("degree").GetInt32();
                    bool isRat = c.GetProp("isRational").GetBoolean();
                    k.Curves3.Add(new NurbsCurve(dim, deg, isRat,
                        ReadDoubleArray(c.GetProp("knots")),
                        ReadDoubleArray(c.GetProp("cvs"))));
                }
            }

            if (elem.TryGetProp("curves2", out var c2Elem))
            {
                foreach (var c in c2Elem.EnumerateArray())
                {
                    int dim = c.GetProp("dimension").GetInt32();
                    int deg = c.GetProp("degree").GetInt32();
                    bool isRat = c.GetProp("isRational").GetBoolean();
                    k.Curves2.Add(new NurbsCurve(dim, deg, isRat,
                        ReadDoubleArray(c.GetProp("knots")),
                        ReadDoubleArray(c.GetProp("cvs"))));
                }
            }

            if (elem.TryGetProp("vertices", out var vElem))
            {
                foreach (var v in vElem.EnumerateArray())
                {
                    var kv = new KVertex
                    {
                        Point = ReadV3(v.GetProp("point")),
                        Tolerance = v.GetProp("tolerance").GetDouble(),
                        Edges = ReadIntList(v.GetProp("edges"))
                    };
                    k.Vertices.Add(kv);
                }
            }

            if (elem.TryGetProp("edges", out var eElem))
            {
                foreach (var e in eElem.EnumerateArray())
                {
                    double[] dom = ReadDoubleArray(e.GetProp("domain"));
                    var ke = new KEdge
                    {
                        Curve3 = e.GetProp("curve3").GetInt32(),
                        Domain = new Interval(dom[0], dom[1]),
                        V0 = e.GetProp("v0").GetInt32(),
                        V1 = e.GetProp("v1").GetInt32(),
                        Trims = ReadIntList(e.GetProp("trims")),
                        Tolerance = e.GetProp("tolerance").GetDouble()
                    };
                    k.Edges.Add(ke);
                }
            }

            if (elem.TryGetProp("trims", out var tElem))
            {
                foreach (var t in tElem.EnumerateArray())
                {
                    double[] dom = ReadDoubleArray(t.GetProp("domain"));
                    string typeStr = t.GetProp("type").GetString();
                    Enum.TryParse<TrimType>(typeStr, true, out var trimType);

                    var kt = new KTrim
                    {
                        Curve2 = t.GetProp("curve2").GetInt32(),
                        Domain = new Interval(dom[0], dom[1]),
                        Edge = t.GetProp("edge").GetInt32(),
                        Loop = t.GetProp("loop").GetInt32(),
                        Reversed3d = t.GetProp("reversed3d").GetBoolean(),
                        Type = trimType,
                        TolU = t.GetProp("tolU").GetDouble(),
                        TolV = t.GetProp("tolV").GetDouble()
                    };
                    k.Trims.Add(kt);
                }
            }

            if (elem.TryGetProp("loops", out var lElem))
            {
                foreach (var l in lElem.EnumerateArray())
                {
                    string typeStr = l.GetProp("type").GetString();
                    Enum.TryParse<LoopType>(typeStr, true, out var loopType);

                    var kl = new KLoop
                    {
                        Face = l.GetProp("face").GetInt32(),
                        Type = loopType,
                        Trims = ReadIntList(l.GetProp("trims"))
                    };
                    k.Loops.Add(kl);
                }
            }

            if (elem.TryGetProp("faces", out var fElem))
            {
                foreach (var f in fElem.EnumerateArray())
                {
                    var kf = new KFace
                    {
                        Surface = f.GetProp("surface").GetInt32(),
                        Reversed = f.GetProp("reversed").GetBoolean(),
                        Loops = ReadIntList(f.GetProp("loops"))
                    };
                    k.Faces.Add(kf);
                }
            }

            return k;
        }

        private static GoldenSplitResult ParseSplitResult(JsonElement elem)
        {
            var res = new GoldenSplitResult
            {
                PieceCount = elem.GetProp("pieceCount").GetInt32()
            };
            if (elem.TryGetProp("pieces", out var pElem))
            {
                foreach (var p in pElem.EnumerateArray())
                {
                    res.Pieces.Add(new GoldenPiece
                    {
                        FaceCount = p.GetProp("faceCount").GetInt32(),
                        Area = p.GetProp("area").GetDouble(),
                        IsValid = p.GetProp("isValid").GetBoolean()
                    });
                }
            }
            return res;
        }

        private static double[] ReadDoubleArray(JsonElement elem)
        {
            var list = new List<double>();
            foreach (var item in elem.EnumerateArray()) list.Add(item.GetDouble());
            return list.ToArray();
        }

        private static List<int> ReadIntList(JsonElement elem)
        {
            var list = new List<int>();
            foreach (var item in elem.EnumerateArray()) list.Add(item.GetInt32());
            return list;
        }

        private static V3 ReadV3(JsonElement elem)
        {
            var e = elem.EnumerateArray();
            e.MoveNext(); double x = e.Current.GetDouble();
            e.MoveNext(); double y = e.Current.GetDouble();
            e.MoveNext(); double z = e.Current.GetDouble();
            return new V3(x, y, z);
        }
    }
}
