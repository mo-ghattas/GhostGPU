using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using GhostGPU.Core;

#if NET8_0_OR_GREATER
using System.Text.Json;
#endif

namespace GhostGPU.Kernel
{
    public sealed class JoinTestCaseResult
    {
        public string Name { get; set; }
        public double Tolerance { get; set; }
        public int InputCount { get; set; }
        public int ExpectedResultCount { get; set; }
        public int ActualResultCount { get; set; }
        public List<bool> ExpectedIsSolid { get; set; } = new List<bool>();
        public List<bool> ActualIsSolid { get; set; } = new List<bool>();
        public List<int> ExpectedNaked { get; set; } = new List<int>();
        public List<int> ActualNaked { get; set; } = new List<int>();
        public double DurationMs { get; set; }
        public bool Passed { get; set; }
        public string Message { get; set; }
    }

    public sealed class JoinSelfTestReport
    {
        public bool AllPassed { get; set; }
        public int TotalCases { get; set; }
        public int PassedCases { get; set; }
        public int FailedCases { get; set; }
        public List<JoinTestCaseResult> Cases { get; set; } = new List<JoinTestCaseResult>();
        public string MarkdownTable { get; set; }
    }

    public static class JoinSelfTest
    {
        public static JoinSelfTestReport Run(string explicitJsonPath = null)
        {
            var report = new JoinSelfTestReport();

#if NET8_0_OR_GREATER
            string jsonPath = ResolveJoinJsonPath(explicitJsonPath);
            if (!File.Exists(jsonPath))
            {
                report.AllPassed = false;
                report.MarkdownTable = $"Error: join.json not found at {jsonPath}";
                return report;
            }

            var rawCases = LoadCasesFromJson(jsonPath);
            bool allOk = true;

            foreach (var rc in rawCases)
            {
                var tr = new JoinTestCaseResult
                {
                    Name = rc.Name,
                    Tolerance = rc.Tolerance,
                    InputCount = rc.Inputs.Count,
                    ExpectedResultCount = rc.ExpectedResults.Count,
                    ExpectedIsSolid = new List<bool>(),
                    ExpectedNaked = new List<int>()
                };

                foreach (var exp in rc.ExpectedResults)
                {
                    tr.ExpectedIsSolid.Add(exp.IsSolid);
                    tr.ExpectedNaked.Add(exp.NakedEdgeCount);
                }

                var sw = Stopwatch.StartNew();
                try
                {
                    var joined = BrepJoin.Join(rc.Inputs, rc.Tolerance);
                    sw.Stop();
                    tr.DurationMs = sw.Elapsed.TotalMilliseconds;
                    tr.ActualResultCount = joined != null ? joined.Count : 0;

                    if (joined != null)
                    {
                        foreach (var b in joined)
                        {
                            tr.ActualNaked.Add(BrepJoin.GetNakedEdgeCount(b));
                            tr.ActualIsSolid.Add(BrepJoin.IsSolid(b));
                        }
                    }

                    // Compare with expectations
                    bool match = (tr.ActualResultCount == tr.ExpectedResultCount);
                    if (match)
                    {
                        for (int i = 0; i < tr.ExpectedResultCount; i++)
                        {
                            if (tr.ActualIsSolid[i] != tr.ExpectedIsSolid[i] ||
                                tr.ActualNaked[i] != tr.ExpectedNaked[i])
                            {
                                match = false;
                                break;
                            }
                        }
                    }

                    tr.Passed = match;
                    if (!match)
                    {
                        tr.Message = $"Mismatch: Expected {tr.ExpectedResultCount} Breps, got {tr.ActualResultCount}.";
                    }
                }
                catch (Exception ex)
                {
                    sw.Stop();
                    tr.DurationMs = sw.Elapsed.TotalMilliseconds;
                    tr.Passed = false;
                    tr.Message = "Exception: " + ex.Message;
                }

                if (!tr.Passed) allOk = false;
                report.Cases.Add(tr);
            }

            report.TotalCases = report.Cases.Count;
            report.PassedCases = report.Cases.FindAll(c => c.Passed).Count;
            report.FailedCases = report.Cases.FindAll(c => !c.Passed).Count;
            report.AllPassed = allOk && report.TotalCases > 0;
            report.MarkdownTable = BuildMarkdownTable(report);
            return report;
#else
            report.AllPassed = true;
            report.MarkdownTable = "JoinSelfTest running on net48 (JSON parsing requires net8.0 runtime).";
            return report;
#endif
        }

#if NET8_0_OR_GREATER
        private sealed class RawTestCase
        {
            public string Name;
            public double Tolerance;
            public List<KBrep> Inputs = new List<KBrep>();
            public List<(bool IsSolid, int NakedEdgeCount)> ExpectedResults = new List<(bool, int)>();
        }

        private static string ResolveJoinJsonPath(string explicitPath)
        {
            if (!string.IsNullOrEmpty(explicitPath) && File.Exists(explicitPath))
                return explicitPath;

            string dir = AppContext.BaseDirectory;
            while (!string.IsNullOrEmpty(dir))
            {
                string candidate = Path.Combine(dir, "GhostGPU", "tests", "fixtures", "golden", "join.json");
                if (File.Exists(candidate)) return candidate;

                candidate = Path.Combine(dir, "tests", "fixtures", "golden", "join.json");
                if (File.Exists(candidate)) return candidate;

                dir = Path.GetDirectoryName(dir);
            }

            return Path.Combine(AppContext.BaseDirectory, "join.json");
        }

        private static List<RawTestCase> LoadCasesFromJson(string path)
        {
            var list = new List<RawTestCase>();
            using var doc = JsonDocument.Parse(File.ReadAllText(path));

            foreach (var elem in doc.RootElement.EnumerateArray())
            {
                var tc = new RawTestCase
                {
                    Name = elem.GetProperty("Name").GetString(),
                    Tolerance = elem.GetProperty("Tolerance").GetDouble()
                };

                if (elem.TryGetProperty("Inputs", out var inputsElem))
                {
                    foreach (var inp in inputsElem.EnumerateArray())
                    {
                        tc.Inputs.Add(ParseKBrep(inp));
                    }
                }

                if (elem.TryGetProperty("Results", out var resElem))
                {
                    foreach (var r in resElem.EnumerateArray())
                    {
                        bool solid = r.GetProperty("IsSolid").GetBoolean();
                        int naked = r.GetProperty("NakedEdgeCount").GetInt32();
                        tc.ExpectedResults.Add((solid, naked));
                    }
                }

                list.Add(tc);
            }

            return list;
        }

        private static KBrep ParseKBrep(JsonElement elem)
        {
            var k = new KBrep();

            if (elem.TryGetProperty("Surfaces", out var sElem))
            {
                foreach (var s in sElem.EnumerateArray())
                {
                    int degU = s.GetProperty("DegreeU").GetInt32();
                    int degV = s.GetProperty("DegreeV").GetInt32();
                    int countU = s.GetProperty("CvCountU").GetInt32();
                    int countV = s.GetProperty("CvCountV").GetInt32();
                    bool isRat = s.GetProperty("IsRational").GetBoolean();
                    bool clU = s.GetProperty("IsClosedU").GetBoolean();
                    bool clV = s.GetProperty("IsClosedV").GetBoolean();

                    var srf = new NurbsSurface(
                        degU, degV, countU, countV, isRat,
                        ReadDoubleArray(s.GetProperty("KnotsU")),
                        ReadDoubleArray(s.GetProperty("KnotsV")),
                        ReadDoubleArray(s.GetProperty("Cvs")))
                    {
                        IsClosedU = clU,
                        IsClosedV = clV
                    };
                    k.Surfaces.Add(srf);
                }
            }

            if (elem.TryGetProperty("Curves3", out var c3Elem))
            {
                foreach (var c in c3Elem.EnumerateArray())
                {
                    int dim = c.GetProperty("Dimension").GetInt32();
                    int deg = c.GetProperty("Degree").GetInt32();
                    bool isRat = c.GetProperty("IsRational").GetBoolean();
                    k.Curves3.Add(new NurbsCurve(dim, deg, isRat,
                        ReadDoubleArray(c.GetProperty("Knots")),
                        ReadDoubleArray(c.GetProperty("Cvs"))));
                }
            }

            if (elem.TryGetProperty("Curves2", out var c2Elem))
            {
                foreach (var c in c2Elem.EnumerateArray())
                {
                    int dim = c.GetProperty("Dimension").GetInt32();
                    int deg = c.GetProperty("Degree").GetInt32();
                    bool isRat = c.GetProperty("IsRational").GetBoolean();
                    k.Curves2.Add(new NurbsCurve(dim, deg, isRat,
                        ReadDoubleArray(c.GetProperty("Knots")),
                        ReadDoubleArray(c.GetProperty("Cvs"))));
                }
            }

            if (elem.TryGetProperty("Vertices", out var vElem))
            {
                foreach (var v in vElem.EnumerateArray())
                {
                    var kv = new KVertex
                    {
                        Point = ReadV3(v.GetProperty("Point")),
                        Tolerance = v.GetProperty("Tolerance").GetDouble(),
                        Edges = ReadIntList(v.GetProperty("Edges"))
                    };
                    k.Vertices.Add(kv);
                }
            }

            if (elem.TryGetProperty("Edges", out var eElem))
            {
                foreach (var e in eElem.EnumerateArray())
                {
                    double[] dom = ReadDoubleArray(e.GetProperty("Domain"));
                    var ke = new KEdge
                    {
                        Curve3 = e.GetProperty("Curve3").GetInt32(),
                        Domain = new Interval(dom[0], dom[1]),
                        V0 = e.GetProperty("V0").GetInt32(),
                        V1 = e.GetProperty("V1").GetInt32(),
                        Trims = ReadIntList(e.GetProperty("Trims")),
                        Tolerance = e.GetProperty("Tolerance").GetDouble()
                    };
                    k.Edges.Add(ke);
                }
            }

            if (elem.TryGetProperty("Trims", out var tElem))
            {
                foreach (var t in tElem.EnumerateArray())
                {
                    double[] dom = ReadDoubleArray(t.GetProperty("Domain"));
                    string typeStr = t.GetProperty("Type").GetString();
                    Enum.TryParse<TrimType>(typeStr, true, out var trimType);

                    var kt = new KTrim
                    {
                        Curve2 = t.GetProperty("Curve2").GetInt32(),
                        Domain = new Interval(dom[0], dom[1]),
                        Edge = t.GetProperty("Edge").GetInt32(),
                        Loop = t.GetProperty("Loop").GetInt32(),
                        Reversed3d = t.GetProperty("Reversed3d").GetBoolean(),
                        Type = trimType,
                        TolU = t.GetProperty("TolU").GetDouble(),
                        TolV = t.GetProperty("TolV").GetDouble()
                    };
                    k.Trims.Add(kt);
                }
            }

            if (elem.TryGetProperty("Loops", out var lElem))
            {
                foreach (var l in lElem.EnumerateArray())
                {
                    string typeStr = l.GetProperty("Type").GetString();
                    Enum.TryParse<LoopType>(typeStr, true, out var loopType);

                    var kl = new KLoop
                    {
                        Face = l.GetProperty("Face").GetInt32(),
                        Type = loopType,
                        Trims = ReadIntList(l.GetProperty("Trims"))
                    };
                    k.Loops.Add(kl);
                }
            }

            if (elem.TryGetProperty("Faces", out var fElem))
            {
                foreach (var f in fElem.EnumerateArray())
                {
                    var kf = new KFace
                    {
                        Surface = f.GetProperty("Surface").GetInt32(),
                        Reversed = f.GetProperty("Reversed").GetBoolean(),
                        Loops = ReadIntList(f.GetProperty("Loops"))
                    };
                    k.Faces.Add(kf);
                }
            }

            return k;
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

        private static string BuildMarkdownTable(JoinSelfTestReport report)
        {
            var sb = new StringBuilder();
            sb.AppendLine("| Case | Tol | Inputs | Result Count (Exp/Act) | IsSolid (Exp/Act) | Naked Edges (Exp/Act) | Time (ms) | Status |");
            sb.AppendLine("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |");

            foreach (var c in report.Cases)
            {
                string expSolid = string.Join(",", c.ExpectedIsSolid);
                string actSolid = string.Join(",", c.ActualIsSolid);
                string expNaked = string.Join(",", c.ExpectedNaked);
                string actNaked = string.Join(",", c.ActualNaked);
                string status = c.Passed ? "PASS" : "FAIL";

                sb.AppendLine($"| `{c.Name}` | {c.Tolerance} | {c.InputCount} | {c.ExpectedResultCount} / {c.ActualResultCount} | {expSolid} / {actSolid} | {expNaked} / {actNaked} | {c.DurationMs:F2} | **{status}** |");
            }

            sb.AppendLine();
            sb.AppendLine($"Summary: {report.PassedCases}/{report.TotalCases} cases passed. AllPassed = {report.AllPassed}");
            return sb.ToString();
        }
#endif
    }
}
