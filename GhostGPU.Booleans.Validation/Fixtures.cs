using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using GhostGPU.Core;

namespace GhostGPU.Booleans.Validation
{
    /// <summary>Boolean inputs exported from the live benchmark .gh (triangulated exactly like MeshBooleanAdapter.ToTriMesh).</summary>
    public sealed class Fixture
    {
        public string Name;
        public BooleanOperation Op;
        public double NativeVolume;
        public TriMesh[] A;
        public TriMesh[] B;

        public static string Dir => Path.Combine(AppContext.BaseDirectory, "fixtures");

        public static Fixture Load(string path)
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            return new Fixture
            {
                Name = Path.GetFileNameWithoutExtension(path),
                Op = Enum.Parse<BooleanOperation>(root.GetProperty("op").GetString()),
                NativeVolume = root.GetProperty("nativeVolume").GetDouble(),
                A = Meshes(root.GetProperty("a")),
                B = Meshes(root.GetProperty("b"))
            };
        }

        private static TriMesh[] Meshes(JsonElement arr)
        {
            var list = new List<TriMesh>();
            foreach (var m in arr.EnumerateArray())
            {
                var v = new List<V3>();
                var e = m.GetProperty("v").EnumerateArray();
                while (e.MoveNext())
                {
                    double x = e.Current.GetDouble(); e.MoveNext();
                    double y = e.Current.GetDouble(); e.MoveNext();
                    v.Add(new V3(x, y, e.Current.GetDouble()));
                }
                var t = new List<int>();
                foreach (var i in m.GetProperty("t").EnumerateArray()) t.Add(i.GetInt32());
                list.Add(new TriMesh(v.ToArray(), t.ToArray()));
            }
            return list.ToArray();
        }
    }
}
