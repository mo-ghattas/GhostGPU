using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using GhostGPU.Core;
using GhostGPU.Runtime;
using GhostGPU.Grasshopper.Parameters;
using GhostGPU.Grasshopper.Session;
using Grasshopper.Kernel;
using Rhino.Geometry;

internal static class Program
{
    [DllImport("kernel32", CharSet = CharSet.Unicode)]
    private static extern bool SetDllDirectory(string path);

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            string install = Environment.GetEnvironmentVariable("RHINO_INSTALL_DIR") ?? @"C:\Program Files\Rhino 8";
            SetDllDirectory(Path.Combine(install, "System"));
            using var core = new Rhino.Runtime.InProcess.RhinoCore(new[] { "/nosplash", "/notemplate" }, Rhino.Runtime.InProcess.WindowStyle.NoWindow);
            Console.WriteLine("Rhino " + typeof(Mesh).Assembly.GetName().Version);
            if (args.Contains("--catalog")) { NativeCatalog(install); return 0; }
            CachedLease();
            DoubleMesh();
            ComponentChecks.Run(install);
            GraphChecks.Run(install, GpuBackend.CpuDebug, args.Contains("--write-example") && !args.Contains("--cuda"));
            if (args.Contains("--cuda")) GraphChecks.Run(install, GpuBackend.CudaOnly, args.Contains("--write-example"));
            Console.WriteLine("Host validation passed.");
            return 0;
        }
        catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    private static void CachedLease()
    {
        using var document = new GH_Document();
        using var editor = new Grasshopper.GUI.GH_DocumentEditor();
        var editorField = typeof(Grasshopper.Instances).GetField("m_docEditor", BindingFlags.NonPublic | BindingFlags.Static);
        var previousEditor = editorField.GetValue(null);
        editorField.SetValue(null, editor);
        Grasshopper.Instances.ActiveCanvas.Document = document;
        using var session = new GpuSession(new SessionOptions { Backend = GpuBackend.CpuDebug, HostWorkThreshold = 0 });
        using var source = new GH_GpuBuffer(session.UploadScalars(new[] { 3.0 }, ElementType.F64, Residency.Device));
        using var duplicate = (GH_GpuBuffer)source.Duplicate();
        var parameter = new Param_GpuBuffer(); parameter.CreateAttributes();
        parameter.AddVolatileData(new Grasshopper.Kernel.Data.GH_Path(0, 2), 0, duplicate);
        document.AddObject(parameter, false);
        // This event previously disposed duplicates via the active canvas.
        var end = typeof(GpuSessionManager).GetMethod("OnDocumentSolutionEnd", BindingFlags.NonPublic | BindingFlags.Static);
        end.Invoke(null, new object[] { document, null });
        source.Dispose();
        Require(duplicate.IsValid && session.DownloadScalars(duplicate.Value)[0] == 3, "Cached retained handle expired.");
        Require(parameter.VolatileData.PathCount == 1 && parameter.VolatileData.get_Path(0).ToString() == "{0;2}", "Cached path changed.");
        duplicate.Dispose();
        Require(session.Metrics.ActiveBytes == 0, "Cached lease was not released.");
        Console.WriteLine("PASS cached duplicate ownership and path");
        Grasshopper.Instances.ActiveCanvas.Document = null;
        editorField.SetValue(null, previousEditor);
    }

    private static void DoubleMesh()
    {
        using var mesh = new Mesh(); mesh.Vertices.UseDoublePrecisionVertices = true;
        mesh.Vertices.Add(100000000d, 0, 0); mesh.Vertices.Add(100000001d, 0, 0); mesh.Vertices.Add(100000001d, 1, 0);
        mesh.Faces.AddFace(0, 1, 2);
        var tri = GhostGPU.Grasshopper.Components.Booleans.MeshBooleanAdapter.ToTriMesh(mesh);
        Require(tri.Vertices[1].X - tri.Vertices[0].X == 1, "Double precision vertices were rounded.");
        using var roundtrip = GhostGPU.Grasshopper.Components.Booleans.MeshBooleanAdapter.ToRhinoMesh(tri);
        Require(roundtrip.Vertices.Point3dAt(1).X - roundtrip.Vertices.Point3dAt(0).X == 1, "Mesh round trip lost double precision.");
        Console.WriteLine("PASS distant mesh double precision");
    }

    private static void NativeCatalog(string install)
    {
        var rows = new List<object>();
        foreach (string path in Directory.GetFiles(Path.Combine(install, "Plug-ins", "Grasshopper", "Components"), "*.gha"))
        {
            Assembly assembly;
            try { assembly = Assembly.LoadFrom(path); } catch { continue; }
            Type[] types;
            try { types = assembly.GetTypes(); } catch (ReflectionTypeLoadException e) { types = e.Types.Where(t => t != null).ToArray(); }
            foreach (var type in types.Where(t => !t.IsAbstract && typeof(GH_Component).IsAssignableFrom(t) && t.GetConstructor(Type.EmptyTypes) != null))
            {
                try
                {
                    var component = (GH_Component)Activator.CreateInstance(type);
                    rows.Add(new { component.Name, component.NickName, component.Category, component.SubCategory, Guid = component.ComponentGuid, Type = type.FullName, Assembly = Path.GetFileName(path), HasIcon = component.Icon_24x24 != null });
                }
                catch { }
            }
        }
        Directory.CreateDirectory("artifacts/phase1");
        File.WriteAllText("artifacts/phase1/native-components.json", JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"Catalogued {rows.Count} native components.");
    }
}
