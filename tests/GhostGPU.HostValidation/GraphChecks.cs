using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Components.IO;
using GhostGPU.Grasshopper.Components.PointVector;
using GhostGPU.Grasshopper.Components.MeshPhase1;
using GhostGPU.Grasshopper.Session;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

internal static class GraphChecks
{
    internal static void Run(string install, GpuBackend backend, bool writeExample)
    {
        using var document = new GH_Document();
        var previousEnable = GH_Document.EnableSolutions;
        GH_Document.EnableSolutions = true;
        var session = GpuSessionManager.GetSession(document, new SessionOptions { Backend = backend, HostWorkThreshold = 0 });
        try
        {
            T Add<T>(T value, float x, float y) where T : IGH_DocumentObject
            { value.CreateAttributes(); value.Attributes.Pivot = new PointF(x, y); document.AddObject(value, false); return value; }
            Param_Number Numeric(string name, double[] first, double[] second, float x, float y)
            {
                var parameter = Add(new Param_Number { NickName = name }, x, y);
                parameter.PersistentData.AppendRange(first.Select(v => new GH_Number(v)), new GH_Path(0, 2));
                parameter.PersistentData.AppendRange(second.Select(v => new GH_Number(v)), new GH_Path(1, 3));
                return parameter;
            }
            void Wire(GH_Component target, int input, IGH_Param source) => target.Params.Input[input].AddSource(source);
            var x = Numeric("X", new double[] { 1, 2, 3 }, new double[] { 4, 5 }, 40, 60);
            var y = Numeric("Y", new double[] { 1 }, new double[] { 2 }, 40, 140);
            var z = Numeric("Z", new double[] { 0 }, new double[] { 0 }, 40, 220);
            var uploads = new[] { Add(new GhUploadNumbers(), 260, 60), Add(new GhUploadNumbers(), 260, 140), Add(new GhUploadNumbers(), 260, 220) };
            var numbers = new[] { x, y, z };
            for (int i = 0; i < 3; i++) Wire(uploads[i], 0, numbers[i]);
            var construct = Add(new GhConstructVector(), 500, 130);
            for (int i = 0; i < 3; i++) Wire(construct, i, uploads[i].Params.Output[0]);
            var length = Add(new GhVectorLength(), 730, 130); Wire(length, 0, construct.Params.Output[0]);
            var download = Add(new GhDownload(), 980, 130); Wire(download, 0, length.Params.Output[0]);

            var nativeAssembly = Assembly.LoadFrom(Path.Combine(install, "Plug-ins", "Grasshopper", "Components", "VectorComponents.gha"));
            GH_Component Native(string name) => (GH_Component)Activator.CreateInstance(nativeAssembly.GetType("VectorComponents.VectorComponents." + name, true));
            var cpuConstruct = Add(Native("Component_ConstructVector"), 500, 350);
            for (int i = 0; i < 3; i++) Wire(cpuConstruct, i, numbers[i]);
            var cpuLength = Add(Native("Component_VectorLength"), 730, 350); Wire(cpuLength, 0, cpuConstruct.Params.Output[0]);

            using var mesh = new Mesh(); mesh.Vertices.UseDoublePrecisionVertices = true;
            mesh.Vertices.Add(0, 0, 0); mesh.Vertices.Add(2, 0, 0); mesh.Vertices.Add(2, 2, 0); mesh.Vertices.Add(0, 2, 0); mesh.Faces.AddFace(0, 1, 2, 3);
            var meshInput = Add(new Param_Mesh { NickName = "Quad mesh" }, 40, 560);
            meshInput.PersistentData.Append(new GH_Mesh(mesh.DuplicateMesh()), new GH_Path(0, 2));
            var meshUpload = Add(new GhUploadMesh(), 260, 560); Wire(meshUpload, 0, meshInput);
            var originInput = Add(new Param_Point { NickName = "Origin" }, 40, 680);
            originInput.PersistentData.Append(new GH_Point(Point3d.Origin), new GH_Path(0, 2));
            var originUpload = Add(new GhUploadPoints(), 260, 680); Wire(originUpload, 0, originInput);
            var factorInput = Numeric("Scale", new double[] { 2 }, new double[] { 2 }, 40, 800);
            var factorUpload = Add(new GhUploadNumbers(), 260, 800); Wire(factorUpload, 0, factorInput);
            var scale = Add(new GhMeshScale(), 500, 590); Wire(scale, 0, meshUpload.Params.Output[0]); Wire(scale, 1, originUpload.Params.Output[0]); Wire(scale, 2, factorUpload.Params.Output[0]);
            // Restrict the factor to the mesh branch so GH does not replicate meshes across unrelated paths.
            factorInput.PersistentData.Clear(); factorInput.PersistentData.Append(new GH_Number(2), new GH_Path(0, 2));
            var face = Add(new GhFaceNormals(), 730, 540); Wire(face, 0, scale.Params.Output[0]);
            var area = Add(new GhFaceAreas(), 730, 670); Wire(area, 0, scale.Params.Output[0]);
            var meshDownload = Add(new GhDownload(), 980, 540); Wire(meshDownload, 1, scale.Params.Output[0]);
            var faceDownload = Add(new GhDownload(), 980, 760); Wire(faceDownload, 0, area.Params.Output[0]);
            var deconstruct = Add(new GhDeconstructMesh(), 730, 850); Wire(deconstruct, 0, scale.Params.Output[0]);

            GH_Document.EnableSolutions = true;
            document.Enabled = true;
            Console.WriteLine($"Starting {backend} private graph: enabled={document.Enabled}, global={GH_Document.EnableSolutions}, objects={document.Objects.Count}");
            document.NewSolution(true, GH_SolutionMode.Silent);
            AssertNoErrors(document);
            var gpuNumbers = download.Params.Output[2].VolatileData.AllData(false).Cast<GH_Number>().Select(n => n.Value).ToArray();
            var cpuNumbers = cpuLength.Params.Output[0].VolatileData.AllData(false).Cast<GH_Number>().Select(n => n.Value).ToArray();
            if (gpuNumbers.Length != 5 || gpuNumbers.Length != cpuNumbers.Length)
            {
                foreach (var component in document.Objects.OfType<GH_Component>())
                    Console.WriteLine($"GRAPH {component.Name} phase={component.Phase} inputs=[{string.Join(",", component.Params.Input.Select(p => p.VolatileData.DataCount))}] outputs=[{string.Join(",", component.Params.Output.Select(p => p.VolatileData.DataCount))}] warnings={string.Join(";", component.RuntimeMessages(GH_RuntimeMessageLevel.Warning))}");
            }
            Require(gpuNumbers.Length == 5 && gpuNumbers.Length == cpuNumbers.Length, $"Native/GPU graph output count mismatch: GPU={gpuNumbers.Length}, CPU={cpuNumbers.Length}.");
            for (int i = 0; i < gpuNumbers.Length; i++) Require(Math.Abs(gpuNumbers[i] - cpuNumbers[i]) < 1e-12, "Native/GPU vector lengths differ.");
            Require(download.Params.Output[2].VolatileData.get_Path(0).ToString() == "{0;2}" && download.Params.Output[2].VolatileData.get_Path(1).ToString() == "{1;3}", "GPU graph lost branch paths.");
            var areas = faceDownload.Params.Output[2].VolatileData.AllData(false).Cast<GH_Number>().ToArray();
            Require(areas.Length == 1 && Math.Abs(areas[0].Value - 16) < 1e-12, "Resident mesh scale/area graph failed.");
            var outputMesh = meshDownload.Params.Output[5].VolatileData.AllData(false).Cast<GH_Mesh>().Single().Value;
            Require(outputMesh.Vertices.Count == 4 && outputMesh.Faces.Count == 1 && outputMesh.Faces[0].IsQuad, "Mesh graph changed fixed topology.");
            Require(outputMesh.Vertices.Point3dAt(2) == new Point3d(4, 4, 0), "Mesh graph transform failed.");

            for (int i = 0; i < 100; i++)
            {
                length.ExpireSolution(false); document.NewSolution(false, GH_SolutionMode.Silent); AssertNoErrors(document);
                Require(download.Params.Output[2].VolatileData.DataCount == 5, "Cached upstream GPU buffer failed on partial recompute.");
            }
            Console.WriteLine($"PASS {backend} live GH graph: native/GPU lengths, two tree paths, fixed-topology mesh scale/analysis, 100 partial recomputes");
            if (writeExample)
            {
                Directory.CreateDirectory("examples");
                var io = new GH_DocumentIO { Document = document };
                Require(io.SaveQuiet(Path.GetFullPath("examples/GhostGPU-Phase1.gh")), "Could not save phase-one example.");
                Console.WriteLine("Wrote examples/GhostGPU-Phase1.gh");
            }
        }
        finally { GpuSessionManager.CloseSession(document); GH_Document.EnableSolutions = previousEnable; }
    }

    private static void AssertNoErrors(GH_Document document)
    {
        foreach (var component in document.Objects.OfType<GH_Component>())
        {
            var messages = component.RuntimeMessages(GH_RuntimeMessageLevel.Error);
            Require(messages.Count == 0, component.Name + ": " + string.Join("; ", messages));
        }
    }
    private static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
}
