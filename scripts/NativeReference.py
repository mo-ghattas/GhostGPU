"""Read-only native-node reference cases in private GH documents; never edits RhinoDoc."""
import clr
import os
import json
import System
import Rhino

SYSTEM = Rhino.Runtime.HostUtils.RhinoAssemblyDirectory
RHINO_INSTALL = os.path.dirname(SYSTEM)
GH = os.path.join(RHINO_INSTALL, "Plug-ins", "Grasshopper")
clr.AddReference(os.path.join(GH, "GH_IO.dll"))
clr.AddReference(os.path.join(GH, "Grasshopper.dll"))
import Grasshopper
from Grasshopper.Kernel import GH_Document
from Grasshopper.Kernel.Data import GH_Path
from Grasshopper.Kernel.Types import GH_Point, GH_Number, GH_Interval, GH_Vector, GH_Mesh, GH_Colour
from Rhino.Geometry import Point3d, Vector3d, Interval, Mesh
from System.Reflection import Assembly
from System.Drawing import Color

ROOT = os.environ.get("GHOSTGPU_REPO_ROOT")
if not ROOT: raise RuntimeError("Set GHOSTGPU_REPO_ROOT to your repository directory.")
if not os.path.isdir(os.path.join(ROOT, "artifacts", "validation")): os.makedirs(os.path.join(ROOT, "artifacts", "validation"))
OUT = os.path.join(ROOT, "artifacts", "validation", "native-reference.json")

def value(goo):
    if goo is None:
        return None
    x = goo.Value
    if isinstance(x, Point3d) or isinstance(x, Vector3d):
        return [x.X, x.Y, x.Z]
    if isinstance(x, Interval):
        return [x.T0, x.T1]
    if isinstance(x, Mesh):
        return {"vertices": [[v.X, v.Y, v.Z] for v in x.Vertices],
                "faces": [[f.A, f.B, f.C, f.D] for f in x.Faces],
                "colours": [c.ToArgb() for c in x.VertexColors]}
    return x

def solve(assembly, type_name, inputs):
    document = GH_Document()
    try:
        document.Enabled = True
        asm = Assembly.LoadFrom(os.path.join(GH, "Components", assembly))
        component = asm.CreateInstance(type_name)
        document.AddObject(component, False)
        for index, items in inputs.items():
            parameter = component.Params.Input[index]
            parameter.PersistentData.Clear()
            for item in items:
                parameter.PersistentData.Append(item, GH_Path(0))
        component.ExpireSolution(False)
        document.NewSolution(False)
        if not any(p.VolatileDataCount for p in component.Params.Output):
            component.CollectData()
            component.ComputeData()
        outputs = [[[value(item) for item in branch] for branch in p.VolatileData.Branches]
                   for p in component.Params.Output]
        if not any(p.VolatileDataCount for p in component.Params.Output):
            raise RuntimeError("Native component produced no output: " + type_name)
        return outputs
    finally:
        document.Dispose()

def run():
    active = Rhino.RhinoDoc.ActiveDoc
    before = None if active is None else [active.RuntimeSerialNumber, active.Objects.Count, active.Modified]
    result = {"rhino_version": str(Rhino.RhinoApp.Version),
              "clr": str(System.Environment.Version), "active_model_before": before, "cases": {}}
    cases = result["cases"]
    try:
        cases["closest_tie"] = solve("VectorComponents.gha", "VectorComponents.PointComponents.Component_ClosestPoint",
            {0: [GH_Point(Point3d(0, 0, 0))],
             1: [GH_Point(Point3d(-1, 0, 0)), GH_Point(Point3d(1, 0, 0))]})
        cases["distance_repeat_last"] = solve("VectorComponents.gha", "VectorComponents.PointComponents.Component_Distance",
            {0: [GH_Point(Point3d(0, 0, 0)), GH_Point(Point3d(3, 0, 0)), GH_Point(Point3d(6, 0, 0))],
             1: [GH_Point(Point3d(0, 4, 0)), GH_Point(Point3d(3, 4, 0))]})
        cases["distance_large"] = solve("VectorComponents.gha", "VectorComponents.PointComponents.Component_Distance",
            {0: [GH_Point(Point3d(0, 0, 0))], 1: [GH_Point(Point3d(1e155, 1e155, 0))]})
        cases["amplitude_negative"] = solve("VectorComponents.gha", "VectorComponents.VectorComponents.Component_VectorAmplitude",
            {0: [GH_Vector(Vector3d(3, 4, 0))], 1: [GH_Number(-10)]})
        cases["amplitude_zero"] = solve("VectorComponents.gha", "VectorComponents.VectorComponents.Component_VectorAmplitude",
            {0: [GH_Vector(Vector3d.Zero)], 1: [GH_Number(10)]})
        cases["remap_singleton"] = solve("MathComponents.gha", "MathComponents.DomainComponents.Component_RemapNumbers",
            {0: [GH_Number(7)], 1: [GH_Interval(Interval(7, 7))], 2: [GH_Interval(Interval(10, 20))]})
        cases["remap_reversed"] = solve("MathComponents.gha", "MathComponents.DomainComponents.Component_RemapNumbers",
            {0: [GH_Number(-1), GH_Number(0.25), GH_Number(2)],
             1: [GH_Interval(Interval(0, 1))], 2: [GH_Interval(Interval(20, 10))]})
        mesh = Mesh()
        for xyz in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]:
            mesh.Vertices.Add(*xyz)
        mesh.Faces.AddFace(0, 1, 2, 3)
        cases["mesh_colours_pattern"] = solve("SurfaceComponents.gha", "SurfaceComponents.MeshComponents.Component_MeshColour",
            {0: [GH_Mesh(mesh)], 1: [GH_Colour(Color.Red), GH_Colour(Color.Blue)]})
        active = Rhino.RhinoDoc.ActiveDoc
        after = None if active is None else [active.RuntimeSerialNumber, active.Objects.Count, active.Modified]
        result["active_model_after"] = after
        result["active_model_modified"] = before != after
        if before != after:
            raise RuntimeError("Active model state changed during private reference checks")
        result["status"] = "passed"
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    folder = os.path.dirname(OUT)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(OUT, "w") as stream:
        json.dump(result, stream, indent=2)
    print("GhostGPU native reference: " + result["status"] + " -> " + OUT)

Rhino.RhinoApp.InvokeOnUiThread(System.Action(run))
