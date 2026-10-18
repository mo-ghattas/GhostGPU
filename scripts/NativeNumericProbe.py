import json
import os
ROOT = os.environ.get("GHOSTGPU_REPO_ROOT")
if not ROOT: raise RuntimeError("Set GHOSTGPU_REPO_ROOT to your repository directory.")
if not os.path.isdir(os.path.join(ROOT, "artifacts", "validation")): os.makedirs(os.path.join(ROOT, "artifacts", "validation"))
import System
import Rhino
from Rhino.Geometry import Vector3d

def run():
    result = []
    for xyz in [(1e308, 1e308, 1e308), (1e-320, 1e-320, 0), (1e-308, 1e-308, 0), (0, 0, 0)]:
        v = Vector3d(*xyz)
        length = v.Length
        success = v.Unitize()
        result.append({"input": xyz, "length": str(length), "unitized": success, "output": [v.X, v.Y, v.Z]})
    with open(os.path.join(ROOT, "artifacts", "validation", "native-numeric.json"), "w") as stream:
        json.dump({"rhino_version": str(Rhino.RhinoApp.Version), "vectors": result}, stream, indent=2)

Rhino.RhinoApp.InvokeOnUiThread(System.Action(run))
