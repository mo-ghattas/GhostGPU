# Grasshopper component project

This project builds `GhostGPU.Grasshopper.gha` for net48 against the Rhino 8 assemblies selected by `RhinoInstallDir`. Host assemblies are referenced with `Private=false`; the build does not redistribute them.

See the [root README](../README.md) for build/install instructions, [architecture](../docs/ARCHITECTURE.md) for the project boundaries, [API contracts](../docs/contracts/api-v2.md) for data semantics and [known issues](../docs/KNOWN-ISSUES.md) before editing lease lifetimes or geometry adapters.

Preserve component and parameter GUIDs and existing input/output access modes. Upload nodes pack tree branches into document-scoped handles; ordinary Grasshopper data is produced at explicit readback boundaries. Kernel mode on the solid Boolean nodes is experimental and can fall back with warnings.
