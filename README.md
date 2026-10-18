# GhostGPU

Experimental GPU-assisted components for **Rhino 8 and Grasshopper**, written in C# with ILGPU.

GhostGPU keeps point, vector, scalar and mesh data in reusable buffers between compatible nodes. Explicit upload and download boundaries let a connected workflow amortize data-transfer costs. The project also includes mesh Booleans and an experimental NURBS/Brep kernel.

**Status: early development, v0.2.0.** This is a component library, not an automatic replacement for every Grasshopper node. Some operations run on the CPU, and the solid Boolean nodes can fall back to RhinoCommon. Read the [known issues](docs/KNOWN-ISSUES.md) before testing important models. Contributions are welcome.

## What is included

| Area | Components and capabilities |
| --- | --- |
| Data | Upload points, vectors, numbers and meshes; explicit download |
| Points and vectors | Closest point, distance, vector between points, unit vector, amplitude, move |
| Numbers | Arithmetic, bounds, remap, linear/smoothstep/sine maps, colour ramp |
| Meshes | Vertex buffers, displacement, vertex colours, union/difference/intersection |
| Solids | Hybrid Brep union/difference/intersection; optional experimental GhostGPU kernel |
| Diagnostics | Preview/readback, memory counters, session settings |

The runtime has a device memory pool, pinned staging buffers, compact storage options, configurable host execution for small workloads, and spill-to-host support. Double precision is the default; compact float storage is opt-in.

GPU acceleration depends on workload size, data transfers, precision requirements and the CPU algorithm used for comparison. Some Boolean stages remain on the CPU. No universal speedup is claimed.

## Build

For the full Grasshopper plug-in you need Windows x64, a .NET 8 or newer SDK, the .NET Framework 4.8 targeting pack, and an installed Rhino 8 with Grasshopper SDK assemblies. CUDA execution additionally requires a compatible NVIDIA GPU and driver. ILGPU and ILGPU.Algorithms are pinned to 1.5.3.

From the repository root:

```powershell
dotnet build GhostGPU.sln -c Release
powershell -NoProfile -File scripts/Build.ps1 -SkipValidation
```

The second command stages the plug-in and dependency DLLs in `artifacts/GhostGPU-0.2.0`. It does not install them. A nonstandard Rhino installation can be selected with `-RhinoInstallDir 'D:\Rhino 8'` on the build script.

To run validation while packaging, omit `-SkipValidation`. The default checks both CPU debug and CUDA. Use `-ValidationBackend cpu` for a machine without CUDA. CPU debug is explicitly labelled and does not establish GPU performance.

## Try the components

1. Build the staged package. Close Rhino before installing or replacing the plug-in.
2. Find Grasshopper's Libraries directory through **File > Special Folders**. Copy the staged package into its own `GhostGPU` subdirectory, keeping the `.gha` and dependency DLLs together. Unblock downloaded files if Windows requires it.
3. Reopen Rhino/Grasshopper and open [GhostGPU-Attractors.gh](examples/GhostGPU-Attractors.gh).

Other definitions are described in [examples/README.md](examples/README.md). GPU handles belong to a Grasshopper document, are not saved as live GPU allocations, and are recomputed when a definition opens. Ordinary Grasshopper nodes need an explicit download boundary. Preview also reads geometry back to the CPU.

## Validate without Rhino

The runtime and standalone validation projects do not need a Rhino installation:

```powershell
powershell -NoProfile -File scripts/Test.ps1 -Backend cpu
```

Use `-Backend cuda` or `-Backend both` to check real hardware. The kernel harness currently contains unexecuted face-split cases; its successful exit is not complete kernel coverage. See [validation](docs/VALIDATION.md) for commands, fixture provenance and current limits.

## Contribute

Start with [CONTRIBUTING.md](CONTRIBUTING.md), the [architecture](docs/ARCHITECTURE.md), and the [known issues](docs/KNOWN-ISSUES.md). Useful contributions include fixing buffer lifetimes and geometry edge cases, completing regression coverage, improving examples, and measuring comparable CPU/GPU workflows.

Bug reports should include a minimal definition or geometry fixture, Rhino version/runtime, GPU/driver, tolerance, backend, expected result and actual result. Pull requests should preserve existing component GUIDs and data-tree contracts.

## License and dependencies

GhostGPU project code is available under the [MIT license](LICENSE). Dependency notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). RhinoCommon, Grasshopper and Rhino native binaries are supplied by the user's Rhino installation and are not included here. Rhino and Grasshopper are trademarks of Robert McNeel & Associates; this is an independent project.

This public snapshot contains project source, examples and numeric test fixtures. Internal agent files, decompiled research material, local build output and previous private Git history are excluded.
