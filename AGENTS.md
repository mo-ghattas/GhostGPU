# Working in GhostGPU

Read README.md, CONTRIBUTING.md and docs/KNOWN-ISSUES.md before editing.

- Preserve Grasshopper component GUIDs, input/output contracts, tree paths and ordering.
- Shared numeric and buffer contracts are in GhostGPU.Core and docs/contracts/.
- Keep CUDA execution and CPU debug validation separate. Do not silently fall back from requested CUDA.
- Preserve double-precision defaults and explicit readback boundaries.
- Runtime/kernel validation can run without Rhino. Full plug-in builds require locally installed Rhino 8 SDK assemblies; geometry integration needs a Rhino host.
- Use scripts/Test.ps1 for validation and scripts/Build.ps1 for staging. Do not install into a running Rhino profile or alter a user's open geometry as part of a build.
- Do not commit build outputs, private agent settings, credentials, proprietary binaries or decompiled vendor source.
- Report checks actually run, any skips and known remaining limitations. Do not claim arbitrary Rhino kernel equivalence or universal GPU speedups.
