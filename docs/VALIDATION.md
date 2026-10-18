# Validation

## Standalone checks

From the repository root:

```powershell
powershell -NoProfile -File scripts/Test.ps1 -Backend cpu
powershell -NoProfile -File scripts/Test.ps1 -Backend both
```

Reports are written under `artifacts/validation/`. The CPU command runs runtime tests, CPU mesh Boolean tests, and the kernel harness. That harness currently tries its own CUDA evaluation check and reports unavailable CUDA as unexecuted coverage. `-SkipKernel` omits that harness explicitly.

Direct commands:

```powershell
dotnet run --project GhostGPU.Validation -c Release -- --backend cpu --no-benchmarks --output artifacts/validation/runtime-cpu.json
$env:GG_BACKEND = 'CpuDebug'
dotnet run --project GhostGPU.Booleans.Validation -c Release -- --output artifacts/validation/mesh-cpu.json
Remove-Item Env:GG_BACKEND
dotnet run --project GhostGPU.Kernel.Validation -c Release -- --golden-dir tests/fixtures/golden --out artifacts/validation/kernel.json
```

Use `--backend cuda` for the runtime hardware checks. `GG_BACKEND=CudaOnly` selects CUDA for the mesh suite; without a selector that suite runs both backends. The wrapper restores any existing selector after running.

## Reference data

The five JSON files in `tests/fixtures/golden/` store NURBS geometry, evaluated derivatives, SSI samples, split statistics and join statistics exported through RhinoCommon. `scripts/ExportGolden.csx` regenerates them in a licensed Rhino host. It requires `GHOSTGPU_REPO_ROOT` set to this checkout. The script expects a C# runner supplying `doc` and `output`, and System.Text.Json; adapt the wrapper to your host rather than invoking it as a standalone .NET program.

`scripts/NativeReference.py` and `scripts/NativeNumericProbe.py` are optional in-Rhino reference scripts. Set `GHOSTGPU_REPO_ROOT` before starting Rhino so its process inherits the value. They write reports under artifacts and do not install the plug-in or edit Rhino document geometry. They require Python/.NET interoperability in Rhino and are not exercised by standalone CI.

## In-Rhino checks

`BrepBooleanSelfCheck.Run(session, kernel)` compares the hybrid and experimental Brep paths with native results inside Rhino. `KernelBrepAdapter.RoundTripCheck(brep)` checks the adapter boundary. Run host checks in a private test document/process using synthetic fixtures. Do not interpret a console build as a test of serialized Grasshopper definitions, component loading or live tree matching.

The 18 October 2026 review of the development snapshot recorded 77/77 runtime checks and 69/69 mesh Boolean checks on both CpuDebug and an RTX 3050 Ti Laptop GPU. Its kernel harness reported 38 passed and 10 M3 cases not run. Isolated Rhino checks passed 21/21 hybrid and 21/21 kernel cases without fallbacks. These are bounded fixture results, not proof of arbitrary solid-operation correctness. Known defects are listed in [KNOWN-ISSUES.md](KNOWN-ISSUES.md).

## CI scope

GitHub Actions runs the standalone validation programs on Windows with CPU debug. It does not have Rhino or an NVIDIA GPU. Its kernel result must be read alongside the explicit missing-coverage counts. CUDA runs, full plug-in builds and live Grasshopper integration are maintainer/contributor checks.

For benchmarks, report hardware, backend, tolerance, precision, input size, warm-up, transfers and output creation. Compare equivalent results and document the CPU algorithm. Do not use CPU debug timings as GPU evidence.
