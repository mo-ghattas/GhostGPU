# Contributing to GhostGPU

Thank you for helping build GhostGPU. This is experimental software; correctness, useful measurements and reproducible examples matter more than adding nodes quickly.

## Get started

1. Fork the repository and create a branch for one focused change.
2. Read [architecture](docs/ARCHITECTURE.md) and [known issues](docs/KNOWN-ISSUES.md).
3. Run `powershell -NoProfile -File scripts/Test.ps1 -Backend cpu` from the root. This covers standalone projects without requiring Rhino. For full plug-in work, build the solution against your installed Rhino 8 SDK assemblies.
4. Run CUDA and in-Rhino checks when the change affects those paths and you have access to the required hardware/host. Report unavailable checks explicitly.

## Areas where help is useful

- Fix and reproduce the buffer/geometry defects in [known issues](docs/KNOWN-ISSUES.md).
- Replace the placeholder face-split tests with real SSI-to-split checks.
- Add small geometry fixtures for difficult intersections, seams, trims and tangencies.
- Improve connected workflows and documentation for first-time Grasshopper users.
- Measure end-to-end GPU workflows against equivalent, sensible CPU algorithms.

If a change affects a public API, component inputs/outputs, serialized data or an algorithm's scope, discuss the design in an issue first. Small, well-reproduced fixes can go straight to a pull request.

## Contracts to preserve

- Keep component and parameter GUIDs stable. Preserve tree paths, order, empty branches, access modes and repeat-last behavior.
- Preserve immutable buffer semantics, reference ownership and document isolation. Make upload/download boundaries explicit.
- Keep CUDA requests strict; do not silently substitute CPU execution. Label CPU debug and hardware results separately.
- Use double precision by default. Document compact-storage error and derive geometric tolerance from the active document where applicable.
- Do not treat a valid/closed output or matching volume alone as proof of correct geometry.
- Keep installed Rhino SDK references `Private=false`; do not commit SDK/application binaries.

## Pull requests

Explain the failure being addressed, the new behavior, how to reproduce it and what you validated. Include a regression test when it protects a real behavior or boundary. A build alone is not a live-host geometry test.

For performance work, state the GPU/driver, Rhino/runtime, input size, precision, warm-up, CPU reference, transfer costs and what the timing excludes. Separate resident-kernel timing from full-workflow timing.

Please keep discussions respectful and feedback specific. Submit only material you can contribute under the [MIT license](LICENSE), and preserve applicable third-party notices. Exclude proprietary/decompiled source, private customer models, credentials and machine-specific files.
