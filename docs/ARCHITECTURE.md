# Architecture

## Projects

| Project | Responsibility |
| --- | --- |
| GhostGPU.Core | Numeric types, typed buffer contracts, session options and metrics |
| GhostGPU.Runtime | ILGPU execution, memory pool, staging, residency, nearest-point/math/colour kernels |
| GhostGPU.Booleans | Triangulated mesh Booleans; GPU broadphase/winding and CPU intersection/retriangulation |
| GhostGPU.Kernel | Experimental NURBS evaluation, SSI, face split and Brep joining; no RhinoCommon reference |
| GhostGPU.Grasshopper | Components, document sessions, Goo ownership, Rhino geometry adapters and hybrid Brep operations |
| GhostGPU.Validation | Runtime conformance and bounded benchmarks |
| GhostGPU.Booleans.Validation | Mesh Boolean fixtures, invariants and independent reference checks |
| GhostGPU.Kernel.Validation | Numeric golden-data checks; currently incomplete for face splitting |

Core, Runtime, Booleans and Kernel target net48/net8.0. The Grasshopper assembly targets net48. The standalone validation programs target net8.0.

## Connected data flow

Upload nodes pack each Grasshopper tree branch into a typed handle. Compatible nodes pass handles referencing session-owned data rather than repeatedly returning full Grasshopper lists. Data may reside on the host for small jobs, on the device, or be spilled back to the host under memory pressure. `Auto` is not a guarantee that an operation runs on the GPU.

The runtime owns allocation pooling and transfers. Handles are document-scoped and ephemeral. Download and preview are readback boundaries. Compact V3F stores float offsets relative to a double-precision buffer origin; compute widens to doubles.

## Geometry pipelines

Mesh Booleans use triangle-box broadphase, CPU exact-predicate intersection and constrained retriangulation, winding-number classification, and output assembly. Winding can dispatch on the host below the configured workload threshold. These components are not an entirely GPU-resident solid modeling kernel.

Brep components default to the RhinoCommon hybrid path: bounding-box checks, exact native splitting, classification, native joining, validation and visible fallback. The optional `GhostGPU` kernel replaces SSI/split/join with the experimental implementation. Its current piece classification uses RhinoCommon point-inside tests. Both modes require Rhino integration at the component boundary.

## Public contracts and fixtures

- [Version 1 numeric contract](contracts/api-v1.md)
- [Version 2 runtime contract](contracts/api-v2.md)
- [Geometry kernel contract](contracts/kernel.md)
- Numeric reference data: `tests/fixtures/golden/`
- Dense mesh fixtures: `GhostGPU.Booleans.Validation/fixtures/`

Contracts describe intended semantics. Read [known issues](KNOWN-ISSUES.md) for implementation gaps. Fixture coverage is bounded and does not establish equivalence to a general CAD kernel.
