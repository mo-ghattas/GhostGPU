# Known issues

These findings were reproduced during the 18 October 2026 development review. The phase-one expansion fixes K1 and K2 below; the other findings remain open.

## K1: Cached duplicated GPU handles can expire too early — fixed in v0.2.0

**Priority: high.** `GH_GpuBuffer.Duplicate` registers retained Goo for unconditional disposal at the active document's `SolutionEnd`. A duplicated handle can still be held in cached Grasshopper parameter data after that event. A probe confirmed the cached duplicate becomes invalid while the original remains valid. Document selection also depends on the active canvas.

`Duplicate` now gives the duplicate its own retained lease without registering it for disposal on the active canvas's `SolutionEnd`. The owning Goo releases it on disposal/finalization. The isolated Rhino validation exercises the cached duplicate and repeated partial recomputes.

## K2: Mesh Boolean adapters can round double vertices to floats — fixed in v0.2.0

**Priority: high.** `MeshBooleanAdapter.ToTriMesh` uses the `MeshVertexList` indexer, which returns `Point3f`. A double-precision mesh with X values 100000000 and 100000001 became zero-width through an adapter round trip.

The adapter now reads `Point3dAt(i)`. A regression in the Rhino host checks a triangle at X = 100000000 with one-unit separation and its round trip. This fixes the adapter defect; it is not a claim of arbitrary chained Boolean correctness.

## K3: Device lease adoption needs stronger ownership checks

`GpuSession.Adopt` accepts the same lease more than once. Disposing one resulting handle returns the shared block to the pool, allowing an untouched live buffer to change after another upload. A zero-count adoption of a nonempty allocation also leaves its block without an owner.

Fix direction: enforce one successful ownership transfer, validate session/type/count, and release unused allocations for empty results.

## K4: Bezier decomposition can discard distinct knot spans

A fixed absolute knot epsilon merges distinct parameter intervals. A degree-(1,1) surface with knots `[0,0,0.5,0.5+1e-13,1,1]` and control rows at X = 0,1,2,3 produced patch boxes ending at X = 2, although the surface ends at X = 3.

Fix direction: preserve distinct intervals/multiplicity or normalize parameters consistently; add parameter-rescaling and tightly-spaced-knot tests.

## K5: Face-split coverage is not wired into the main harness

`SuiteM3` returns `NotRun` for all ten supplied cases. The harness can exit successfully with that milestone unexecuted. Independent SSI-to-split probes passed the supplied pairs in both directions, but the standard harness does not enforce those checks.

Fix direction: feed actual SSI branches into the split tests, compare per-piece counts/areas/topology and fail missing required coverage. CI running the harness does not resolve this gap.

## Performance and scope limits

Small jobs may run on the host. GPU double-precision throughput depends heavily on hardware. Mesh Booleans retain substantial CPU work, and Brep Booleans may be slower than native operations. Preview reads back data. The current sine implementation uses medium-range argument reduction and has reduced accuracy for very large arguments. See the source comments and API contracts before extending supported input ranges.
