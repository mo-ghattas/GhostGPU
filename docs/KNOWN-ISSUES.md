# Known issues

These findings were reproduced during the 18 October 2026 development review. This public preparation preserves the algorithms and exposes the issues for focused fixes; cleanup is not a correctness release.

## K1: Cached duplicated GPU handles can expire too early

**Priority: high.** `GH_GpuBuffer.Duplicate` registers retained Goo for unconditional disposal at the active document's `SolutionEnd`. A duplicated handle can still be held in cached Grasshopper parameter data after that event. A probe confirmed the cached duplicate becomes invalid while the original remains valid. Document selection also depends on the active canvas.

Fix direction: attach lease ownership to the actual consumer/cache lifetime. Add a partial-recompute regression case.

## K2: Mesh Boolean adapters can round double vertices to floats

**Priority: high.** `MeshBooleanAdapter.ToTriMesh` uses the `MeshVertexList` indexer, which returns `Point3f`. A double-precision mesh with X values 100000000 and 100000001 became zero-width through an adapter round trip.

Fix direction: use `Point3dAt(i)` and test distant coordinates and chained Booleans.

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
