# Runtime API contract, version 2

Supersedes version 1 where they conflict. Shared public types live in GhostGPU.Core/GpuContracts.cs. Preserve existing component GUIDs and numeric/data-tree behavior when changing interfaces.

## 1. Datatypes (Core)

`ElementType { F64, F32, I32, U16, U8, U32, V3D, V3F }`. Compute is always double;
compact formats widen on load and narrow on store inside kernels.
- V3F = float xyz offset from a per-buffer `V3 Origin` (double). Origin = bbox centre at upload.
- Indices auto-narrow to U16 when all values fit [0,65535], else I32. Download widens to int[].
- Masks (`BufferKind.Mask`) are U8 0/1.
- Colours stay packed ARGB U32.
- `GpuBuffer.ByteSize = Count * SizeOf(ElementType)`; budget accounting uses it.
- Existing overloads keep defaults: `SessionOptions.DefaultVectorFormat` (V3D) /
  `DefaultScalarFormat` (F64), `Residency.Auto`.
- Outputs of ops: points/vectors in the widest input format among V3*, scalars F64
  unless every scalar input is F32. Op results never silently lose more precision
  than their inputs.

## 2. Memory model (gpu_runtime)

Everything configurable through `SessionOptions` (see Core). Required pieces, each its
own file under `GhostGPU.Runtime/Memory/`:

- `DeviceMemoryPool`: power-of-two size classes (min 256 B), per-class free lists,
  cached free bytes capped by `PoolCacheBytes` (LRU trim), `TrimPool()` frees all.
  Budget = live + pooled bytes <= `MemoryLimitBytes`; on pressure trim pool first,
  then spill (below), then throw with an actionable message.
- `StagingRing`: `StagingSlots` pinned (page-locked) host buffers of `StagingSlotBytes`,
  round-robin, one ILGPU stream + fence per slot so copy N+1 overlaps kernel N.
  All uploads/downloads go through it; transfers larger than a slot are chunked.
- `DeviceArena`: bump allocator over one pooled block for per-op temporaries
  (reduction partials, grid cells); reset after each op.
- Residency: buffers can be Host-resident (managed array) or Device. `Auto` upload
  keeps payload on the host. A device op promotes inputs (upload once, cached on the
  buffer). An op whose output count < `HostWorkThreshold` and whose inputs are all
  host-resident runs on the host path (counted in `HostDispatches`), no transfer.
  "Not everything is worth the GPU."
- Host path and device kernels share one per-element function (static, no ArrayView),
  so results are bit-identical between paths. Host path uses plain loops /
  Parallel.For above a small size. Never use ILGPU's CPU accelerator for the host path.
- Spill (`SpillPolicy.SpillToHost`): under budget pressure, least-recently-used
  device buffers not in use by the current op are downloaded to the host and their
  device block released; re-promoted on next device use. Buffers are immutable so
  this is always safe. `SpillPolicy.Fail` throws instead.
- Streaming: ops with one huge host-resident input and small resident other inputs
  (e.g. ClosestPoints with 50M host queries, 1k device targets) stream the big input
  through the ring in chunks and never hold it fully on device.
- Metrics: fill every `GpuMetrics` field.

## 3. Kernels (gpu_runtime)

Split `GpuKernels.cs` / `GpuSession.cs` by domain under `GhostGPU.Runtime/Kernels/`:
`VectorKernels`, `ScalarKernels`, `ReductionKernels`, `SpatialKernels`, `ColourKernels`,
plus `KernelCache` (compile once per accelerator, keyed by method). Collapse the
copy-pasted upload/download/binary-op boilerplate in GpuSession into generic helpers.

- Bounds: parallel two-pass reduction (group shared memory or ILGPU.Algorithms),
  NaN handling identical to v1. Keep v1 serial kernel only as a validation oracle.
- ClosestPoints: uniform spatial grid (cell keys + ILGPU.Algorithms radix sort,
  cell start/end, ring-expanding search with exact stop criterion). Ties compare
  (distance, index) lexicographically so lowest target index still wins. Small
  inputs fall back to brute scan; `MaximumNearestPairs` guards brute scans only.
- Grid-stride loops; no per-thread allocation; kernels take compact formats.

## 4. Runtime extension API (gpu_runtime, public, consumed by booleans)

`GhostGPU.Runtime.GpuSession` exposes, in addition to `IGpuSession`:

```csharp
public Accelerator Accelerator { get; }
public SessionOptions Options { get; }                       // clone, read-only use
public TDelegate GetKernel<TDelegate>(Delegate kernelMethod) where TDelegate : Delegate;
    // cached LoadAutoGroupedStreamKernel for the method's signature
public DeviceLease<T> Rent<T>(long count) where T : unmanaged;      // pooled, budget-accounted
public GpuBuffer Adopt<T>(DeviceLease<T> lease, BufferKind kind, ElementType type, int count,
                          V3 origin = default) where T : unmanaged; // ownership moves into an immutable buffer
public ArrayView<T> DeviceView<T>(GpuBuffer buffer) where T : unmanaged; // validates, promotes host->device
public ReadOnlySpan<T> HostView<T>(GpuBuffer buffer) where T : unmanaged; // host-resident payload or throws
public V3 OriginOf(GpuBuffer buffer);                              // V3F origin, zero otherwise
public void CountDispatch(); public void CountHostDispatch();      // metrics hooks for external kernels
public sealed class DeviceLease<T> : IDisposable { public ArrayView<T> View; public long Length; }
```

Publish the final signatures to board key `gpu_runtime.api.extension` as soon as they
compile (before the rest of the refactor), so booleans can build.

## 5. GPU geometry + mesh booleans (booleans)

New project `GhostGPU.Booleans` (net48;net8.0, refs Core + Runtime, ILGPU 1.5.3,
NO RhinoCommon). Namespace `GhostGPU.Booleans`. Public API:

```csharp
public enum BooleanOperation { Union, Difference, Intersection }
public sealed class TriMesh { public V3[] Vertices; public int[] Triangles; } // 3 ints per tri, CCW = outward normal
public sealed class GpuTriMesh : IDisposable {                    // resident positions + indices + per-tri AABB
    public static GpuTriMesh Upload(GpuSession s, TriMesh m, ElementType positionFormat = ElementType.V3D);
    public GpuBuffer Vertices { get; } public GpuBuffer Triangles { get; } public int TriangleCount { get; }
}
public static class GpuGeometry {
    // Stage 1 broadphase. Boxes = min/max point buffers. Returns Indices buffer of (i,j) pairs, 2 ints per pair.
    public static GpuBuffer OverlappingBoxPairs(GpuSession s, GpuBuffer minA, GpuBuffer maxA,
                                                GpuBuffer minB, GpuBuffer maxB, double tolerance);
    // Stage 4. Generalized winding number of each point vs a closed or near-closed mesh. Scalars buffer.
    public static GpuBuffer WindingNumbers(GpuSession s, GpuBuffer points, GpuTriMesh mesh);
}
public sealed class MeshBooleanResult { public TriMesh Mesh; public string[] Warnings; public bool IsClosed; }
public static class MeshBoolean {
    // GH semantics: A list, B list, op applied like native Mesh Boolean components.
    public static MeshBooleanResult Compute(GpuSession s, IReadOnlyList<TriMesh> a, IReadOnlyList<TriMesh> b,
                                            BooleanOperation op, double tolerance);
}
```

Mesh boolean pipeline: BVH or grid broadphase on triangle AABBs (GPU) -> robust
tri-tri intersection segments (GPU, double, filtered orientation predicates) ->
per-triangle constrained re-triangulation (host Parallel.For is fine: streaming
from CPU is allowed where GPU is the wrong tool; say so) -> winding-number
classification of each new triangle (GPU) -> keep/flip by rule table -> vertex weld
(GPU sort/hash). WindingNumbers: start brute O(points*tris) on GPU, then BVH-clustered
(fast winding number) above a size threshold.

Rule table (shared with brep):
| op | keep from A | keep from B |
| --- | --- | --- |
| Union | outside B | outside A |
| Intersection | inside B | inside A |
| Difference | outside B | inside A, normals/winding FLIPPED |
Inside = |W| >= 0.5. Coincident faces (|W| near 0.5) are a documented limitation unless resolved.

Validation project `GhostGPU.Booleans.Validation` (net8.0 console, CpuDebug + CUDA):
box/sphere/overlapping-box cases, volume by divergence theorem vs analytic, closedness,
winding numbers vs brute oracle, empty/disjoint cases. JSON results like existing validation.

## 6. Hybrid analytic Brep booleans + GH (brep)

Lives in `GhostGPU.Grasshopper` (needs RhinoCommon). Folder `Brep/` for the pipeline,
`Components/Booleans/` for nodes. References `GhostGPU.Booleans`.

The hybrid Brep path uses these five stages, with native behavior checked through RhinoCommon reference fixtures:
0. Prep exactly like native `EnsureCorrectBrep`: duplicate, SplitKinkyFaces with clamped
   angle tolerance, Flip if SolidOrientation is Inward. Document tolerance from GH.
1. GPU stage: per-face bounding boxes for every A and B brep -> `OverlappingBoxPairs`
   in ONE batched dispatch across all pairs. Disjoint brep pairs short-circuit
   (Difference keeps A, Union keeps both, Intersection drops).
2-3. Exact stage on RhinoCommon: intersection + split of touched breps
   (verify the exact Brep.Split / Intersection.BrepBrep overloads against installed
   Rhino 8 RhinoCommon before using them).
4. GPU stage: tessellate each operand once (GpuTriMesh), take one interior sample
   point per split sub-face, classify ALL sample points of ALL pairs with batched
   `WindingNumbers`, apply the rule table (flip B faces for Difference).
5. Join (Brep.JoinBreps), validate IsValid/IsSolid. On failure fall back to the native
   Brep.CreateBooleanX for that pair with a visible runtime warning — never silent.
Iteration order and outputs match native SUnion / SDiff / SInt components.

GH work also: Mesh Boolean GPU components (Union/Difference/Intersection) over
`MeshBoolean.Compute`; a "GPU Session Settings" component exposing SessionOptions
(budget, pool cache, staging ring, host threshold, spill, default formats) that
recreates the document session; Diagnostics shows the new metrics; Upload nodes get
optional format/residency inputs. Existing components keep working. Preserve tree
paths, nulls produce errors not silent drops.

## Validation expectations

- Builds: `dotnet build GhostGPU.sln -c Release`.
- Existing validation still passes in CpuDebug and CUDA; new tests added for new code.
- Report: files changed, checks run with real output, hardware vs CpuDebug
  results reported separately, known limitations. Document public API changes in the pull request.
