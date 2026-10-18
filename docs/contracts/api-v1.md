# First node pack contract, version 1

Authoritative interfaces: GhostGPU.Core/GpuContracts.cs and existing V3 in Field.cs.
The old implicit Field model is preserved but is not this implementation's scope.

Runtime implements public `GhostGPU.Runtime.GpuSession : IGpuSession` with
constructor `GpuSession(SessionOptions options = null)`. Pin packages 1.5.3.
Outputs are immutable device buffers, independent lease objects. Retain shares
storage safely; disposing one lease does not invalidate another. Final lease
release frees its allocation. Session Dispose/Reset invalidates all old handles.
Foreign-session, disposed and incorrect-kind buffers are rejected before dispatch.
Metrics is a snapshot, not externally mutable runtime accounting.

Use double precision V3 coordinates. Source uploads reject nonfinite coordinates
and values; outputs may contain IEEE nonfinite results for divide-by-zero. Empty
buffers remain valid zero-length handles. Pairwise operations repeat the last
element of a shorter nonempty buffer, matching GH longest-list behavior; either
empty gives empty. Preserve order. Zero vector Unitize/Amplitude returns zero.
Scalar math, pairwise distances, vector subtraction/move use normal definitions.

ClosestPoints returns nearest target points, original indices and distances in
query order. Exact distance ties choose lowest target index. Nonempty queries
with empty targets are an error. Empty queries give three empty outputs. Initial
algorithm may be direct O(query*target) on GPU, with MaximumNearestPairs guard;
label it honestly. Never allocate a full pairwise distance matrix.

Bounds returns two resident scalars [min,max]; empty bounds are [NaN,NaN]. Remap
consumes those resident bounds (exactly two scalars), maps to targetMin/Max and
optionally clamps. Native reference behavior: a collapsed finite source interval maps
to the target MIDPOINT, matching native Remap Numbers, not targetMin. Native also
leaves its Clipped output unset in the singleton case; explicitly account for this
at the adapter boundary or state that this case is unsupported for that output.
Normal target clamping uses min(targetMin,targetMax)/max(...), preserving reversed
target intervals. Map:
Linear(x)=x; SmoothStep(x)=clamp(x,0,1)^2*(3-2*clamp(x,0,1)); Sine(x)=sin(x).
ColourRamp clamps x to [0,1], linearly interpolates each ARGB byte, rounds to
nearest integer using floor(value+0.5), and returns uint ARGB values. Nonfinite
colour values use lowArgb. All kernels except explicit Download must retain
payload on device; do not call GetAsArray on intermediates. Bounds may use a
simple one-thread GPU reduction for correctness first, with its limitation noted.

Native Mesh Colours accepts an ARGB colour PATTERN and repeats it cyclically, not
repeat-last. Added UploadColours(uint[]) and RepeatColours(GpuBuffer colours,int count)
to the Core interface. Empty input with requested nonzero output is an error;
zero requested count returns empty. ColourRamp is a separately named new convenience
node/function, never silently substituted for native Mesh Colours semantics.

The Grasshopper project references Core and Runtime. Use net48 installed SDK references, no
host-owned DLL copying. Implement typed GPU Goo/parameter with explicit conversion
boundaries. Upload input trees pack each branch into one handle, preserving GH_Path,
order, empty branches; reject null items per branch with an actionable error rather
than dropping them. Downstream GPU components operate one handle per branch via GH
item matching. Download expands buffer elements back into original branch order.
Document this deliberate GPU-handle contract; it is not automatic replacement of
arbitrary existing definitions. Copy/duplicate Goo must retain an independent lease.
GPU handles are ephemeral; serialize metadata only, recompute on reopen.

Mesh wrappers keep immutable original topology and attributes on host, plus GPU
position and optional colour leases. Mesh Vertices is a retained view (no download).
Mesh Displace has explicit per-vertex vectors; preserve faces and attributes.
Mesh Colours supports a resident scalar ramp or colour buffer; do not silently
convert quad topology. Download meshes reconstructs a duplicate of the topology
template with updated vertices/colours. Preview may use an explicitly labelled
readback boundary cached per solve; do not claim zero-copy/direct GPU preview.

First pack target: Upload Points, Upload Numbers, Upload Mesh, Mesh Vertices,
Closest Point, Distance, Vector 2Pt, Unit Vector, Amplitude, Move, scalar math,
Bounds, Remap, Map, Mesh Displace, Mesh Colours, Download and Preview/Diagnostics.
Can combine related functionality in nodes where clear. Set new stable component
GUIDs. Sessions scoped to GH document, clean up on close/removal; buffer leases
must avoid unbounded allocation across repeated solves. Warn if preview downloads.
Default backend CUDA-only, no silent CPU fallback. Provide explicit CpuDebug only
for validation. No auto-install into active Rhino; stage a separate distribution.

Validation uses net8.0 console project referencing Core/Runtime, no test
framework needed. Assert known nearest results/ties, repeat-last operations,
zero vectors/empty buffers, bounds/remap/map/ramp, lease lifetime, reset/stale and
foreign handles, memory guard and resident-chain zero readback until download.
Separate CpuDebug and CudaOnly modes. Supply machine-readable JSON results and a
bounded benchmark with independently computed CPU reference; label naive reference
as such, do not claim optimized kd-tree comparison or mesh/Brep correctness.
