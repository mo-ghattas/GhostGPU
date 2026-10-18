# Resident point, vector and mesh pack

The first-phase expansion adds 34 components for independent point/vector batches and meshes with fixed connectivity. Numeric geometry stays in GhostGPU buffers between compatible components. Native geometry requires explicit upload and download boundaries.

| Group | New GPU components |
| --- | --- |
| Coordinates | Construct Point, Deconstruct Point, Construct Vector, Deconstruct Vector, Cylindrical Point, Spherical Point |
| Vector algebra | Add, Subtract, Reverse, Length, Scale, Divide, Dot Product, Cross Product, Angle |
| Axes | Unit X, Unit Y, Unit Z |
| Transforms | Point Interpolate, Project Point, Project Vector, Rotate Vector, Scale Point, Rotate Point, Mirror Point |
| Mesh analysis | Face Normals and centers, Face Areas, Vertex Normals, Deconstruct Mesh |
| Mesh transforms | Translate, Scale, Rotate, Mirror, Transform |

All component names end in **GPU**. Existing Closest Point, Distance, Vector 2Pt, Unit Vector, Amplitude, Move, Mesh Vertices, Mesh Displace, Mesh Colours and numeric mapping nodes retain their existing GUIDs.

Read the [point/vector contracts](contracts/phase1-point-vector.md) and [mesh contracts](contracts/phase1-mesh.md) for ports, units and edge cases. A resident buffer contains a batch: repeat-last matching happens inside buffers, while Grasshopper matches buffer items across branches. This differs from native geometry/list matching. Mesh transformations preserve vertex count; per-vertex displacement has its own explicit contract.

Where there is a precise native counterpart, its installed icon is recolored at runtime, preserving the native silhouette and alpha mask. Custom operations use original fallback artwork. See [icons](ICONS.md). No extracted native artwork or vendor code is distributed.

## Scope boundary

The earlier node study placed mesh spatial queries, proximity graphs, NURBS evaluation, smoothing and robust Booleans after the first attractor/deformation workflow. This release does not claim every component under Grasshopper's Point, Vector or Mesh menus.

Topology-changing operations (welding, remeshing, deletion, triangulation, hulls and Voronoi), plane/field structures, cumulative scans, sorting/grouping and point-cloud attributes need separate resident data and algorithm contracts. They are not presented as finished GPU implementations here.

## Verification

Standalone tests cover host and device execution, empty/repeat-last batches, invalid/foreign buffers, large coordinates, compact origins, warped quads, isolated vertices and transfer counters. CPU debug and CUDA are checked separately. Mesh index validation can read back a four-byte status flag; it does not download mesh vertices for analysis.

`tests/GhostGPU.HostValidation` uses an isolated installed Rhino runtime to test cached lease lifetime, double-precision mesh adapters, component GUIDs/ports and native icon alpha masks. It does not install into a Rhino profile or edit an open document. See [validation](VALIDATION.md).
