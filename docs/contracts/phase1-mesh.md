# Phase 1 Fixed-Topology Mesh Analysis & Transforms Specification

**Date:** 2026-10-18  

---

## 1. Overview & Architectural Principles

These operations analyze or transform meshes with fixed connectivity. Vertex calculations stay on resident buffers on the device execution path. Each device index/CSR safety check downloads a four-byte status flag. Host topology conversion and buffer uploads still carry a cost; no general speedup is promised.

### Core Guarantees:

1. **Zero-Readback Device Operations:**  
   Face analysis (centers, normals, areas), vertex normals, and geometric transformations run strictly on resident GPU vertex buffers (`GpuBuffer`, kind `Points`). Vertex coordinates and geometric payloads are never downloaded to host memory for computation.
2. **Triangulation-Independent Nonplanar Quad Normals (Newell Formulation):**  
   For nonplanar quad faces $(A, B, C, D)$, face normal vectors are computed using the diagonal cross-product Newell formulation:  
   $$W = (C - A) \times (D - B) = A \times B + B \times C + C \times D + D \times A$$  
   This formulation is invariant to the choice of triangulation diagonal and equals the sum of boundary edge cross products.
3. **Declared Triangulated Quad Surface Area Contract ($ABC + ACD$):**  
   The surface area of a nonplanar quad is calculated under a declared contract as the sum of its two constitutive triangular facets across the primary diagonal $AC$:  
   $$Area(Quad) = \frac{1}{2} \|(B - A) \times (C - A)\| + \frac{1}{2} \|(C - A) \times (D - A)\|$$  
   Note: This is a declared $AC$ diagonal triangulation contract. Host CAD runtimes such as Rhino may choose an alternate diagonal ($BD$) depending on planarity heuristics or aspect ratio.
4. **Centroid Formulation:**  
   Quad centroids are calculated as the arithmetic mean of all four vertices: $(A + B + C + D) / 4.0$. Triangle centroids use the three-vertex average $(A + B + C) / 3.0$.
5. **Degenerate & Boundary Safety:**  
   Finite zero-area or collinear faces produce zero area and zero normals; their centers remain the vertex average. Non-finite face coordinates produce zero centers, normals and areas. Double-precision range limits still apply to cross products and coordinate sums.
6. **Precomputed Host CSR Adjacency for Race-Free Vertex Normals:**  
   To evaluate vertex normals in parallel on the GPU without atomic contention or write hazards, mesh connectivity is precomputed on the host into Compressed Sparse Row (CSR) format:  
   - `VertexOffsets`: array of length $V + 1$ defining slice bounds.  
   - `AdjacentFaces`: flattened array of incident face indices.  
   GPU kernels assign one thread per vertex to accumulate incident face contributions in parallel. Both `AreaWeighted` (summing unnormalized cross products $W_f$) and `Unweighted` (summing unit normals $N_f$) are supported. Unmeshed or isolated vertices produce zero normal vectors $(0, 0, 0)$.
7. **Safe GPU Indexing & Strict CSR Offsets Validation:**  
   `MeshTopologyDescriptor` and `MeshVertexAdjacency` defensively snapshot all caller-supplied arrays upon construction. Indices and CSR offsets are strictly validated on both host and device (using a lightweight kernel reading back a single 4-byte control flag, without vertex position readback) to ensure all face and adjacency references lie strictly in $[0, V - 1]$ and offsets are monotonic with valid endpoints. Illegal indices or malformed CSR structures throw `IndexOutOfRangeException` or `ArgumentException`.
8. **Fixed-Topology Geometric Transformations:**  
   Translate, Scale, Rotate, Mirror, and 4x4 Affine Transform create new vertex buffers while preserving index ordering, connectivity, vertex count and texture coordinates. Parameter buffers must have count 1 (uniform) or $V$ (per-vertex); arbitrary lengths are rejected. Normal-dependent consumers must recompute from the new positions.
9. **Numerical Parity & IEEE-754 Tolerance:**  
   Host CPU fallback paths share the exact same pure element functions as device kernels, utilizing `Fp.Mul` and `Fp.Sub` to prevent uncontracted FMA rounding discrepancies between CPU and CUDA. Operations adhere to standard IEEE-754 finite double-precision tolerances.

---

## 2. Core Contracts Specification (`GhostGPU.Core/Phase1MeshContracts.cs`)

### 2.1 `MeshFaceIndices`
```csharp
public readonly struct MeshFaceIndices : IEquatable<MeshFaceIndices>
{
    public readonly int A, B, C, D;
    public bool IsTriangle => C == D;
    public bool IsQuad => C != D;
    public MeshFaceIndices(int a, int b, int c, int d);
    public MeshFaceIndices(int a, int b, int c);
}
```

### 2.2 `MeshNormalWeighting`
```csharp
public enum MeshNormalWeighting : byte
{
    Unweighted = 0,
    AreaWeighted = 1
}
```

### 2.3 `MeshVertexAdjacency`
```csharp
public sealed class MeshVertexAdjacency
{
    public int VertexCount { get; }
    public IReadOnlyList<int> VertexOffsets { get; }
    public IReadOnlyList<int> AdjacentFaces { get; }
    public int TotalIncidences => AdjacentFaces.Count;
    public int[] GetOffsetsSnapshot();
    public int[] GetFacesSnapshot();
    public MeshVertexAdjacency(int vertexCount, int[] vertexOffsets, int[] adjacentFaces);
}
```

### 2.4 `MeshTopologyDescriptor`
```csharp
public sealed class MeshTopologyDescriptor
{
    public int VertexCount { get; }
    public int FaceCount { get; }
    public MeshVertexAdjacency Adjacency { get; }
    public IReadOnlyList<MeshFaceIndices> Faces { get; }
    public MeshFaceIndices[] GetFacesSnapshot();
    public static MeshTopologyDescriptor CreateValidated(int vertexCount, MeshFaceIndices[] faces, MeshVertexAdjacency adjacency = null);
    public void Validate();
}
```

### 2.5 `FaceAnalysisResult`
```csharp
public sealed class FaceAnalysisResult : IDisposable
{
    public GpuBuffer Centers { get; }
    public GpuBuffer Normals { get; }
    public GpuBuffer Areas { get; }
    public FaceAnalysisResult(GpuBuffer centers, GpuBuffer normals, GpuBuffer areas);
    public void Dispose();
}
```

---

## 3. Runtime API Specification (`GpuSession.MeshAnalysis.cs`)

All methods are implemented as partial methods of `GhostGPU.Runtime.GpuSession`.

```csharp
// Uploads / downloads face indices
public GpuBuffer UploadMeshFaceIndices(MeshFaceIndices[] faces);
public GpuBuffer UploadMeshFaceIndices(int[] rawQuadIndices);
public MeshFaceIndices[] DownloadMeshFaceIndices(GpuBuffer buffer);

// Unified face analysis (centers, normals, areas) in a single pass
public FaceAnalysisResult AnalyzeMeshFaces(GpuBuffer vertices, GpuBuffer faceIndices);

// Individual face analysis queries
public GpuBuffer MeshFaceCenters(GpuBuffer vertices, GpuBuffer faceIndices);
public GpuBuffer MeshFaceNormals(GpuBuffer vertices, GpuBuffer faceIndices);
public GpuBuffer MeshFaceAreas(GpuBuffer vertices, GpuBuffer faceIndices);

// Parallel race-free vertex normals via CSR adjacency
public GpuBuffer MeshVertexNormals(
    GpuBuffer vertices,
    GpuBuffer faceIndices,
    GpuBuffer vertexOffsets,
    GpuBuffer adjacentFaces,
    MeshNormalWeighting weighting = MeshNormalWeighting.AreaWeighted);

// Fixed-topology geometric transformations (param counts must be 1 or vertex count; affine matrix row 3 must be [0, 0, 0, 1])
public GpuBuffer MeshScale(GpuBuffer vertices, GpuBuffer center, GpuBuffer factor);
public GpuBuffer MeshRotate(GpuBuffer vertices, GpuBuffer axisOrigin, GpuBuffer axisDirection, GpuBuffer angleRadians);
public GpuBuffer MeshMirror(GpuBuffer vertices, GpuBuffer planeOrigin, GpuBuffer planeNormal);
public GpuBuffer MeshTransform(GpuBuffer vertices, double[] matrix4x4); // Strictly requires row 3 == [0, 0, 0, 1]; perspective rejected
```

---

## 4. Grasshopper Components Catalog (`GhostGPU.Grasshopper/Components/MeshPhase1/**`)

| Class Name | Component Name | Nickname | GUID | Native GH Counterpart | Description |
|---|---|---|---|---|---|
| `GhFaceNormals` | Face Normals GPU | `FNorm` | `A194B101-7E23-4211-9A31-92F6F03F8001` | Face Normals (`cb4ca22c-3419-4962-a078-ad4ff7f1f929`, `Component_MeshFaceNormals`) | Calculates center points and Newell normal vectors for all faces |
| `GhFaceAreas` | Face Areas GPU | `FArea` | `A194B102-7E23-4211-9A31-92F6F03F8002` | Custom Analysis | Calculates exact triangulated surface area ($ABC+ACD$) for each face |
| `GhVertexNormals` | Vertex Normals GPU | `VNorm` | `A194B103-7E23-4211-9A31-92F6F03F8003` | Custom Analysis | Computes vertex unit normals via precomputed CSR adjacency |
| `GhDeconstructMesh` | Deconstruct Mesh GPU | `DeMesh` | `A194B104-7E23-4211-9A31-92F6F03F8004` | Deconstruct Mesh (`ba2d8f57-0738-42b4-b5a5-fe4d853517eb`, `Component_DeconstructMesh`) | Deconstructs GPU mesh into vertices, face indices, colours, and normals |
| `GhMeshTranslate` | Mesh Translate GPU | `MTrans` | `A194B105-7E23-4211-9A31-92F6F03F8005` | Move (`e9eb1dcf-92f6-4d4d-84ae-96222d60f56b`, `Component_Move`) | Translates mesh vertices uniformly or per-vertex |
| `GhMeshScale` | Mesh Scale GPU | `MScale` | `A194B106-7E23-4211-9A31-92F6F03F8006` | Scale (`4d2a06bd-4b0f-4c65-9ee0-4220e4c01703`, `Component_Scale`) | Scales mesh vertices around center points (uniform or nonuniform XYZ) |
| `GhMeshRotate` | Mesh Rotate GPU | `MRot` | `A194B107-7E23-4211-9A31-92F6F03F8007` | Rotate Axis (`3ac8e589-37f5-477d-aa61-6699702c5728`, `Component_RotateAxis`) | Rotates mesh vertices around an axis by angle in radians |
| `GhMeshMirror` | Mesh Mirror GPU | `MMirr` | `A194B108-7E23-4211-9A31-92F6F03F8008` | Mirror (`f12daa2f-4fd5-48c1-8ac3-5dea476912ca`, `Component_Mirror`) | Reflects mesh vertices across a plane |
| `GhMeshTransform` | Mesh Transform GPU | `MXform` | `A194B109-7E23-4211-9A31-92F6F03F8009` | Transform (`610e689b-5adc-47b3-af8f-e3a32b7ea341`, `Component_ApplyTransform`) | Applies 4x4 affine transformation matrix to vertices (row 3 must be [0,0,0,1]; projective rejected) |

---

## 5. Explicit Limitations & Non-Goals

1. **Fixed Topology Constraint:**  
   Phase 1 nodes strictly preserve vertex and face counts. Dynamic topology restructuring, remeshing, vertex welding/unwelding, quad-to-triangle subdivision, and mesh decimation are outside the scope of Phase 1.
2. **Spatial Acceleration Queries:**  
   Point-in-mesh tests, ray-mesh intersections, closest point queries, and BVH spatial acceleration trees remain future milestones.
3. **Mesh Booleans:**  
   Solid mesh Booleans (Union, Difference, Intersection) belong to dedicated Boolean pipelines and are not part of fixed-topology analysis.
4. **Mirroring Spatial Winding:**  
   `GhMeshMirror` reflects vertex positions across the plane. While vertex-face indexing topology is strictly preserved, 3D face orientation / winding is reflected; downstream lighting or rendering that requires outward normals should recompute normals.
