# Phase 1 Point & Vector Primitives Specification

**Date:** 2026-10-18  

---

## 1. Overview & Architectural Principles

These operations process resident point/vector batches. Device inputs stay on the device between compatible operations; explicit host execution remains available for small host-resident inputs. Performance depends on workload, precision and transfer costs.

### Core Guarantees:
1. **Device Residency & Immutable Storage:** All point and vector operations consume and return immutable `GpuBuffer` instances. Compute chains stay entirely on the GPU.
2. **Double-Precision Compute (`F64` / `V3D`):** All intermediate calculations and default allocations run in IEEE 754 64-bit double precision.
3. **Compact Format Support (`F32` / `V3F`):** Compact inputs widen on load and outputs narrow on store. Compact point transforms retain a local coordinate origin. Precision depends on the magnitude of the local offsets; compact storage does not guarantee a fixed absolute error.
4. **Repeat-Last Buffer Matching:** When binary or ternary operations receive buffers of differing lengths, shorter buffers repeat their final item up to `max(lengths)`. If any required buffer is empty (`Count == 0`), the output buffer is empty (`Count == 0`).
5. **Numerical Parity:** Host/device paths share element functions. `Fp.Mul` limits fused multiply-add differences. Conformance tests compare numerical results with declared tolerances; universal bit identity or machine-epsilon accuracy for every transcendental input is not promised.
6. **Boundary Rules:** Normalization avoids intermediate overflow, vector division uses direct division for subnormal divisors, and interpolation handles exact endpoints. Ordinary dot/cross products can still overflow double precision for extreme inputs. The per-operation rules below define zero and non-finite behavior.

---

## 2. Public API Specification (`GpuSession.PointVector.cs`)

All methods are implemented as partial methods of `GhostGPU.Runtime.GpuSession`.

### 2.1 Point Construction & Deconstruction

```csharp
// Constructs resident points from coordinate scalar buffers with repeat-last matching
public GpuBuffer ConstructPoints(GpuBuffer x, GpuBuffer y, GpuBuffer z);

// Deconstructs resident points into three scalar buffers (X, Y, Z)
public (GpuBuffer X, GpuBuffer Y, GpuBuffer Z) DeconstructPoints(GpuBuffer points);
public void DeconstructPoints(GpuBuffer points, out GpuBuffer x, out GpuBuffer y, out GpuBuffer z);
```

### 2.2 Vector Construction & Deconstruction

```csharp
// Constructs resident vectors from component scalar buffers with repeat-last matching
public GpuBuffer ConstructVectors(GpuBuffer x, GpuBuffer y, GpuBuffer z);

// Deconstructs resident vectors into three scalar buffers (X, Y, Z)
public (GpuBuffer X, GpuBuffer Y, GpuBuffer Z) DeconstructVectors(GpuBuffer vectors);
public void DeconstructVectors(GpuBuffer vectors, out GpuBuffer x, out GpuBuffer y, out GpuBuffer z);
```

### 2.3 Curvilinear Coordinate Systems

```csharp
// Constructs points from cylindrical coordinates (radius r, azimuth angle theta in radians, elevation z)
public GpuBuffer ConstructCylindricalPoints(GpuBuffer radius, GpuBuffer theta, GpuBuffer z);

// Constructs points from spherical coordinates (radius r, azimuth angle theta in radians, elevation angle phi in radians)
public GpuBuffer ConstructSphericalPoints(GpuBuffer radius, GpuBuffer theta, GpuBuffer phi);
```

### 2.4 Vector Algebra

```csharp
// Vector addition: A + B
public GpuBuffer VectorAdd(GpuBuffer a, GpuBuffer b);

// Vector subtraction: A - B
public GpuBuffer VectorSubtract(GpuBuffer a, GpuBuffer b);

// Vector negation: -V
public GpuBuffer VectorReverse(GpuBuffer vectors);

// Euclidean length: ||V|| (robust hypot without intermediate overflow)
public GpuBuffer VectorLength(GpuBuffer vectors);

// Direct component scaling: V * F (does not unitize first)
public GpuBuffer VectorScale(GpuBuffer vectors, GpuBuffer factors);

// Vector division: V / D (safe division; returns 0 for zero divisors or non-finite inputs)
public GpuBuffer VectorDivide(GpuBuffer vectors, GpuBuffer divisors);

// Scalar dot product: A . B
public GpuBuffer VectorDotProduct(GpuBuffer a, GpuBuffer b);

// Vector cross product: A x B
public GpuBuffer VectorCrossProduct(GpuBuffer a, GpuBuffer b);

// Angle between vectors in radians in [0, pi] (normalized inputs protect against 1e155 overflow)
public GpuBuffer VectorAngle(GpuBuffer a, GpuBuffer b);
```

### 2.5 Unit Axis Vectors

```csharp
// Unit axis vectors optionally scaled by scalar factor buffers (default 1.0)
public GpuBuffer UnitX(GpuBuffer factors = null);
public GpuBuffer UnitY(GpuBuffer factors = null);
public GpuBuffer UnitZ(GpuBuffer factors = null);
```

### 2.6 Interpolation & Projection

```csharp
// Linear interpolation between points: (1-T)*A + T*B (with endpoint guards: t==0 => A, t==1 => B)
public GpuBuffer PointInterpolate(GpuBuffer a, GpuBuffer b, GpuBuffer t);

// Projects resident vectors onto planes defined by normal vectors: V - (V . Unit(N)) * Unit(N)
public GpuBuffer ProjectVectorToPlane(GpuBuffer vectors, GpuBuffer planeNormals);

// Projects resident points onto planes defined by origin points and normal vectors
public GpuBuffer PointProject(GpuBuffer points, GpuBuffer planeOrigins, GpuBuffer planeNormals);

// Rotates vectors around an arbitrary axis vector by angle in radians (Rodrigues formula)
public GpuBuffer RotateVectorAxis(GpuBuffer vectors, GpuBuffer axisVectors, GpuBuffer angles);
```

### 2.7 Point Affine Transforms

```csharp
// Scales points relative to center points: C + (P - C) * F
public GpuBuffer PointScale(GpuBuffer points, GpuBuffer centers, GpuBuffer factors);

// Rotates points around an axis defined by a center point and direction vector
public GpuBuffer PointRotateAxis(GpuBuffer points, GpuBuffer centers, GpuBuffer axisVectors, GpuBuffer angles);

// Reflects points across mirror planes defined by origin points and normal vectors
public GpuBuffer PointMirror(GpuBuffer points, GpuBuffer planeOrigins, GpuBuffer planeNormals);
```

---

## 3. Grasshopper Component Catalog

All components are registered under category `GhostGPU`, subcategory `01 | Vector`, and use `Param_GpuBuffer`.

| # | Class Name | Component Name | Nickname | GUID | Native Counterpart | Description |
|---|------------|----------------|----------|------|--------------------|-------------|
| 1 | `GhConstructPoint` | Construct Point GPU | Pt | `6B0A8211-7F45-4B9D-9457-302E1528BA01` | Construct Point | Constructs resident GPU points from X, Y, Z coordinate scalar buffers. |
| 2 | `GhDeconstructPoint` | Deconstruct Point GPU | DePt | `E623A849-0DF1-4835-B4C2-930B48002712` | Deconstruct | Deconstructs resident GPU points into X, Y, Z coordinate scalar buffers. |
| 3 | `GhConstructVector` | Construct Vector GPU | Vec | `9C465228-44A5-4BC5-88AA-E2E109605333` | Vector XYZ | Constructs resident GPU vectors from X, Y, Z scalar buffers. |
| 4 | `GhDeconstructVector` | Deconstruct Vector GPU | DeVec | `79C1D067-EB8D-476E-8BDD-8820B1333B14` | Deconstruct Vector | Deconstructs resident GPU vectors into X, Y, Z scalar buffers. |
| 5 | `GhCylindricalPoint` | Cylindrical Point GPU | CylPt | `54B27DF9-635E-4B02-B792-506B8F8A2825` | Point Cylindrical | Constructs resident GPU points from cylindrical coordinates (r, theta rad, z). |
| 6 | `GhSphericalPoint` | Spherical Point GPU | SphPt | `19CD16A7-7D9A-4BA4-8A87-A1C9369C9736` | Spherical Cloud | Constructs resident GPU points from spherical coordinates (r, theta rad, phi rad). |
| 7 | `GhVectorAdd` | Vector Add GPU | VAdd | `2867F5B1-8C56-424D-B5DF-C51B6BE48C47` | Addition | Adds two resident GPU vector buffers (A + B). |
| 8 | `GhVectorSubtract` | Vector Subtract GPU | VSub | `86D5A250-9852-4740-97FE-4458EE858658` | Subtraction | Subtracts resident GPU vector buffer B from A (A - B). |
| 9 | `GhVectorReverse` | Vector Reverse GPU | VRev | `342BA8B5-9BE2-4363-9562-F2E515286169` | Reverse | Reverses resident GPU vectors (-V). |
| 10 | `GhVectorLength` | Vector Length GPU | VLen | `4ACFF388-1647-493B-B051-CA18FE5D627A` | Vector Length | Computes Euclidean vector lengths using robust hypot math. |
| 11 | `GhVectorScale` | Vector Scale GPU | VScale | `88939BC8-F648-433C-8472-E7A4A28BE28B` | Multiply | Directly multiplies vector components by scalar factors without unitizing. |
| 12 | `GhVectorDivide` | Vector Divide GPU | VDiv | `3D573DF1-0985-4D3B-A816-B0E329437FE5` | Divide | Divides vector components by scalar divisors (safe zero-divide returns zero vector). |
| 13 | `GhVectorDot` | Vector Dot Product GPU | Dot | `3EE0929F-49C3-4886-A060-61A601E6959C` | Dot Product | Computes scalar dot product of two vector buffers. |
| 14 | `GhVectorCross` | Vector Cross Product GPU | Cross | `F91DC962-DF50-4F5A-8A46-BE0D9E9D75AD` | Cross Product | Computes vector cross product of two vector buffers (A x B). |
| 15 | `GhVectorAngle` | Vector Angle GPU | VAngle | `ACCE1DF2-8B18-4EE4-B974-9C7802D33BBE` | Angle | Computes angle in radians between vectors in range [0, pi]. |
| 16 | `GhUnitX` | Unit X GPU | UnitX | `17D01FA3-75AC-44C1-8B39-86B5A10A2B11` | Unit X | Creates unit X vectors (1, 0, 0), optionally scaled by scalar factor buffer. |
| 17 | `GhUnitY` | Unit Y GPU | UnitY | `8B5D4E12-F559-4D33-A2A9-3A7F11C8E522` | Unit Y | Creates unit Y vectors (0, 1, 0), optionally scaled by scalar factor buffer. |
| 18 | `GhUnitZ` | Unit Z GPU | UnitZ | `4E347A22-3168-45F4-9844-428CE2B7B333` | Unit Z | Creates unit Z vectors (0, 0, 1), optionally scaled by scalar factor buffer. |
| 19 | `GhPointInterpolate` | Point Interpolate GPU | PtInterp | `B7E84B05-C02A-4D39-A1B2-7935216EDECF` | Point Between | Linearly interpolates between points A and B by parameter T. |
| 20 | `GhPointProject` | Project Point GPU | ProjPt | `91B33B88-C488-46E1-8BD5-5EF092A7A444` | Project Point | Projects resident points onto planes defined by origin points and normal vectors. |
| 21 | `GhProjectVectorToPlane` | Project Vector GPU | ProjVec | `C4F02DB5-F329-4112-BF9B-A5C9B39B2AE0` | Project | Projects resident vectors onto planes defined by normal vectors. |
| 22 | `GhRotateVectorAxis` | Rotate Vector GPU | RotVec | `6986C8CE-2E24-42DE-8C56-DA15697669F1` | Rotate | Rotates resident vectors around an axis vector by angle in radians. |
| 23 | `GhPointScale` | Scale Point GPU | ScalePt | `11E1BF3E-C246-4E27-B6DF-D502B0F4E002` | Scale | Scales resident points relative to center points by scalar factors. |
| 24 | `GhPointRotateAxis` | Rotate Point GPU | RotPt | `77FE7C7A-A1BA-4AE1-931B-EFEA420FE913` | Rotate Axis | Rotates resident points around an axis line (center + direction) by angle in radians. |
| 25 | `GhPointMirror` | Mirror Point GPU | MirPt | `EBEFF8DF-7C0F-4D2A-B9C0-1C6DE7CFFD24` | Mirror | Reflects resident points across mirror planes defined by origin points and normal vectors. |

---

## 4. Numerical Behavior & Extreme Cases

1. **Subnormal Coordinates & Divisors:**
   - In `VectorDivide`, division is executed directly per coordinate (`v.X / d`) rather than computing `1.0 / d`. This correctly computes divisions when `d` is subnormal (e.g. `1e-320`), avoiding premature overflow to infinity.
2. **Huge Magnitude Vector Angles:**
   - In `VectorAngle`, vectors are normalized prior to evaluating cross and dot products. Vectors with components at `1e155` or larger avoid squaring overflow in intermediate products, accurately returning angles (e.g. `pi/2`).
3. **Endpoint Stability in Interpolation:**
   - `PointInterpolate` evaluates `(1-T)*A + T*B` with explicit short-circuits: `T == 0.0 => A` and `T == 1.0 => B`. This guarantees exact preservation of endpoints and avoids overflow when `A` and `B` have large opposite coordinates.
4. **Zero & Degenerate Planes:**
   - In `ProjectVectorToPlane`, `PointProject`, and `PointMirror`, if the plane normal has length zero or non-finite components, the input vector/point is returned unchanged without NaN corruption.
5. **Memory Management:**
   - In `DeconstructPoints` and `DeconstructVectors`, three leases are rented with deterministic rollback: if any allocation or adoption fails, all previously acquired leases and buffers are disposed immediately.
