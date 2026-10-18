# GhostGPU Native Icon Recoloring & Correspondence Architecture

This document describes the runtime icon resolution, color transformation engine, and component correspondence architecture for GhostGPU in Grasshopper.

---

## 1. Architectural Principles

GhostGPU components maintain visual coherence with native Grasshopper components while providing an immediate, unambiguous visual identity as GPU-accelerated operations.

### Key Tenets
1. **Dynamic Runtime Resolution (No Copied Assets)**
   - No native Grasshopper bitmaps, `.png` files, or decompiled resources are stored in the repository.
   - Icons are retrieved dynamically at runtime from the host Rhino 8 / Grasshopper environment via proxy inspection (`Grasshopper.Instances.ComponentServer`) or direct reflection on installed Grasshopper component assemblies (`C:\Program Files\Rhino 8\Plug-ins\Grasshopper\Components\*.gha`).
2. **Byte-for-Byte Alpha Channel Preservation**
   - The native icon's shape, antialiasing, and silhouette are preserved exactly.
   - For every pixel $(x, y)$, destination alpha strictly matches source alpha: $\text{dest.A} = \text{source.A}$.
   - Fully transparent pixels ($\text{A} = 0$) are zeroed out (`0x00000000`), ensuring hidden RGB artifacts in transparent regions have no effect.
3. **GhostGPU Cyan/Teal Brand Hue Transformation**
   - RGB colors are transformed in HSL color space.
   - Colored accents (orange, red, green, blue) have their hue shifted to the GhostGPU cyan/teal hue ($\text{TargetHue} = 186^\circ$, corresponding to `#00E5FF` / `#00B4D8`).
   - Luminance contrast and shading gradients are preserved.
   - Dark outlines ($\text{Lightness} < 0.22$) and white highlights ($\text{Lightness} > 0.94$) are maintained on colored icons to keep icon iconography crisp and readable.
   - For fully monochrome icons (e.g. Construct Point, Deconstruct Point), black/dark stroke pixels ($\text{Lightness} \le 0.95$) are recolored to visible electric teal ($\text{TargetHue} = 186^\circ$, $\text{Saturation} = 0.75$, $\text{Lightness} \ge 0.38$) while white highlights and exact alpha bytes are strictly preserved. This guarantees at least one visible RGB pixel change for GPU identity across all components.
4. **Deterministic Non-Obsolete Mappings**
   - Mappings from GhostGPU components to native Grasshopper components are established by exact GUID and type names derived directly from installed Rhino 8 metadata (`artifacts/phase1/native-components.json`).
   - Mappings strictly target current, active, non-obsolete Grasshopper 8 components with strict GUID verification prior to icon extraction.
5. **Procedural Fallback Generator**
   - Components lacking a direct native counterpart (e.g. `GhSphericalPoint`, `GhPointInterpolate`, `GhFaceAreas`, `GhVertexNormals`, `GhMeshDisplace`, `GhSessionSettings`, `GhPreviewDiagnostics`, and custom upload/download components) receive procedurally generated 24x24 icons with a rounded slate badge, cyan top accent, and clear monogram labels.
6. **Thread Safety & Bounded Process Cache**
   - Cache access and bitmap creation are serialized via synchronization locks to eliminate race conditions, duplicate allocations, and orphaned GDI+ bitmap leaks.
   - Normal component usage caches one rendered bitmap per component GUID (61 component types in v0.2.0). External callers supplying arbitrary cache keys can increase this count.
   - `ClearCache()` resets internal dictionary entries. Active references held by Grasshopper canvas renderers remain valid, and memory is reclaimed by the garbage collector when no references remain.
7. **Anti-Recursion Protection**
   - When inspecting `ComponentServer.ObjectProxies`, the extractor explicitly ignores proxies belonging to `GhostGPU`, preventing recursive icon lookup loops.

---

## 2. Public API Structure

The icon infrastructure is located in `GhostGPU.Grasshopper/Icons/`:

- **`GhostGpuIconProvider`**: Public facade called by `GhostGpuComponentBase.Icon`. Handles caching, proxy resolution, recoloring, and fallback dispatch.
- **`NativeIconCatalog`**: Explicit lookup dictionary mapping GhostGPU component GUIDs and class names to native counterpart metadata.
- **`NativeIconExtractor`**: Safely retrieves native source bitmaps via Grasshopper `ComponentServer` or isolated assembly reflection.
- **`IconRecolorEngine`**: High-performance pixel manipulation engine executing safe managed memory scans (`Marshal.Copy`) and HSL hue rotation.
- **`FallbackIconGenerator`**: Procedural GDI+ vector renderer for custom GPU nodes.

The single integration hook in `GhostGPU.Grasshopper/Components/ComponentBase.cs` is:
```csharp
protected override System.Drawing.Bitmap Icon => Icons.GhostGpuIconProvider.GetIcon(this);
```

---

## 3. Authoritative Component Mapping Catalog

### A. Point & Vector Expansion Components (Phase 1)
| GhostGPU Class | Component Name | GhostGPU GUID | Native Counterpart | Native GUID | Native Type | Assembly |
|---|---|---|---|---|---|---|
| `GhConstructPoint` | Construct Point GPU | `6B0A8211-7F45-4B9D-9457-302E1528BA01` | Construct Point | `3581f42a-9592-4549-bd6b-1c0fc39d067b` | `Component_ConstructPoint` | `VectorComponents.gha` |
| `GhDeconstructPoint` | Deconstruct Point GPU | `E623A849-0DF1-4835-B4C2-930B48002712` | Deconstruct | `9abae6b7-fa1d-448c-9209-4a8155345841` | `Component_DeconstructPoint` | `VectorComponents.gha` |
| `GhConstructVector` | Construct Vector GPU | `9C465228-44A5-4BC5-88AA-E2E109605333` | Vector XYZ | `56b92eab-d121-43f7-94d3-6cd8f0ddead8` | `Component_ConstructVector` | `VectorComponents.gha` |
| `GhDeconstructVector` | Deconstruct Vector GPU | `79C1D067-EB8D-476E-8BDD-8820B1333B14` | Deconstruct Vector | `a50fcd4a-cf42-4c3f-8616-022761e6cc93` | `Component_DeconstructVector` | `VectorComponents.gha` |
| `GhCylindricalPoint` | Cylindrical Point GPU | `54B27DF9-635E-4B02-B792-506B8F8A2825` | Point Cylindrical | `23603075-be64-4d86-9294-c3c125a12104` | `Component_PointCylindrical` | `VectorComponents.gha` |
| `GhSphericalPoint` | Spherical Point GPU | `19CD16A7-7D9A-4BA4-8A87-A1C9369C9736` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhVectorAdd` | Vector Add GPU | `2867F5B1-8C56-424D-B5DF-C51B6BE48C47` | Addition (VAdd) | `fb012ef9-4734-4049-84a0-b92b85bb09da` | `Component_VectorSum` | `VectorComponents.gha` |
| `GhVectorSubtract` | Vector Subtract GPU | `86D5A250-9852-4740-97FE-4458EE858658` | Subtraction | `9c007a04-d0d9-48e4-9da3-9ba142bc4d46` | `Component_VariableSubtraction` | `MathComponents.gha` |
| `GhVectorReverse` | Vector Reverse GPU | `342BA8B5-9BE2-4363-9562-F2E515286169` | Reverse | `d5788074-d75d-4021-b1a3-0bf992928584` | `Component_VectorReverse` | `VectorComponents.gha` |
| `GhVectorLength` | Vector Length GPU | `4ACFF388-1647-493B-B051-CA18FE5D627A` | Vector Length | `675e31bf-1775-48d7-bb8d-76b77786dd53` | `Component_VectorLength` | `VectorComponents.gha` |
| `GhVectorScale` | Vector Scale GPU | `88939BC8-F648-433C-8472-E7A4A28BE28B` | Multiply | `63fff845-7c61-4dfb-ba12-44d481b4bf0f` | `Component_VectorMultiply` | `VectorComponents.gha` |
| `GhVectorDivide` | Vector Divide GPU | `3D573DF1-0985-4D3B-A816-B0E329437FE5` | Divide | `310e1065-d03a-4858-bcd1-809d39c042af` | `Component_VectorDivide` | `VectorComponents.gha` |
| `GhVectorDot` | Vector Dot Product GPU | `3EE0929F-49C3-4886-A060-61A601E6959C` | Dot Product | `43b9ea8f-f772-40f2-9880-011a9c3cbbb0` | `Component_VectorDotProduct` | `VectorComponents.gha` |
| `GhVectorCross` | Vector Cross Product GPU | `F91DC962-DF50-4F5A-8A46-BE0D9E9D75AD` | Cross Product | `2a5cfb31-028a-4b34-b4e1-9b20ae15312e` | `Component_VectorCrossProduct` | `VectorComponents.gha` |
| `GhVectorAngle` | Vector Angle GPU | `ACCE1DF2-8B18-4EE4-B974-9C7802D33BBE` | Angle | `b464fccb-50e7-41bd-9789-8438db9bea9f` | `Component_VectorAngle` | `VectorComponents.gha` |
| `GhUnitX` | Unit X GPU | `17D01FA3-75AC-44C1-8B39-86B5A10A2B11` | Unit X | `79f9fbb3-8f1d-4d9a-88a9-f7961b1012cd` | `Component_UnitVectorX` | `VectorComponents.gha` |
| `GhUnitY` | Unit Y GPU | `8B5D4E12-F559-4D33-A2A9-3A7F11C8E522` | Unit Y | `d3d195ea-2d59-4ffa-90b1-8b7ff3369f69` | `Component_UnitVectorY` | `VectorComponents.gha` |
| `GhUnitZ` | Unit Z GPU | `4E347A22-3168-45F4-9844-428CE2B7B333` | Unit Z | `9103c240-a6a9-4223-9b42-dbd19bf38e2b` | `Component_UnitVectorZ` | `VectorComponents.gha` |
| `GhPointInterpolate` | Point Interpolate GPU | `B7E84B05-C02A-4D39-A1B2-7935216EDECF` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhPointProject` | Project Point GPU | `91B33B88-C488-46E1-8BD5-5EF092A7A444` | Project Point | `5184b8cb-b71e-4def-a590-cd2c9bc58906` | `Component_ProjectPoint` | `TriangulationComponents.gha` |
| `GhProjectVectorToPlane` | Project Vector GPU | `C4F02DB5-F329-4112-BF9B-A5C9B39B2AE0` | Project | `23285717-156c-468f-a691-b242488c06a6` | `Component_Project` | `XformComponents.gha` |
| `GhRotateVectorAxis` | Rotate Vector GPU | `6986C8CE-2E24-42DE-8C56-DA15697669F1` | Rotate | `b6d7ba20-cf74-4191-a756-2216a36e30a7` | `Component_VectorRotate` | `VectorComponents.gha` |
| `GhPointScale` | Scale Point GPU | `11E1BF3E-C246-4E27-B6DF-D502B0F4E002` | Scale | `4d2a06bd-4b0f-4c65-9ee0-4220e4c01703` | `Component_Scale` | `XformComponents.gha` |
| `GhPointRotateAxis` | Rotate Point GPU | `77FE7C7A-A1BA-4AE1-931B-EFEA420FE913` | Rotate Axis | `3ac8e589-37f5-477d-aa61-6699702c5728` | `Component_RotateAxis` | `XformComponents.gha` |
| `GhPointMirror` | Mirror Point GPU | `EBEFF8DF-7C0F-4D2A-B9C0-1C6DE7CFFD24` | Mirror | `f12daa2f-4fd5-48c1-8ac3-5dea476912ca` | `Component_Mirror` | `XformComponents.gha` |

### B. Fixed-Topology Mesh Components (Phase 1)
| GhostGPU Class | Component Name | GhostGPU GUID | Native Counterpart | Native GUID | Native Type | Assembly |
|---|---|---|---|---|---|---|
| `GhFaceNormals` | Face Normals GPU | `A194B101-7E23-4211-9A31-92F6F03F8001` | Face Normals | `cb4ca22c-3419-4962-a078-ad4ff7f1f929` | `Component_MeshFaceNormals` | `SurfaceComponents.gha` |
| `GhFaceAreas` | Face Areas GPU | `A194B102-7E23-4211-9A31-92F6F03F8002` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhVertexNormals` | Vertex Normals GPU | `A194B103-7E23-4211-9A31-92F6F03F8003` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhDeconstructMesh` | Deconstruct Mesh GPU | `A194B104-7E23-4211-9A31-92F6F03F8004` | Deconstruct Mesh | `ba2d8f57-0738-42b4-b5a5-fe4d853517eb` | `Component_DeconstructMesh` | `SurfaceComponents.gha` |
| `GhMeshTranslate` | Mesh Translate GPU | `A194B105-7E23-4211-9A31-92F6F03F8005` | Move | `e9eb1dcf-92f6-4d4d-84ae-96222d60f56b` | `Component_Move` | `XformComponents.gha` |
| `GhMeshScale` | Mesh Scale GPU | `A194B106-7E23-4211-9A31-92F6F03F8006` | Scale | `4d2a06bd-4b0f-4c65-9ee0-4220e4c01703` | `Component_Scale` | `XformComponents.gha` |
| `GhMeshRotate` | Mesh Rotate GPU | `A194B107-7E23-4211-9A31-92F6F03F8007` | Rotate Axis | `3ac8e589-37f5-477d-aa61-6699702c5728` | `Component_RotateAxis` | `XformComponents.gha` |
| `GhMeshMirror` | Mesh Mirror GPU | `A194B108-7E23-4211-9A31-92F6F03F8008` | Mirror | `f12daa2f-4fd5-48c1-8ac3-5dea476912ca` | `Component_Mirror` | `XformComponents.gha` |
| `GhMeshTransform` | Mesh Transform GPU | `A194B109-7E23-4211-9A31-92F6F03F8009` | Transform | `610e689b-5adc-47b3-af8f-e3a32b7ea341` | `Component_ApplyTransform` | `XformComponents.gha` |

### C. Foundation Components
| GhostGPU Class | Component Name | GhostGPU GUID | Native Counterpart | Native GUID | Native Type | Assembly |
|---|---|---|---|---|---|---|
| `GhBounds` | Bounds GPU | `318C7E77-FB9D-40C6-9EE3-EF0F5EE73112` | Bounds | `f44b92b0-3b5b-493a-86f4-fd7408c3daf3` | `Component_NumericBounds` | `MathComponents.gha` |
| `GhColourRamp` | Colour Ramp GPU | `B687C804-A759-4E6D-B27F-7B27C1748117` | Colour Gradient | `298e0adf-625a-4207-9d0f-c22212480c4c` | `Component_DisplayColorGradient` | `IOComponents.gha` |
| `GhMap` | Map GPU | `1251E14F-C00D-477D-9730-C7BAAC88F114` | Remap Numbers | `2fcc2743-8339-4cdf-a046-a1f17439191d` | `Component_RemapNumbers` | `MathComponents.gha` |
| `GhRemap` | Remap GPU | `68BD017C-BE9C-4CD5-827B-CE74F34A8113` | Remap Numbers | `2fcc2743-8339-4cdf-a046-a1f17439191d` | `Component_RemapNumbers` | `MathComponents.gha` |
| `GhScalarMath` | Scalar Math GPU | `8688D0F1-5246-4B89-AE94-4F81BF6AA111` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhMeshColours` | Mesh Colours GPU | `9846E062-8BC4-4C5D-913C-5586617A2116` | Mesh Colours | `d2cedf38-1149-4adc-8dbf-b06571cb5106` | `Component_MeshColour` | `SurfaceComponents.gha` |
| `GhMeshDisplace` | Mesh Displace GPU | `8DA3A7BD-BC84-48FE-8B4F-A16DC19A6115` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhMeshVertices` | Mesh Vertices GPU | `24BC97D1-85A3-4DC1-8314-2C6F158BA104` | Deconstruct Mesh | `ba2d8f57-0738-42b4-b5a5-fe4d853517eb` | `Component_DeconstructMesh` | `SurfaceComponents.gha` |
| `GhAmplitude` | Amplitude GPU | `23000673-E2E1-4FFD-8B42-4D907CFB2109` | Amplitude | `6ec39468-dae7-4ffa-a766-f2ab22a2c62e` | `Component_VectorAmplitude` | `VectorComponents.gha` |
| `GhClosestPoint` | Closest Point GPU | `71708AC6-B7E0-4BA8-868C-D9B97F7A6105` | Closest Point | `571ca323-6e55-425a-bf9e-ee103c7ba4b9` | `Component_ClosestPoint` | `VectorComponents.gha` |
| `GhDistance` | Distance GPU | `F0D9BC05-3948-4394-B6AA-EF78A9C22106` | Distance | `93b8e93d-f932-402c-b435-84be04d87666` | `Component_Distance` | `VectorComponents.gha` |
| `GhMove` | Move GPU | `0994B67E-B8B8-43F1-9BE4-332EB7AEB110` | Move | `e9eb1dcf-92f6-4d4d-84ae-96222d60f56b` | `Component_Move` | `XformComponents.gha` |
| `GhUnitVector` | Unit Vector GPU | `C750798C-D3F1-4F2D-BCBF-DF06D8DFB108` | Unit Vector | `d2da1306-259a-4994-85a4-672d8a4c7805` | `Component_UnitVector` | `VectorComponents.gha` |
| `GhVector2Pt` | Vector 2Pt GPU | `5E035987-9DE8-48D5-BC40-EBCBE0114107` | Vector 2Pt | `934ede4a-924a-4973-bb05-0dc4b36fae75` | `Component_Vector2Pt` | `VectorComponents.gha` |
| `GhMeshDifference` | Mesh Difference GPU | `50ED9CD5-C557-4741-8404-7603EF5E2B5F` | Mesh Difference | `4f3147f4-9fcd-4a7e-be0e-b1841caa5f97` | `Component_MeshBooleanDifference` | `SurfaceComponents.gha` |
| `GhMeshIntersection` | Mesh Intersection GPU | `A1372AA7-6986-4982-A5B2-05069149B0E5` | Mesh Intersection | `95aef4f6-66fc-477e-b8f8-32395a837831` | `Component_MeshBooleanIntersection` | `SurfaceComponents.gha` |
| `GhMeshUnion` | Mesh Union GPU | `9FD9BFE8-4ACF-472E-8DA4-076B55524059` | Mesh Union | `88060a82-0bf7-46bb-9af8-bdc860cf7e1d` | `Component_MeshBooleanUnion` | `SurfaceComponents.gha` |
| `GhSolidDifference` | Solid Difference GPU | `2F5D629E-2468-4DC3-83BE-FD4693A5796B` | Solid Difference | `fab11c30-2d9c-4d15-ab3c-2289f1ae5c21` | `Component_BooleanDifference` | `SurfaceComponents.gha` |
| `GhSolidIntersection` | Solid Intersection GPU | `16E286EE-2659-41B0-9625-E19A0E2F059F` | Solid Intersection | `5723c845-cafc-442d-a667-8c76532845e6` | `Component_BooleanIntersection` | `SurfaceComponents.gha` |
| `GhSolidUnion` | Solid Union GPU | `627AD9C1-0D3C-4BCC-BE6F-B8BBBB23C6A1` | Solid Union | `10434a15-da85-4281-bb64-a2b3a995b9c6` | `Component_BooleanUnion` | `SurfaceComponents.gha` |
| `GhDownload` | Download GPU | `35E00A63-958C-4FA9-A6CE-7B4D228DF118` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhUploadMesh` | Upload Mesh GPU | `54B55E5D-47C0-463E-B19E-6C0D9B9DF103` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhUploadNumbers` | Upload Numbers GPU | `C475A952-B4D2-48F4-A04B-7F01358CE102` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhUploadPoints` | Upload Points GPU | `E3F9210B-0831-4F42-999D-42013AC1D101` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhUploadVectors` | Upload Vectors GPU | `77626F74-B831-4F42-999D-E3F9210B2120` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhPreviewDiagnostics` | Preview Diagnostics GPU | `99AA8FDC-EAF4-482B-89E9-183F25A0F119` | Custom Fallback | — | Original Custom GPU Icon | — |
| `GhSessionSettings` | Session Settings GPU | `340D18FE-7F4F-4DEE-B587-C1D7E38A3013` | Custom Fallback | — | Original Custom GPU Icon | — |
