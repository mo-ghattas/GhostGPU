using System;
using System.Collections.Generic;

namespace GhostGPU.Grasshopper.Icons
{
    /// <summary>
    /// Metadata mapping describing the relationship between a GhostGPU component and its
    /// native Grasshopper counterpart or fallback generation strategy.
    /// </summary>
    public sealed class NativeMapping
    {
        public Guid GhostGpuGuid { get; }
        public string GhostGpuClassName { get; }
        public Guid NativeGuid { get; }
        public string NativeTypeName { get; }
        public string NativeAssemblyName { get; }
        public string FallbackLabel { get; }

        public bool HasNativeCounterpart => NativeGuid != Guid.Empty && !string.IsNullOrEmpty(NativeTypeName);

        public NativeMapping(
            Guid ghostGpuGuid,
            string ghostGpuClassName,
            Guid nativeGuid,
            string nativeTypeName,
            string nativeAssemblyName,
            string fallbackLabel)
        {
            GhostGpuGuid = ghostGpuGuid;
            GhostGpuClassName = ghostGpuClassName;
            NativeGuid = nativeGuid;
            NativeTypeName = nativeTypeName;
            NativeAssemblyName = nativeAssemblyName;
            FallbackLabel = fallbackLabel;
        }

        public static NativeMapping CreateNative(
            Guid ghostGpuGuid,
            string ghostGpuClassName,
            Guid nativeGuid,
            string nativeTypeName,
            string nativeAssemblyName,
            string fallbackLabel = null)
        {
            return new NativeMapping(ghostGpuGuid, ghostGpuClassName, nativeGuid, nativeTypeName, nativeAssemblyName, fallbackLabel);
        }

        public static NativeMapping CreateFallback(
            Guid ghostGpuGuid,
            string ghostGpuClassName,
            string fallbackLabel)
        {
            return new NativeMapping(ghostGpuGuid, ghostGpuClassName, Guid.Empty, null, null, fallbackLabel);
        }
    }

    /// <summary>
    /// Authoritative catalog mapping GhostGPU components by GUID or class name to their exact
    /// native Grasshopper 8 counterpart (from artifacts/phase1/native-components.json)
    /// or fallback generator label.
    /// </summary>
    public static class NativeIconCatalog
    {
        private static readonly Dictionary<Guid, NativeMapping> _byGuid;
        private static readonly Dictionary<string, NativeMapping> _byName;

        static NativeIconCatalog()
        {
            _byGuid = new Dictionary<Guid, NativeMapping>();
            _byName = new Dictionary<string, NativeMapping>(StringComparer.OrdinalIgnoreCase);

            RegisterAll();
        }

        private static void Register(NativeMapping mapping)
        {
            if (mapping.GhostGpuGuid != Guid.Empty)
            {
                _byGuid[mapping.GhostGpuGuid] = mapping;
            }
            if (!string.IsNullOrEmpty(mapping.GhostGpuClassName))
            {
                _byName[mapping.GhostGpuClassName] = mapping;
            }
        }

        public static bool TryGetMapping(Guid ghostGpuGuid, out NativeMapping mapping)
        {
            return _byGuid.TryGetValue(ghostGpuGuid, out mapping);
        }

        public static bool TryGetMapping(string classNameOrType, out NativeMapping mapping)
        {
            if (string.IsNullOrEmpty(classNameOrType))
            {
                mapping = null;
                return false;
            }

            int lastDot = classNameOrType.LastIndexOf('.');
            string shortName = lastDot >= 0 ? classNameOrType.Substring(lastDot + 1) : classNameOrType;

            return _byName.TryGetValue(shortName, out mapping) || _byName.TryGetValue(classNameOrType, out mapping);
        }

        private static void RegisterAll()
        {
            // =========================================================================
            // A. Point & Vector Expansion Components (25 Components from pointvector)
            // =========================================================================

            // 1. Construct Point GPU
            Register(NativeMapping.CreateNative(
                new Guid("6B0A8211-7F45-4B9D-9457-302E1528BA01"), "GhConstructPoint",
                new Guid("3581f42a-9592-4549-bd6b-1c0fc39d067b"),
                "VectorComponents.PointComponents.Component_ConstructPoint",
                "VectorComponents.gha", "Pt"));

            // 2. Deconstruct Point GPU
            Register(NativeMapping.CreateNative(
                new Guid("E623A849-0DF1-4835-B4C2-930B48002712"), "GhDeconstructPoint",
                new Guid("9abae6b7-fa1d-448c-9209-4a8155345841"),
                "VectorComponents.PointComponents.Component_DeconstructPoint",
                "VectorComponents.gha", "DePt"));

            // 3. Construct Vector GPU
            Register(NativeMapping.CreateNative(
                new Guid("9C465228-44A5-4BC5-88AA-E2E109605333"), "GhConstructVector",
                new Guid("56b92eab-d121-43f7-94d3-6cd8f0ddead8"),
                "VectorComponents.VectorComponents.Component_ConstructVector",
                "VectorComponents.gha", "Vec"));

            // 4. Deconstruct Vector GPU
            Register(NativeMapping.CreateNative(
                new Guid("79C1D067-EB8D-476E-8BDD-8820B1333B14"), "GhDeconstructVector",
                new Guid("a50fcd4a-cf42-4c3f-8616-022761e6cc93"),
                "VectorComponents.VectorComponents.Component_DeconstructVector",
                "VectorComponents.gha", "DeVec"));

            // 5. Cylindrical Point GPU
            Register(NativeMapping.CreateNative(
                new Guid("54B27DF9-635E-4B02-B792-506B8F8A2825"), "GhCylindricalPoint",
                new Guid("23603075-be64-4d86-9294-c3c125a12104"),
                "VectorComponents.PointComponents.Component_PointCylindrical",
                "VectorComponents.gha", "CylPt"));

            // 6. Spherical Point GPU (Custom original fallback icon)
            Register(NativeMapping.CreateFallback(
                new Guid("19CD16A7-7D9A-4BA4-8A87-A1C9369C9736"), "GhSphericalPoint", "SPH"));

            // 7. Vector Add GPU
            Register(NativeMapping.CreateNative(
                new Guid("2867F5B1-8C56-424D-B5DF-C51B6BE48C47"), "GhVectorAdd",
                new Guid("fb012ef9-4734-4049-84a0-b92b85bb09da"),
                "VectorComponents.VectorComponents.Component_VectorSum",
                "VectorComponents.gha", "VAdd"));

            // 8. Vector Subtract GPU (arithmetic Subtraction counterpart)
            Register(NativeMapping.CreateNative(
                new Guid("86D5A250-9852-4740-97FE-4458EE858658"), "GhVectorSubtract",
                new Guid("9c007a04-d0d9-48e4-9da3-9ba142bc4d46"),
                "MathComponents.OperatorComponents.Component_VariableSubtraction",
                "MathComponents.gha", "VSub"));

            // 9. Vector Reverse GPU
            Register(NativeMapping.CreateNative(
                new Guid("342BA8B5-9BE2-4363-9562-F2E515286169"), "GhVectorReverse",
                new Guid("d5788074-d75d-4021-b1a3-0bf992928584"),
                "VectorComponents.VectorComponents.Component_VectorReverse",
                "VectorComponents.gha", "VRev"));

            // 10. Vector Length GPU
            Register(NativeMapping.CreateNative(
                new Guid("4ACFF388-1647-493B-B051-CA18FE5D627A"), "GhVectorLength",
                new Guid("675e31bf-1775-48d7-bb8d-76b77786dd53"),
                "VectorComponents.VectorComponents.Component_VectorLength",
                "VectorComponents.gha", "VLen"));

            // 11. Vector Scale GPU
            Register(NativeMapping.CreateNative(
                new Guid("88939BC8-F648-433C-8472-E7A4A28BE28B"), "GhVectorScale",
                new Guid("63fff845-7c61-4dfb-ba12-44d481b4bf0f"),
                "VectorComponents.VectorComponents.Component_VectorMultiply",
                "VectorComponents.gha", "VScl"));

            // 12. Vector Divide GPU
            Register(NativeMapping.CreateNative(
                new Guid("3D573DF1-0985-4D3B-A816-B0E329437FE5"), "GhVectorDivide",
                new Guid("310e1065-d03a-4858-bcd1-809d39c042af"),
                "VectorComponents.VectorComponents.Component_VectorDivide",
                "VectorComponents.gha", "VDiv"));

            // 13. Vector Dot Product GPU
            Register(NativeMapping.CreateNative(
                new Guid("3EE0929F-49C3-4886-A060-61A601E6959C"), "GhVectorDot",
                new Guid("43b9ea8f-f772-40f2-9880-011a9c3cbbb0"),
                "VectorComponents.VectorComponents.Component_VectorDotProduct",
                "VectorComponents.gha", "Dot"));

            // 14. Vector Cross Product GPU
            Register(NativeMapping.CreateNative(
                new Guid("F91DC962-DF50-4F5A-8A46-BE0D9E9D75AD"), "GhVectorCross",
                new Guid("2a5cfb31-028a-4b34-b4e1-9b20ae15312e"),
                "VectorComponents.VectorComponents.Component_VectorCrossProduct",
                "VectorComponents.gha", "Cross"));

            // 15. Vector Angle GPU (non-obsolete Angle component)
            Register(NativeMapping.CreateNative(
                new Guid("ACCE1DF2-8B18-4EE4-B974-9C7802D33BBE"), "GhVectorAngle",
                new Guid("b464fccb-50e7-41bd-9789-8438db9bea9f"),
                "VectorComponents.VectorComponents.Component_VectorAngle",
                "VectorComponents.gha", "Angle"));

            // 16. Unit X GPU
            Register(NativeMapping.CreateNative(
                new Guid("17D01FA3-75AC-44C1-8B39-86B5A10A2B11"), "GhUnitX",
                new Guid("79f9fbb3-8f1d-4d9a-88a9-f7961b1012cd"),
                "VectorComponents.VectorComponents.Component_UnitVectorX",
                "VectorComponents.gha", "UnitX"));

            // 17. Unit Y GPU
            Register(NativeMapping.CreateNative(
                new Guid("8B5D4E12-F559-4D33-A2A9-3A7F11C8E522"), "GhUnitY",
                new Guid("d3d195ea-2d59-4ffa-90b1-8b7ff3369f69"),
                "VectorComponents.VectorComponents.Component_UnitVectorY",
                "VectorComponents.gha", "UnitY"));

            // 18. Unit Z GPU
            Register(NativeMapping.CreateNative(
                new Guid("4E347A22-3168-45F4-9844-428CE2B7B333"), "GhUnitZ",
                new Guid("9103c240-a6a9-4223-9b42-dbd19bf38e2b"),
                "VectorComponents.VectorComponents.Component_UnitVectorZ",
                "VectorComponents.gha", "UnitZ"));

            // 19. Point Interpolate GPU (Custom original fallback icon)
            Register(NativeMapping.CreateFallback(
                new Guid("B7E84B05-C02A-4D39-A1B2-7935216EDECF"), "GhPointInterpolate", "INTP"));

            // 20. Project Point GPU
            Register(NativeMapping.CreateNative(
                new Guid("91B33B88-C488-46E1-8BD5-5EF092A7A444"), "GhPointProject",
                new Guid("5184b8cb-b71e-4def-a590-cd2c9bc58906"),
                "TriangulationComponents.Component_ProjectPoint",
                "TriangulationComponents.gha", "ProjPt"));

            // 21. Project Vector GPU (non-obsolete Project)
            Register(NativeMapping.CreateNative(
                new Guid("C4F02DB5-F329-4112-BF9B-A5C9B39B2AE0"), "GhProjectVectorToPlane",
                new Guid("23285717-156c-468f-a691-b242488c06a6"),
                "LinearXForms.Component_Project",
                "XformComponents.gha", "ProjV"));

            // 22. Rotate Vector GPU
            Register(NativeMapping.CreateNative(
                new Guid("6986C8CE-2E24-42DE-8C56-DA15697669F1"), "GhRotateVectorAxis",
                new Guid("b6d7ba20-cf74-4191-a756-2216a36e30a7"),
                "VectorComponents.VectorComponents.Component_VectorRotate",
                "VectorComponents.gha", "RotV"));

            // 23. Scale Point GPU (non-obsolete Scale)
            Register(NativeMapping.CreateNative(
                new Guid("11E1BF3E-C246-4E27-B6DF-D502B0F4E002"), "GhPointScale",
                new Guid("4d2a06bd-4b0f-4c65-9ee0-4220e4c01703"),
                "LinearXForms.Component_Scale",
                "XformComponents.gha", "SclPt"));

            // 24. Rotate Point GPU (non-obsolete Rotate Axis)
            Register(NativeMapping.CreateNative(
                new Guid("77FE7C7A-A1BA-4AE1-931B-EFEA420FE913"), "GhPointRotateAxis",
                new Guid("3ac8e589-37f5-477d-aa61-6699702c5728"),
                "LinearXForms.Component_RotateAxis",
                "XformComponents.gha", "RotPt"));

            // 25. Mirror Point GPU (non-obsolete Mirror)
            Register(NativeMapping.CreateNative(
                new Guid("EBEFF8DF-7C0F-4D2A-B9C0-1C6DE7CFFD24"), "GhPointMirror",
                new Guid("f12daa2f-4fd5-48c1-8ac3-5dea476912ca"),
                "LinearXForms.Component_Mirror",
                "XformComponents.gha", "MirPt"));

            // =========================================================================
            // B. Fixed-Topology Mesh Components (9 Components from meshes)
            // =========================================================================

            // 1. Face Normals GPU
            Register(NativeMapping.CreateNative(
                new Guid("A194B101-7E23-4211-9A31-92F6F03F8001"), "GhFaceNormals",
                new Guid("cb4ca22c-3419-4962-a078-ad4ff7f1f929"),
                "SurfaceComponents.MeshComponents.Component_MeshFaceNormals",
                "SurfaceComponents.gha", "FNorm"));

            // 2. Face Areas GPU (Custom Analysis)
            Register(NativeMapping.CreateFallback(
                new Guid("A194B102-7E23-4211-9A31-92F6F03F8002"), "GhFaceAreas", "FA"));

            // 3. Vertex Normals GPU (Custom Analysis)
            Register(NativeMapping.CreateFallback(
                new Guid("A194B103-7E23-4211-9A31-92F6F03F8003"), "GhVertexNormals", "VN"));

            // 4. Deconstruct Mesh GPU
            Register(NativeMapping.CreateNative(
                new Guid("A194B104-7E23-4211-9A31-92F6F03F8004"), "GhDeconstructMesh",
                new Guid("ba2d8f57-0738-42b4-b5a5-fe4d853517eb"),
                "SurfaceComponents.MeshComponents.Component_DeconstructMesh",
                "SurfaceComponents.gha", "DeMesh"));

            // 5. Mesh Translate GPU (non-obsolete Move)
            Register(NativeMapping.CreateNative(
                new Guid("A194B105-7E23-4211-9A31-92F6F03F8005"), "GhMeshTranslate",
                new Guid("e9eb1dcf-92f6-4d4d-84ae-96222d60f56b"),
                "LinearXForms.Component_Move",
                "XformComponents.gha", "MTrans"));

            // 6. Mesh Scale GPU (non-obsolete Scale)
            Register(NativeMapping.CreateNative(
                new Guid("A194B106-7E23-4211-9A31-92F6F03F8006"), "GhMeshScale",
                new Guid("4d2a06bd-4b0f-4c65-9ee0-4220e4c01703"),
                "LinearXForms.Component_Scale",
                "XformComponents.gha", "MScale"));

            // 7. Mesh Rotate GPU (non-obsolete Rotate Axis)
            Register(NativeMapping.CreateNative(
                new Guid("A194B107-7E23-4211-9A31-92F6F03F8007"), "GhMeshRotate",
                new Guid("3ac8e589-37f5-477d-aa61-6699702c5728"),
                "LinearXForms.Component_RotateAxis",
                "XformComponents.gha", "MRot"));

            // 8. Mesh Mirror GPU (non-obsolete Mirror)
            Register(NativeMapping.CreateNative(
                new Guid("A194B108-7E23-4211-9A31-92F6F03F8008"), "GhMeshMirror",
                new Guid("f12daa2f-4fd5-48c1-8ac3-5dea476912ca"),
                "LinearXForms.Component_Mirror",
                "XformComponents.gha", "MMirr"));

            // 9. Mesh Transform GPU (non-obsolete Transform)
            Register(NativeMapping.CreateNative(
                new Guid("A194B109-7E23-4211-9A31-92F6F03F8009"), "GhMeshTransform",
                new Guid("610e689b-5adc-47b3-af8f-e3a32b7ea341"),
                "LinearXForms.Component_ApplyTransform",
                "XformComponents.gha", "MXform"));

            // =========================================================================
            // C. Existing Foundation Components
            // =========================================================================

            // Bounds GPU
            Register(NativeMapping.CreateNative(
                new Guid("318C7E77-FB9D-40C6-9EE3-EF0F5EE73112"), "GhBounds",
                new Guid("f44b92b0-3b5b-493a-86f4-fd7408c3daf3"),
                "MathComponents.DomainComponents.Component_NumericBounds",
                "MathComponents.gha", "Bnd"));

            // Colour Ramp GPU
            Register(NativeMapping.CreateNative(
                new Guid("B687C804-A759-4E6D-B27F-7B27C1748117"), "GhColourRamp",
                new Guid("298e0adf-625a-4207-9d0f-c22212480c4c"),
                "IOComponents.Rhinoceros.Display.Component_DisplayColorGradient",
                "IOComponents.gha", "Ramp"));

            // Map GPU (non-obsolete Remap Numbers)
            Register(NativeMapping.CreateNative(
                new Guid("1251E14F-C00D-477D-9730-C7BAAC88F114"), "GhMap",
                new Guid("2fcc2743-8339-4cdf-a046-a1f17439191d"),
                "MathComponents.DomainComponents.Component_RemapNumbers",
                "MathComponents.gha", "Map"));

            // Remap GPU (non-obsolete Remap Numbers)
            Register(NativeMapping.CreateNative(
                new Guid("68BD017C-BE9C-4CD5-827B-CE74F34A8113"), "GhRemap",
                new Guid("2fcc2743-8339-4cdf-a046-a1f17439191d"),
                "MathComponents.DomainComponents.Component_RemapNumbers",
                "MathComponents.gha", "Remap"));

            // Scalar Math GPU (Custom math operations node)
            Register(NativeMapping.CreateFallback(
                new Guid("8688D0F1-5246-4B89-AE94-4F81BF6AA111"), "GhScalarMath", "MATH"));

            // Mesh Colours GPU
            Register(NativeMapping.CreateNative(
                new Guid("9846E062-8BC4-4C5D-913C-5586617A2116"), "GhMeshColours",
                new Guid("d2cedf38-1149-4adc-8dbf-b06571cb5106"),
                "SurfaceComponents.MeshComponents.Component_MeshColour",
                "SurfaceComponents.gha", "MCol"));

            // Mesh Displace GPU (Custom displacement modifier)
            Register(NativeMapping.CreateFallback(
                new Guid("8DA3A7BD-BC84-48FE-8B4F-A16DC19A6115"), "GhMeshDisplace", "DISP"));

            // Mesh Vertices GPU
            Register(NativeMapping.CreateNative(
                new Guid("24BC97D1-85A3-4DC1-8314-2C6F158BA104"), "GhMeshVertices",
                new Guid("ba2d8f57-0738-42b4-b5a5-fe4d853517eb"),
                "SurfaceComponents.MeshComponents.Component_DeconstructMesh",
                "SurfaceComponents.gha", "MVert"));

            // Amplitude GPU
            Register(NativeMapping.CreateNative(
                new Guid("23000673-E2E1-4FFD-8B42-4D907CFB2109"), "GhAmplitude",
                new Guid("6ec39468-dae7-4ffa-a766-f2ab22a2c62e"),
                "VectorComponents.VectorComponents.Component_VectorAmplitude",
                "VectorComponents.gha", "Amp"));

            // Closest Point GPU
            Register(NativeMapping.CreateNative(
                new Guid("71708AC6-B7E0-4BA8-868C-D9B97F7A6105"), "GhClosestPoint",
                new Guid("571ca323-6e55-425a-bf9e-ee103c7ba4b9"),
                "VectorComponents.PointComponents.Component_ClosestPoint",
                "VectorComponents.gha", "CP"));

            // Distance GPU
            Register(NativeMapping.CreateNative(
                new Guid("F0D9BC05-3948-4394-B6AA-EF78A9C22106"), "GhDistance",
                new Guid("93b8e93d-f932-402c-b435-84be04d87666"),
                "VectorComponents.PointComponents.Component_Distance",
                "VectorComponents.gha", "Dist"));

            // Move GPU (non-obsolete Move)
            Register(NativeMapping.CreateNative(
                new Guid("0994B67E-B8B8-43F1-9BE4-332EB7AEB110"), "GhMove",
                new Guid("e9eb1dcf-92f6-4d4d-84ae-96222d60f56b"),
                "LinearXForms.Component_Move",
                "XformComponents.gha", "Move"));

            // Unit Vector GPU
            Register(NativeMapping.CreateNative(
                new Guid("C750798C-D3F1-4F2D-BCBF-DF06D8DFB108"), "GhUnitVector",
                new Guid("d2da1306-259a-4994-85a4-672d8a4c7805"),
                "VectorComponents.VectorComponents.Component_UnitVector",
                "VectorComponents.gha", "Unit"));

            // Vector 2Pt GPU
            Register(NativeMapping.CreateNative(
                new Guid("5E035987-9DE8-48D5-BC40-EBCBE0114107"), "GhVector2Pt",
                new Guid("934ede4a-924a-4973-bb05-0dc4b36fae75"),
                "VectorComponents.VectorComponents.Component_Vector2Pt",
                "VectorComponents.gha", "V2Pt"));

            // Mesh Difference GPU
            Register(NativeMapping.CreateNative(
                new Guid("50ED9CD5-C557-4741-8404-7603EF5E2B5F"), "GhMeshDifference",
                new Guid("4f3147f4-9fcd-4a7e-be0e-b1841caa5f97"),
                "SurfaceComponents.MeshComponents.Component_MeshBooleanDifference",
                "SurfaceComponents.gha", "MDif"));

            // Mesh Intersection GPU
            Register(NativeMapping.CreateNative(
                new Guid("A1372AA7-6986-4982-A5B2-05069149B0E5"), "GhMeshIntersection",
                new Guid("95aef4f6-66fc-477e-b8f8-32395a837831"),
                "SurfaceComponents.MeshComponents.Component_MeshBooleanIntersection",
                "SurfaceComponents.gha", "MInt"));

            // Mesh Union GPU
            Register(NativeMapping.CreateNative(
                new Guid("9FD9BFE8-4ACF-472E-8DA4-076B55524059"), "GhMeshUnion",
                new Guid("88060a82-0bf7-46bb-9af8-bdc860cf7e1d"),
                "SurfaceComponents.MeshComponents.Component_MeshBooleanUnion",
                "SurfaceComponents.gha", "MUni"));

            // Solid Difference GPU
            Register(NativeMapping.CreateNative(
                new Guid("2F5D629E-2468-4DC3-83BE-FD4693A5796B"), "GhSolidDifference",
                new Guid("fab11c30-2d9c-4d15-ab3c-2289f1ae5c21"),
                "SurfaceComponents.SolidComponents.Component_BooleanDifference",
                "SurfaceComponents.gha", "SDif"));

            // Solid Intersection GPU
            Register(NativeMapping.CreateNative(
                new Guid("16E286EE-2659-41B0-9625-E19A0E2F059F"), "GhSolidIntersection",
                new Guid("5723c845-cafc-442d-a667-8c76532845e6"),
                "SurfaceComponents.SolidComponents.Component_BooleanIntersection",
                "SurfaceComponents.gha", "SInt"));

            // Solid Union GPU
            Register(NativeMapping.CreateNative(
                new Guid("627AD9C1-0D3C-4BCC-BE6F-B8BBBB23C6A1"), "GhSolidUnion",
                new Guid("10434a15-da85-4281-bb64-a2b3a995b9c6"),
                "SurfaceComponents.SolidComponents.Component_BooleanUnion",
                "SurfaceComponents.gha", "SUni"));

            // Custom IO / Diagnostics / Settings
            Register(NativeMapping.CreateFallback(
                new Guid("35E00A63-958C-4FA9-A6CE-7B4D228DF118"), "GhDownload", "DN"));
            Register(NativeMapping.CreateFallback(
                new Guid("54B55E5D-47C0-463E-B19E-6C0D9B9DF103"), "GhUploadMesh", "UPM"));
            Register(NativeMapping.CreateFallback(
                new Guid("C475A952-B4D2-48F4-A04B-7F01358CE102"), "GhUploadNumbers", "UP#"));
            Register(NativeMapping.CreateFallback(
                new Guid("E3F9210B-0831-4F42-999D-42013AC1D101"), "GhUploadPoints", "UPP"));
            Register(NativeMapping.CreateFallback(
                new Guid("77626F74-B831-4F42-999D-E3F9210B2120"), "GhUploadVectors", "UPV"));
            Register(NativeMapping.CreateFallback(
                new Guid("99AA8FDC-EAF4-482B-89E9-183F25A0F119"), "GhPreviewDiagnostics", "DIAG"));
            Register(NativeMapping.CreateFallback(
                new Guid("340D18FE-7F4F-4DEE-B587-C1D7E38A3013"), "GhSessionSettings", "CONF"));
        }
    }
}
