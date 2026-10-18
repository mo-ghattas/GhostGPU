using System;
using GhostGPU.Core;
using ILGPU;
using ILGPU.Algorithms;

namespace GhostGPU.Runtime.Kernels
{
    internal static class MeshAnalysisKernels
    {
        private const double Epsilon = 1e-15;

        // =====================================================================
        // Pure Element Functions (Shared by CPU Host Path and Device Kernels)
        // =====================================================================

        public static V3 CrossProduct(V3 u, V3 v)
        {
            double x = Fp.Sub(Fp.Mul(u.Y, v.Z), Fp.Mul(u.Z, v.Y));
            double y = Fp.Sub(Fp.Mul(u.Z, v.X), Fp.Mul(u.X, v.Z));
            double z = Fp.Sub(Fp.Mul(u.X, v.Y), Fp.Mul(u.Y, v.X));
            return new V3(x, y, z);
        }

        public static void ComputeFaceGeometry(
            V3 a, V3 b, V3 c, V3 d, bool isQuad,
            out V3 center, out V3 normal, out double area, out V3 unnormalizedVectorArea)
        {
            if (!a.Finite || !b.Finite || !c.Finite || (isQuad && !d.Finite))
            {
                center = new V3(0, 0, 0);
                normal = new V3(0, 0, 0);
                area = 0.0;
                unnormalizedVectorArea = new V3(0, 0, 0);
                return;
            }

            if (isQuad)
            {
                // Quad centroid: average of all 4 vertices
                center = new V3(
                    (a.X + b.X + c.X + d.X) * 0.25,
                    (a.Y + b.Y + c.Y + d.Y) * 0.25,
                    (a.Z + b.Z + c.Z + d.Z) * 0.25);

                // Newell / diagonal normal: W = (C - A) x (D - B)
                // Invariant to diagonal selection and equal to boundary edge cross sum
                V3 diag1 = new V3(c.X - a.X, c.Y - a.Y, c.Z - a.Z);
                V3 diag2 = new V3(d.X - b.X, d.Y - b.Y, d.Z - b.Z);
                unnormalizedVectorArea = CrossProduct(diag1, diag2);

                double newellLen = VectorKernels.GetLengthHelper(
                    unnormalizedVectorArea.X, unnormalizedVectorArea.Y, unnormalizedVectorArea.Z);

                if (newellLen > 2.2250738585072014e-308 && VectorKernels.IsValidDouble(newellLen))
                {
                    normal = new V3(
                        unnormalizedVectorArea.X / newellLen,
                        unnormalizedVectorArea.Y / newellLen,
                        unnormalizedVectorArea.Z / newellLen);
                }
                else
                {
                    normal = new V3(0, 0, 0);
                }

                // Surface area of nonplanar quad: sum of triangles ABC + ACD (standard Rhino diagonal AC)
                V3 ab = new V3(b.X - a.X, b.Y - a.Y, b.Z - a.Z);
                V3 ac = new V3(c.X - a.X, c.Y - a.Y, c.Z - a.Z);
                V3 ad = new V3(d.X - a.X, d.Y - a.Y, d.Z - a.Z);

                V3 cross1 = CrossProduct(ab, ac);
                V3 cross2 = CrossProduct(ac, ad);

                double len1 = VectorKernels.GetLengthHelper(cross1.X, cross1.Y, cross1.Z);
                double len2 = VectorKernels.GetLengthHelper(cross2.X, cross2.Y, cross2.Z);

                area = 0.5 * (len1 + len2);
                if (!VectorKernels.IsValidDouble(area) || area < 0.0)
                {
                    area = 0.0;
                }
            }
            else
            {
                // Triangle centroid
                center = new V3(
                    (a.X + b.X + c.X) / 3.0,
                    (a.Y + b.Y + c.Y) / 3.0,
                    (a.Z + b.Z + c.Z) / 3.0);

                V3 ab = new V3(b.X - a.X, b.Y - a.Y, b.Z - a.Z);
                V3 ac = new V3(c.X - a.X, c.Y - a.Y, c.Z - a.Z);
                unnormalizedVectorArea = CrossProduct(ab, ac);

                double len = VectorKernels.GetLengthHelper(
                    unnormalizedVectorArea.X, unnormalizedVectorArea.Y, unnormalizedVectorArea.Z);

                area = 0.5 * len;
                if (!VectorKernels.IsValidDouble(area) || area < 0.0)
                {
                    area = 0.0;
                }

                if (len > 2.2250738585072014e-308 && VectorKernels.IsValidDouble(len))
                {
                    normal = new V3(
                        unnormalizedVectorArea.X / len,
                        unnormalizedVectorArea.Y / len,
                        unnormalizedVectorArea.Z / len);
                }
                else
                {
                    normal = new V3(0, 0, 0);
                }
            }
        }

        public static V3 ScalePoint(V3 p, V3 center, V3 factor)
        {
            if (!p.Finite || !center.Finite || !factor.Finite)
                return new V3(0, 0, 0);

            double dx = p.X - center.X;
            double dy = p.Y - center.Y;
            double dz = p.Z - center.Z;

            return new V3(
                center.X + Fp.Mul(dx, factor.X),
                center.Y + Fp.Mul(dy, factor.Y),
                center.Z + Fp.Mul(dz, factor.Z));
        }

        public static V3 RotatePointAxis(V3 p, V3 origin, V3 unitAxis, double cosAngle, double sinAngle)
        {
            if (!p.Finite || !origin.Finite || !unitAxis.Finite)
                return new V3(0, 0, 0);

            // Vector from origin to point
            double vx = p.X - origin.X;
            double vy = p.Y - origin.Y;
            double vz = p.Z - origin.Z;

            // dot = U . v
            double dot = Fp.Mul(unitAxis.X, vx) + Fp.Mul(unitAxis.Y, vy) + Fp.Mul(unitAxis.Z, vz);

            // cross = U x v
            double cx = Fp.Sub(Fp.Mul(unitAxis.Y, vz), Fp.Mul(unitAxis.Z, vy));
            double cy = Fp.Sub(Fp.Mul(unitAxis.Z, vx), Fp.Mul(unitAxis.X, vz));
            double cz = Fp.Sub(Fp.Mul(unitAxis.X, vy), Fp.Mul(unitAxis.Y, vx));

            // Rodrigues: v_rot = v * cos + cross * sin + U * dot * (1 - cos)
            double oneMinusCos = 1.0 - cosAngle;
            double termScalar = dot * oneMinusCos;

            double rx = Fp.Mul(vx, cosAngle) + Fp.Mul(cx, sinAngle) + Fp.Mul(unitAxis.X, termScalar);
            double ry = Fp.Mul(vy, cosAngle) + Fp.Mul(cy, sinAngle) + Fp.Mul(unitAxis.Y, termScalar);
            double rz = Fp.Mul(vz, cosAngle) + Fp.Mul(cz, sinAngle) + Fp.Mul(unitAxis.Z, termScalar);

            return new V3(origin.X + rx, origin.Y + ry, origin.Z + rz);
        }

        public static V3 MirrorPointPlane(V3 p, V3 planeOrigin, V3 unitNormal)
        {
            if (!p.Finite || !planeOrigin.Finite || !unitNormal.Finite)
                return new V3(0, 0, 0);

            double dx = p.X - planeOrigin.X;
            double dy = p.Y - planeOrigin.Y;
            double dz = p.Z - planeOrigin.Z;

            double dist = Fp.Mul(dx, unitNormal.X) + Fp.Mul(dy, unitNormal.Y) + Fp.Mul(dz, unitNormal.Z);
            double doubleDist = 2.0 * dist;

            return new V3(
                p.X - Fp.Mul(doubleDist, unitNormal.X),
                p.Y - Fp.Mul(doubleDist, unitNormal.Y),
                p.Z - Fp.Mul(doubleDist, unitNormal.Z));
        }

        public static V3 AffineTransformPoint(
            V3 p,
            double m0, double m1, double m2, double m3,
            double m4, double m5, double m6, double m7,
            double m8, double m9, double m10, double m11)
        {
            if (!p.Finite) return new V3(0, 0, 0);

            double x = Fp.Mul(m0, p.X) + Fp.Mul(m1, p.Y) + Fp.Mul(m2, p.Z) + m3;
            double y = Fp.Mul(m4, p.X) + Fp.Mul(m5, p.Y) + Fp.Mul(m6, p.Z) + m7;
            double z = Fp.Mul(m8, p.X) + Fp.Mul(m9, p.Y) + Fp.Mul(m10, p.Z) + m11;
            return new V3(x, y, z);
        }

        // =====================================================================
        // ILGPU Device Kernels
        // =====================================================================

        /// <summary>
        /// Analyzes faces in parallel on GPU: calculates center, Newell normal, and triangulated surface area.
        /// </summary>
        public static void FaceAnalysisKernel(
            Index1D index,
            ArrayView<V3> vertices,
            int vertexCount,
            ArrayView<int> faceIndices, // 4 ints per face: A, B, C, D
            int faceCount,
            ArrayView<V3> outCenters,
            ArrayView<V3> outNormals,
            ArrayView<double> outAreas)
        {
            int f = index;
            if (f >= faceCount) return;

            int baseIdx = f * 4;
            int aIdx = faceIndices[baseIdx];
            int bIdx = faceIndices[baseIdx + 1];
            int cIdx = faceIndices[baseIdx + 2];
            int dIdx = faceIndices[baseIdx + 3];

            // Guard against illegal out-of-range indexing
            if (aIdx < 0 || aIdx >= vertexCount ||
                bIdx < 0 || bIdx >= vertexCount ||
                cIdx < 0 || cIdx >= vertexCount ||
                dIdx < 0 || dIdx >= vertexCount)
            {
                outCenters[f] = new V3(0, 0, 0);
                outNormals[f] = new V3(0, 0, 0);
                outAreas[f] = 0.0;
                return;
            }

            V3 a = vertices[aIdx];
            V3 b = vertices[bIdx];
            V3 c = vertices[cIdx];
            V3 d = vertices[dIdx];
            bool isQuad = (cIdx != dIdx);

            ComputeFaceGeometry(a, b, c, d, isQuad, out V3 center, out V3 normal, out double area, out _);

            outCenters[f] = center;
            outNormals[f] = normal;
            outAreas[f] = area;
        }

        /// <summary>
        /// Computes vertex normals in parallel using precomputed CSR vertex-to-face adjacency.
        /// One thread per vertex, completely race-free without atomic contention.
        /// </summary>
        public static void VertexNormalsCSRKernel(
            Index1D index,
            ArrayView<V3> vertices,
            int vertexCount,
            ArrayView<int> faceIndices,
            int faceCount,
            ArrayView<int> vertexOffsets,  // length vertexCount + 1
            ArrayView<int> adjacentFaces,  // length totalIncidences
            int totalIncidences,
            byte weighting,               // 0: Unweighted, 1: AreaWeighted
            ArrayView<V3> outNormals)
        {
            int v = index;
            if (v >= vertexCount) return;

            int start = vertexOffsets[v];
            int end = vertexOffsets[v + 1];

            if (start < 0 || end > totalIncidences || start > end)
            {
                outNormals[v] = new V3(0, 0, 0);
                return;
            }

            double sumX = 0.0;
            double sumY = 0.0;
            double sumZ = 0.0;

            for (int k = start; k < end; k++)
            {
                int f = adjacentFaces[k];
                if (f < 0 || f >= faceCount) continue;

                int baseIdx = f * 4;
                int aIdx = faceIndices[baseIdx];
                int bIdx = faceIndices[baseIdx + 1];
                int cIdx = faceIndices[baseIdx + 2];
                int dIdx = faceIndices[baseIdx + 3];

                if (aIdx < 0 || aIdx >= vertexCount ||
                    bIdx < 0 || bIdx >= vertexCount ||
                    cIdx < 0 || cIdx >= vertexCount ||
                    dIdx < 0 || dIdx >= vertexCount)
                    continue;

                V3 a = vertices[aIdx];
                V3 b = vertices[bIdx];
                V3 c = vertices[cIdx];
                V3 d = vertices[dIdx];
                bool isQuad = (cIdx != dIdx);

                ComputeFaceGeometry(a, b, c, d, isQuad, out _, out V3 normal, out _, out V3 unnormCross);

                if (weighting == 1) // AreaWeighted
                {
                    sumX += unnormCross.X;
                    sumY += unnormCross.Y;
                    sumZ += unnormCross.Z;
                }
                else // Unweighted
                {
                    sumX += normal.X;
                    sumY += normal.Y;
                    sumZ += normal.Z;
                }
            }

            double len = VectorKernels.GetLengthHelper(sumX, sumY, sumZ);
            if (len > 2.2250738585072014e-308 && VectorKernels.IsValidDouble(len))
            {
                outNormals[v] = new V3(sumX / len, sumY / len, sumZ / len);
            }
            else
            {
                outNormals[v] = new V3(0, 0, 0);
            }
        }

        /// <summary>
        /// Scales vertices in parallel around center points with uniform or 3D scaling factors.
        /// </summary>
        public static void MeshScaleKernel(
            Index1D index,
            ArrayView<V3> points,
            ArrayView<V3> centers,
            ArrayView<V3> factors,
            int pCount,
            int cCount,
            int fCount,
            ArrayView<V3> output)
        {
            int i = index;
            int idxP = i < pCount ? i : pCount - 1;
            int idxC = i < cCount ? i : cCount - 1;
            int idxF = i < fCount ? i : fCount - 1;

            output[i] = ScalePoint(points[idxP], centers[idxC], factors[idxF]);
        }

        /// <summary>
        /// Rotates vertices in parallel around an arbitrary 3D axis.
        /// </summary>
        public static void MeshRotateKernel(
            Index1D index,
            ArrayView<V3> points,
            V3 axisOrigin,
            V3 unitAxisDir,
            double cosAngle,
            double sinAngle,
            int pCount,
            ArrayView<V3> output)
        {
            int i = index;
            if (i >= pCount) return;

            output[i] = RotatePointAxis(points[i], axisOrigin, unitAxisDir, cosAngle, sinAngle);
        }

        /// <summary>
        /// Mirrors vertices in parallel across an arbitrary 3D plane.
        /// </summary>
        public static void MeshMirrorKernel(
            Index1D index,
            ArrayView<V3> points,
            V3 planeOrigin,
            V3 unitNormal,
            int pCount,
            ArrayView<V3> output)
        {
            int i = index;
            if (i >= pCount) return;

            output[i] = MirrorPointPlane(points[i], planeOrigin, unitNormal);
        }

        /// <summary>
        /// Applies an affine 4x4 matrix transformation in parallel to resident vertices.
        /// Matrix layout: m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 (row-major).
        /// </summary>
        public static void MeshAffineTransformKernel(
            Index1D index,
            ArrayView<V3> points,
            ArrayView<double> matrix16,
            int pCount,
            ArrayView<V3> output)
        {
            int i = index;
            if (i >= pCount) return;

            output[i] = AffineTransformPoint(
                points[i],
                matrix16[0], matrix16[1], matrix16[2], matrix16[3],
                matrix16[4], matrix16[5], matrix16[6], matrix16[7],
                matrix16[8], matrix16[9], matrix16[10], matrix16[11]);
        }

        public static void WidenU16Kernel(
            Index1D index,
            ArrayView<ushort> input,
            ArrayView<int> output)
        {
            int i = index;
            output[i] = (int)input[i];
        }

        public static void ValidateIndicesBoundsKernel(
            Index1D index,
            ArrayView<int> indices,
            int count,
            int maxExclusive,
            ArrayView<int> outInvalidFlag)
        {
            int i = index;
            if (i >= count) return;
            int val = indices[i];
            if (val < 0 || val >= maxExclusive)
            {
                outInvalidFlag[0] = 1;
            }
        }

        public static void ValidateCsrOffsetsKernel(
            Index1D index,
            ArrayView<int> offsets,
            int vertexCount,
            int expectedTotalIncidences,
            ArrayView<int> outInvalidFlag)
        {
            int i = index;
            if (i > vertexCount) return;

            if (i == 0 && offsets[0] != 0)
            {
                outInvalidFlag[0] = 1;
                return;
            }

            if (i == vertexCount && offsets[vertexCount] != expectedTotalIncidences)
            {
                outInvalidFlag[0] = 1;
                return;
            }

            if (i < vertexCount)
            {
                int start = offsets[i];
                int end = offsets[i + 1];
                if (start < 0 || end > expectedTotalIncidences || start > end)
                {
                    outInvalidFlag[0] = 1;
                }
            }
        }
    }
}
