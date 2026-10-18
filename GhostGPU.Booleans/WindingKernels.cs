using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using GhostGPU.Core;
using ILGPU;
using ILGPU.Algorithms;

namespace GhostGPU.Booleans
{
    /// <summary>
    /// BVH node for fast winding numbers (Barill et al. 2018). Internal nodes carry the Taylor expansion of
    /// the solid angle about the area-weighted centroid <see cref="Centroid"/>, with d = x - Centroid over each
    /// triangle and every moment integrated exactly over the triangle:
    /// order 0 <see cref="NormalArea"/> = sum a n; order 1 rows M0..M2 = sum a n d_c^T (d_c = triangle centroid offset);
    /// order 2 from T_ijk = sum a n_i &lt;d_j d_k&gt;: <see cref="P"/> = T contracted (j,i), <see cref="U"/> = T contracted (j,k),
    /// and the fully symmetrised cubic form coefficients C (xxx,yyy,zzz | xxy,xxz,xyy | yyz,xzz,yzz | xyz).
    /// </summary>
    public struct GpuBvhNode
    {
        public V3 Centroid;
        public V3 NormalArea;
        public V3 M0;
        public V3 M1;
        public V3 M2;
        public V3 P;
        public V3 U;
        public V3 C0;
        public V3 C1;
        public V3 C2;
        public double Cxyz;
        public double RadiusSq;
        public int LeftChild;
        public int RightChild;
        public int TriStart;
        public int TriCount;
    }

    public static class WindingKernels
    {
        public const double InvFourPi = 0.07957747154594766788; // 1 / (4 pi)
        public const int BvhStackSize = 64;

        /// <summary>Signed solid angle of triangle (a,b,c) seen from p (Van Oosterom &amp; Strackee).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double TriangleSolidAngle(V3 p, V3 a, V3 b, V3 c)
        {
            double ax = a.X - p.X, ay = a.Y - p.Y, az = a.Z - p.Z;
            double bx = b.X - p.X, by = b.Y - p.Y, bz = b.Z - p.Z;
            double cx = c.X - p.X, cy = c.Y - p.Y, cz = c.Z - p.Z;

            double la = XMath.Sqrt(ax * ax + ay * ay + az * az);
            double lb = XMath.Sqrt(bx * bx + by * by + bz * bz);
            double lc = XMath.Sqrt(cx * cx + cy * cy + cz * cz);
            if (la < 1e-300 || lb < 1e-300 || lc < 1e-300)
                return 0.0;

            double num = ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
            double den = la * lb * lc + (ax * bx + ay * by + az * bz) * lc
                       + (ax * cx + ay * cy + az * cz) * lb + (bx * cx + by * cy + bz * cz) * la;
            return 2.0 * XMath.Atan2(num, den);
        }

        /// <summary>
        /// Far-field solid angle of a node seen from a query, r = node.Centroid - query, from the Taylor expansion of
        /// f(y) = y/|y|^3 at y = r (solid angle = sum over x of a n . f(x - query)).
        /// Order 0: N.r/|r|^3. Order 1: tr(M)/|r|^3 - 3 r^T M r/|r|^5.
        /// Order 2: (1/2) T_ijk d_j d_k f_i = -1.5 (2 P.r + U.r)/|r|^5 + 7.5 C(r)/|r|^7.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double FarFieldSolidAngle(GpuBvhNode n, double rx, double ry, double rz, double distSq)
        {
            double inv3 = 1.0 / (distSq * XMath.Sqrt(distSq));
            double inv5 = inv3 / distSq;
            double order0 = (n.NormalArea.X * rx + n.NormalArea.Y * ry + n.NormalArea.Z * rz) * inv3;
            double trace = n.M0.X + n.M1.Y + n.M2.Z;
            double rMr = rx * (n.M0.X * rx + n.M0.Y * ry + n.M0.Z * rz)
                       + ry * (n.M1.X * rx + n.M1.Y * ry + n.M1.Z * rz)
                       + rz * (n.M2.X * rx + n.M2.Y * ry + n.M2.Z * rz);
            double inv7 = inv5 / distSq;
            double cubic = n.C0.X * rx * rx * rx + n.C0.Y * ry * ry * ry + n.C0.Z * rz * rz * rz
                         + n.C1.X * rx * rx * ry + n.C1.Y * rx * rx * rz + n.C1.Z * rx * ry * ry
                         + n.C2.X * ry * ry * rz + n.C2.Y * rx * rz * rz + n.C2.Z * ry * rz * rz
                         + n.Cxyz * rx * ry * rz;
            double pr = 2.0 * (n.P.X * rx + n.P.Y * ry + n.P.Z * rz) + (n.U.X * rx + n.U.Y * ry + n.U.Z * rz);
            double order2 = -1.5 * pr * inv5 + 7.5 * cubic * inv7;
            return order0 + trace * inv3 - 3.0 * rMr * inv5 + order2;
        }

        public static void BruteWindingNumberKernelI32(
            Index1D index, ArrayView<V3> points, ArrayView<V3> verts, ArrayView<int> tris, int triCount, ArrayView<double> windingOut)
        {
            V3 p = points[index];
            double sum = 0.0;
            for (int t = 0; t < triCount; t++)
                sum += TriangleSolidAngle(p, verts[tris[3 * t]], verts[tris[3 * t + 1]], verts[tris[3 * t + 2]]);
            windingOut[index] = sum * InvFourPi;
        }

        public static void BruteWindingNumberKernelU16(
            Index1D index, ArrayView<V3> points, ArrayView<V3> verts, ArrayView<ushort> tris, int triCount, ArrayView<double> windingOut)
        {
            V3 p = points[index];
            double sum = 0.0;
            for (int t = 0; t < triCount; t++)
                sum += TriangleSolidAngle(p, verts[(int)tris[3 * t]], verts[(int)tris[3 * t + 1]], verts[(int)tris[3 * t + 2]]);
            windingOut[index] = sum * InvFourPi;
        }

        /// <summary>
        /// BVH-clustered fast winding number. Leaves are exact; internal nodes use the order-2 far field
        /// when RadiusSq &lt; thetaSq * distSq. If the fixed per-thread stack would overflow, the query
        /// falls back to the exact brute sum over every triangle (never silently skips children).
        /// </summary>
        public static void BvhWindingNumberKernel(
            Index1D index,
            ArrayView<V3> points,
            ArrayView<GpuBvhNode> nodes,
            ArrayView<V3> verts,
            ArrayView<int> sortedTris,
            int triCount,
            double thetaSq,
            ArrayView<double> windingOut)
        {
            var stack = LocalMemory.Allocate<int>(BvhStackSize);
            V3 p = points[index];
            double sum = 0.0;
            int sp = 0;
            stack[sp++] = 0;
            bool overflow = false;

            while (sp > 0)
            {
                GpuBvhNode node = nodes[stack[--sp]];
                if (node.TriCount > 0)
                {
                    for (int i = 0; i < node.TriCount; i++)
                    {
                        int t = node.TriStart + i;
                        sum += TriangleSolidAngle(p, verts[sortedTris[3 * t]], verts[sortedTris[3 * t + 1]], verts[sortedTris[3 * t + 2]]);
                    }
                    continue;
                }

                double rx = node.Centroid.X - p.X, ry = node.Centroid.Y - p.Y, rz = node.Centroid.Z - p.Z;
                double distSq = rx * rx + ry * ry + rz * rz;
                if (distSq > 0.0 && node.RadiusSq < thetaSq * distSq)
                {
                    sum += FarFieldSolidAngle(node, rx, ry, rz, distSq);
                }
                else
                {
                    if (sp + 2 > BvhStackSize)
                    {
                        overflow = true;
                        break;
                    }
                    stack[sp++] = node.LeftChild;
                    stack[sp++] = node.RightChild;
                }
            }

            if (overflow)
            {
                sum = 0.0;
                for (int t = 0; t < triCount; t++)
                    sum += TriangleSolidAngle(p, verts[sortedTris[3 * t]], verts[sortedTris[3 * t + 1]], verts[sortedTris[3 * t + 2]]);
            }

            windingOut[index] = sum * InvFourPi;
        }

        /// <summary>Host twin of <see cref="BvhWindingNumberKernel"/> (same node math, same acceptance test).</summary>
        internal static double HostBvhWinding(V3 p, GpuBvhNode[] nodes, V3[] verts, int[] sortedTris, double thetaSq)
        {
            double sum = 0.0;
            var stack = new Stack<int>();
            stack.Push(0);
            while (stack.Count > 0)
            {
                GpuBvhNode node = nodes[stack.Pop()];
                if (node.TriCount > 0)
                {
                    for (int i = 0; i < node.TriCount; i++)
                    {
                        int t = node.TriStart + i;
                        sum += TriangleSolidAngle(p, verts[sortedTris[3 * t]], verts[sortedTris[3 * t + 1]], verts[sortedTris[3 * t + 2]]);
                    }
                    continue;
                }
                double rx = node.Centroid.X - p.X, ry = node.Centroid.Y - p.Y, rz = node.Centroid.Z - p.Z;
                double distSq = rx * rx + ry * ry + rz * rz;
                if (distSq > 0.0 && node.RadiusSq < thetaSq * distSq)
                    sum += FarFieldSolidAngle(node, rx, ry, rz, distSq);
                else
                {
                    stack.Push(node.LeftChild);
                    stack.Push(node.RightChild);
                }
            }
            return sum * InvFourPi;
        }

        /// <summary>Host twin of the brute kernels.</summary>
        internal static double HostBruteWinding(V3 p, V3[] verts, int[] tris)
        {
            double sum = 0.0;
            for (int t = 0; t < tris.Length; t += 3)
                sum += TriangleSolidAngle(p, verts[tris[t]], verts[tris[t + 1]], verts[tris[t + 2]]);
            return sum * InvFourPi;
        }

        /// <summary>Builds the BVH on the host (recursive midpoint split, leaves of up to 8 triangles).</summary>
        public static (GpuBvhNode[] Nodes, int[] SortedTris) BuildBvh(V3[] verts, int[] tris)
        {
            int triCount = tris.Length / 3;
            if (triCount == 0)
                return (Array.Empty<GpuBvhNode>(), Array.Empty<int>());

            int[] order = new int[triCount];
            V3[] centroids = new V3[triCount];
            V3[] normalAreas = new V3[triCount];
            for (int i = 0; i < triCount; i++)
            {
                order[i] = i;
                V3 a = verts[tris[3 * i]], b = verts[tris[3 * i + 1]], c = verts[tris[3 * i + 2]];
                centroids[i] = new V3((a.X + b.X + c.X) / 3.0, (a.Y + b.Y + c.Y) / 3.0, (a.Z + b.Z + c.Z) / 3.0);
                V3 ab = b - a, ac = c - a;
                normalAreas[i] = new V3(0.5 * (ab.Y * ac.Z - ab.Z * ac.Y), 0.5 * (ab.Z * ac.X - ab.X * ac.Z), 0.5 * (ab.X * ac.Y - ab.Y * ac.X));
            }

            var nodes = new List<GpuBvhNode>();
            int[] sortedTris = new int[triCount * 3];
            int written = 0;
            Build(0, triCount, order, centroids, normalAreas, verts, tris, sortedTris, ref written, nodes);
            return (nodes.ToArray(), sortedTris);
        }

        private static int Build(int start, int count, int[] order, V3[] centroids, V3[] normalAreas, V3[] verts, int[] tris,
                                 int[] sortedTris, ref int written, List<GpuBvhNode> nodes)
        {
            int nodeIdx = nodes.Count;
            nodes.Add(default);

            double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
            double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;
            double nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0, areaSum = 0;
            for (int i = start; i < start + count; i++)
            {
                int t = order[i];
                V3 na = normalAreas[t];
                double area = Math.Sqrt(na.X * na.X + na.Y * na.Y + na.Z * na.Z);
                areaSum += area;
                nx += na.X; ny += na.Y; nz += na.Z;
                cx += centroids[t].X * area; cy += centroids[t].Y * area; cz += centroids[t].Z * area;
                for (int k = 0; k < 3; k++)
                {
                    V3 v = verts[tris[3 * t + k]];
                    minX = Math.Min(minX, v.X); minY = Math.Min(minY, v.Y); minZ = Math.Min(minZ, v.Z);
                    maxX = Math.Max(maxX, v.X); maxY = Math.Max(maxY, v.Y); maxZ = Math.Max(maxZ, v.Z);
                }
            }

            V3 centre = areaSum > 0
                ? new V3(cx / areaSum, cy / areaSum, cz / areaSum)
                : new V3((minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5);

            double radiusSq = 0;
            double m00 = 0, m01 = 0, m02 = 0, m10 = 0, m11 = 0, m12 = 0, m20 = 0, m21 = 0, m22 = 0;
            var pv = new double[3];
            var uv = new double[3];
            var cub = new double[10]; // xxx yyy zzz xxy xxz xyy yyz xzz yzz xyz
            var avg = new double[3, 3];
            var nv = new double[3];
            for (int i = start; i < start + count; i++)
            {
                int t = order[i];
                for (int k = 0; k < 3; k++)
                {
                    V3 d = verts[tris[3 * t + k]] - centre;
                    radiusSq = Math.Max(radiusSq, V3.Dot(d, d));
                }
                V3 n = normalAreas[t];
                V3 dc = centroids[t] - centre;
                m00 += n.X * dc.X; m01 += n.X * dc.Y; m02 += n.X * dc.Z;
                m10 += n.Y * dc.X; m11 += n.Y * dc.Y; m12 += n.Y * dc.Z;
                m20 += n.Z * dc.X; m21 += n.Z * dc.Y; m22 += n.Z * dc.Z;

                // Exact triangle average of d d^T: (sum_k v_k v_k^T + s s^T) / 12, v_k = vertex - centre, s = sum v_k.
                V3 v0 = verts[tris[3 * t]] - centre, v1 = verts[tris[3 * t + 1]] - centre, v2 = verts[tris[3 * t + 2]] - centre;
                V3 sv = v0 + v1 + v2;
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++)
                        avg[j, k] = (Coord(v0, j) * Coord(v0, k) + Coord(v1, j) * Coord(v1, k) + Coord(v2, j) * Coord(v2, k)
                                     + Coord(sv, j) * Coord(sv, k)) / 12.0;
                nv[0] = n.X; nv[1] = n.Y; nv[2] = n.Z;
                double tr = avg[0, 0] + avg[1, 1] + avg[2, 2];
                for (int a = 0; a < 3; a++)
                {
                    uv[a] += tr * nv[a];
                    for (int j = 0; j < 3; j++)
                    {
                        pv[a] += avg[a, j] * nv[j];
                        for (int k = 0; k < 3; k++)
                            cub[CubicSlot(a, j, k)] += nv[a] * avg[j, k];
                    }
                }
            }

            var node = new GpuBvhNode
            {
                Centroid = centre,
                NormalArea = new V3(nx, ny, nz),
                M0 = new V3(m00, m01, m02),
                M1 = new V3(m10, m11, m12),
                M2 = new V3(m20, m21, m22),
                P = new V3(pv[0], pv[1], pv[2]),
                U = new V3(uv[0], uv[1], uv[2]),
                C0 = new V3(cub[0], cub[1], cub[2]),
                C1 = new V3(cub[3], cub[4], cub[5]),
                C2 = new V3(cub[6], cub[7], cub[8]),
                Cxyz = cub[9],
                RadiusSq = radiusSq,
                LeftChild = -1,
                RightChild = -1,
                TriStart = -1,
                TriCount = 0
            };

            const int LeafSize = 8;
            if (count <= LeafSize)
            {
                node.TriStart = written / 3;
                node.TriCount = count;
                for (int i = start; i < start + count; i++)
                {
                    int t = order[i];
                    sortedTris[written++] = tris[3 * t];
                    sortedTris[written++] = tris[3 * t + 1];
                    sortedTris[written++] = tris[3 * t + 2];
                }
                nodes[nodeIdx] = node;
                return nodeIdx;
            }

            double ex = maxX - minX, ey = maxY - minY, ez = maxZ - minZ;
            int axis = ex >= ey && ex >= ez ? 0 : (ey >= ez ? 1 : 2);
            double mid = axis == 0 ? (minX + maxX) * 0.5 : axis == 1 ? (minY + maxY) * 0.5 : (minZ + maxZ) * 0.5;

            int left = start, right = start + count - 1;
            while (left <= right)
            {
                V3 c = centroids[order[left]];
                double val = axis == 0 ? c.X : axis == 1 ? c.Y : c.Z;
                if (val < mid) left++;
                else { int tmp = order[left]; order[left] = order[right]; order[right] = tmp; right--; }
            }
            int leftCount = left - start;
            if (leftCount == 0 || leftCount == count)
                leftCount = count / 2;

            node.LeftChild = Build(start, leftCount, order, centroids, normalAreas, verts, tris, sortedTris, ref written, nodes);
            node.RightChild = Build(start + leftCount, count - leftCount, order, centroids, normalAreas, verts, tris, sortedTris, ref written, nodes);
            nodes[nodeIdx] = node;
            return nodeIdx;
        }

        private static double Coord(V3 v, int axis) => axis == 0 ? v.X : axis == 1 ? v.Y : v.Z;

        // Monomial slot of y_i y_j y_k in the order xxx yyy zzz xxy xxz xyy yyz xzz yzz xyz.
        private static int CubicSlot(int i, int j, int k)
        {
            int cx = (i == 0 ? 1 : 0) + (j == 0 ? 1 : 0) + (k == 0 ? 1 : 0);
            int cy = (i == 1 ? 1 : 0) + (j == 1 ? 1 : 0) + (k == 1 ? 1 : 0);
            int cz = 3 - cx - cy;
            if (cx == 3) return 0;
            if (cy == 3) return 1;
            if (cz == 3) return 2;
            if (cx == 2) return cy == 1 ? 3 : 4;
            if (cy == 2) return cx == 1 ? 5 : 6;
            if (cz == 2) return cx == 1 ? 7 : 8;
            return 9;
        }
    }
}
