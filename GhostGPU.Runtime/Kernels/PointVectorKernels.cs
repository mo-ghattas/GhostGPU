using System;
using GhostGPU.Core;
using ILGPU;
using ILGPU.Algorithms;

namespace GhostGPU.Runtime.Kernels
{
    internal static class PointVectorKernels
    {
        // =====================================================================
        // Pure static element functions (shared bit-identical between host & device)
        // =====================================================================

        public static V3 ConstructPointElement(double x, double y, double z)
        {
            return new V3(x, y, z);
        }

        public static double DeconstructXElement(V3 v) => v.X;
        public static double DeconstructYElement(V3 v) => v.Y;
        public static double DeconstructZElement(V3 v) => v.Z;

        public static V3 ConstructCylindricalPointElement(double r, double theta, double z)
        {
            if (!VectorKernels.IsValidDouble(r) || !VectorKernels.IsValidDouble(theta) || !VectorKernels.IsValidDouble(z))
                return new V3(0.0, 0.0, 0.0);

            double x = Fp.Mul(r, XMath.Cos(theta));
            double y = Fp.Mul(r, XMath.Sin(theta));
            return new V3(x, y, z);
        }

        public static V3 ConstructSphericalPointElement(double r, double theta, double phi)
        {
            if (!VectorKernels.IsValidDouble(r) || !VectorKernels.IsValidDouble(theta) || !VectorKernels.IsValidDouble(phi))
                return new V3(0.0, 0.0, 0.0);

            // Standard spherical convention:
            // theta = azimuth angle in XY plane from +X
            // phi = elevation angle from XY plane
            double cosPhi = XMath.Cos(phi);
            double sinPhi = XMath.Sin(phi);
            double x = Fp.Mul(r, Fp.Mul(cosPhi, XMath.Cos(theta)));
            double y = Fp.Mul(r, Fp.Mul(cosPhi, XMath.Sin(theta)));
            double z = Fp.Mul(r, sinPhi);
            return new V3(x, y, z);
        }

        public static V3 VectorAddElement(V3 a, V3 b)
        {
            return new V3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        }

        public static V3 VectorSubtractElement(V3 a, V3 b)
        {
            return new V3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        public static V3 VectorReverseElement(V3 v)
        {
            return new V3(-v.X, -v.Y, -v.Z);
        }

        public static double VectorLengthElement(V3 v)
        {
            return VectorKernels.GetLengthHelper(v.X, v.Y, v.Z);
        }

        public static V3 VectorScaleElement(V3 v, double s)
        {
            if (!VectorKernels.IsValidDouble(s))
                return new V3(0.0, 0.0, 0.0);
            return new V3(Fp.Mul(v.X, s), Fp.Mul(v.Y, s), Fp.Mul(v.Z, s));
        }

        public static V3 VectorDivideElement(V3 v, double d)
        {
            if (!VectorKernels.IsValidDouble(d) || d == 0.0)
                return new V3(0.0, 0.0, 0.0);
            // Direct division avoids 1.0 / d overflow for subnormals (e.g. 1e-320) and avoids 0.0 * inf = NaN
            return new V3(v.X / d, v.Y / d, v.Z / d);
        }

        public static double VectorDotElement(V3 a, V3 b)
        {
            return Fp.Mul(a.X, b.X) + Fp.Mul(a.Y, b.Y) + Fp.Mul(a.Z, b.Z);
        }

        public static V3 VectorCrossElement(V3 a, V3 b)
        {
            double x = Fp.Mul(a.Y, b.Z) - Fp.Mul(a.Z, b.Y);
            double y = Fp.Mul(a.Z, b.X) - Fp.Mul(a.X, b.Z);
            double z = Fp.Mul(a.X, b.Y) - Fp.Mul(a.Y, b.X);
            return new V3(x, y, z);
        }

        public static double VectorAngleElement(V3 a, V3 b)
        {
            // Normalize inputs first to prevent overflow in dot/cross when coordinates are huge (e.g. 1e155)
            V3 uA = VectorKernels.NormalizeVector(a);
            V3 uB = VectorKernels.NormalizeVector(b);
            if ((uA.X == 0.0 && uA.Y == 0.0 && uA.Z == 0.0) || (uB.X == 0.0 && uB.Y == 0.0 && uB.Z == 0.0))
                return 0.0;

            V3 cross = VectorCrossElement(uA, uB);
            double crossLen = VectorKernels.GetLengthHelper(cross.X, cross.Y, cross.Z);
            double dot = VectorDotElement(uA, uB);
            return XMath.Atan2(crossLen, dot);
        }

        public static V3 UnitXElement(double s) => new V3(s, 0.0, 0.0);
        public static V3 UnitYElement(double s) => new V3(0.0, s, 0.0);
        public static V3 UnitZElement(double s) => new V3(0.0, 0.0, s);

        public static V3 PointInterpolateElement(V3 a, V3 b, double t)
        {
            if (!VectorKernels.IsValidDouble(t))
                return a;
            if (t == 0.0) return a;
            if (t == 1.0) return b;

            // Stable weighted combination: handles finite endpoints whose difference (b - a) might overflow
            double oneMinusT = 1.0 - t;
            double x = Fp.Mul(oneMinusT, a.X) + Fp.Mul(t, b.X);
            double y = Fp.Mul(oneMinusT, a.Y) + Fp.Mul(t, b.Y);
            double z = Fp.Mul(oneMinusT, a.Z) + Fp.Mul(t, b.Z);
            return new V3(x, y, z);
        }

        public static V3 ProjectVectorToPlaneElement(V3 v, V3 normal)
        {
            V3 u = VectorKernels.NormalizeVector(normal);
            if (u.X == 0.0 && u.Y == 0.0 && u.Z == 0.0)
                return v;
            double dot = VectorDotElement(v, u);
            return new V3(v.X - Fp.Mul(u.X, dot), v.Y - Fp.Mul(u.Y, dot), v.Z - Fp.Mul(u.Z, dot));
        }

        public static V3 PointProjectElement(V3 p, V3 planeOrigin, V3 planeNormal)
        {
            V3 u = VectorKernels.NormalizeVector(planeNormal);
            if (u.X == 0.0 && u.Y == 0.0 && u.Z == 0.0)
                return p;
            V3 diff = new V3(p.X - planeOrigin.X, p.Y - planeOrigin.Y, p.Z - planeOrigin.Z);
            double d = VectorDotElement(diff, u);
            return new V3(p.X - Fp.Mul(u.X, d), p.Y - Fp.Mul(u.Y, d), p.Z - Fp.Mul(u.Z, d));
        }

        public static V3 RotateVectorAxisElement(V3 v, V3 axis, double angle)
        {
            if (!VectorKernels.IsValidDouble(angle) || angle == 0.0)
                return v;

            V3 k = VectorKernels.NormalizeVector(axis);
            if (k.X == 0.0 && k.Y == 0.0 && k.Z == 0.0)
                return v;

            double cosT = XMath.Cos(angle);
            double sinT = XMath.Sin(angle);
            double oneMinusCos = 1.0 - cosT;

            double dot = VectorDotElement(k, v);
            V3 cross = VectorCrossElement(k, v);

            // Rodrigues rotation formula: v*cos(t) + (k x v)*sin(t) + k*(k . v)*(1 - cos(t))
            double x = Fp.Mul(v.X, cosT) + Fp.Mul(cross.X, sinT) + Fp.Mul(k.X, Fp.Mul(dot, oneMinusCos));
            double y = Fp.Mul(v.Y, cosT) + Fp.Mul(cross.Y, sinT) + Fp.Mul(k.Y, Fp.Mul(dot, oneMinusCos));
            double z = Fp.Mul(v.Z, cosT) + Fp.Mul(cross.Z, sinT) + Fp.Mul(k.Z, Fp.Mul(dot, oneMinusCos));
            return new V3(x, y, z);
        }

        public static V3 PointScaleElement(V3 p, V3 center, double factor)
        {
            if (!VectorKernels.IsValidDouble(factor))
                return p;
            double dx = p.X - center.X;
            double dy = p.Y - center.Y;
            double dz = p.Z - center.Z;
            return new V3(center.X + Fp.Mul(dx, factor), center.Y + Fp.Mul(dy, factor), center.Z + Fp.Mul(dz, factor));
        }

        public static V3 PointRotateAxisElement(V3 p, V3 center, V3 axis, double angle)
        {
            V3 diff = new V3(p.X - center.X, p.Y - center.Y, p.Z - center.Z);
            V3 rot = RotateVectorAxisElement(diff, axis, angle);
            return new V3(center.X + rot.X, center.Y + rot.Y, center.Z + rot.Z);
        }

        public static V3 PointMirrorElement(V3 p, V3 planeOrigin, V3 planeNormal)
        {
            V3 u = VectorKernels.NormalizeVector(planeNormal);
            if (u.X == 0.0 && u.Y == 0.0 && u.Z == 0.0)
                return p;
            V3 diff = new V3(p.X - planeOrigin.X, p.Y - planeOrigin.Y, p.Z - planeOrigin.Z);
            double d = VectorDotElement(diff, u);
            double twoD = 2.0 * d;
            return new V3(p.X - Fp.Mul(u.X, twoD), p.Y - Fp.Mul(u.Y, twoD), p.Z - Fp.Mul(u.Z, twoD));
        }

        // =====================================================================
        // ILGPU Device Kernels
        // =====================================================================

        public static void ConstructV3Kernel(
            Index1D index,
            ArrayView<double> x,
            ArrayView<double> y,
            ArrayView<double> z,
            int xCount,
            int yCount,
            int zCount,
            ArrayView<V3> output)
        {
            int i = index;
            int ix = i < xCount ? i : xCount - 1;
            int iy = i < yCount ? i : yCount - 1;
            int iz = i < zCount ? i : zCount - 1;
            output[i] = ConstructPointElement(x[ix], y[iy], z[iz]);
        }

        public static void DeconstructV3Kernel(
            Index1D index,
            ArrayView<V3> input,
            ArrayView<double> outX,
            ArrayView<double> outY,
            ArrayView<double> outZ)
        {
            int i = index;
            V3 v = input[i];
            outX[i] = v.X;
            outY[i] = v.Y;
            outZ[i] = v.Z;
        }

        public static void ConstructCylindricalKernel(
            Index1D index,
            ArrayView<double> r,
            ArrayView<double> theta,
            ArrayView<double> z,
            int rCount,
            int tCount,
            int zCount,
            ArrayView<V3> output)
        {
            int i = index;
            int ir = i < rCount ? i : rCount - 1;
            int it = i < tCount ? i : tCount - 1;
            int iz = i < zCount ? i : zCount - 1;
            output[i] = ConstructCylindricalPointElement(r[ir], theta[it], z[iz]);
        }

        public static void ConstructSphericalKernel(
            Index1D index,
            ArrayView<double> r,
            ArrayView<double> theta,
            ArrayView<double> phi,
            int rCount,
            int tCount,
            int pCount,
            ArrayView<V3> output)
        {
            int i = index;
            int ir = i < rCount ? i : rCount - 1;
            int it = i < tCount ? i : tCount - 1;
            int ip = i < pCount ? i : pCount - 1;
            output[i] = ConstructSphericalPointElement(r[ir], theta[it], phi[ip]);
        }

        public static void VectorBinaryOpKernel(
            Index1D index,
            ArrayView<V3> a,
            ArrayView<V3> b,
            int aCount,
            int bCount,
            int opCode,
            ArrayView<V3> output)
        {
            int i = index;
            int ia = i < aCount ? i : aCount - 1;
            int ib = i < bCount ? i : bCount - 1;
            V3 va = a[ia];
            V3 vb = b[ib];
            if (opCode == 0) output[i] = VectorAddElement(va, vb);
            else if (opCode == 1) output[i] = VectorSubtractElement(va, vb);
            else if (opCode == 2) output[i] = VectorCrossElement(va, vb);
        }

        public static void VectorReverseKernel(
            Index1D index,
            ArrayView<V3> input,
            ArrayView<V3> output)
        {
            int i = index;
            output[i] = VectorReverseElement(input[i]);
        }

        public static void VectorLengthKernel(
            Index1D index,
            ArrayView<V3> input,
            ArrayView<double> output)
        {
            int i = index;
            output[i] = VectorLengthElement(input[i]);
        }

        public static void VectorScaleKernel(
            Index1D index,
            ArrayView<V3> vectors,
            ArrayView<double> factors,
            int vCount,
            int fCount,
            ArrayView<V3> output)
        {
            int i = index;
            int iv = i < vCount ? i : vCount - 1;
            int @if = i < fCount ? i : fCount - 1;
            output[i] = VectorScaleElement(vectors[iv], factors[@if]);
        }

        public static void VectorDivideKernel(
            Index1D index,
            ArrayView<V3> vectors,
            ArrayView<double> divisors,
            int vCount,
            int dCount,
            ArrayView<V3> output)
        {
            int i = index;
            int iv = i < vCount ? i : vCount - 1;
            int id = i < dCount ? i : dCount - 1;
            output[i] = VectorDivideElement(vectors[iv], divisors[id]);
        }

        public static void VectorDotKernel(
            Index1D index,
            ArrayView<V3> a,
            ArrayView<V3> b,
            int aCount,
            int bCount,
            ArrayView<double> output)
        {
            int i = index;
            int ia = i < aCount ? i : aCount - 1;
            int ib = i < bCount ? i : bCount - 1;
            output[i] = VectorDotElement(a[ia], b[ib]);
        }

        public static void VectorAngleKernel(
            Index1D index,
            ArrayView<V3> a,
            ArrayView<V3> b,
            int aCount,
            int bCount,
            ArrayView<double> output)
        {
            int i = index;
            int ia = i < aCount ? i : aCount - 1;
            int ib = i < bCount ? i : bCount - 1;
            output[i] = VectorAngleElement(a[ia], b[ib]);
        }

        public static void UnitAxisKernel(
            Index1D index,
            ArrayView<double> factors,
            int fCount,
            int axis,
            ArrayView<V3> output)
        {
            int i = index;
            double s = fCount > 0 ? (i < fCount ? factors[i] : factors[fCount - 1]) : 1.0;
            if (axis == 0) output[i] = UnitXElement(s);
            else if (axis == 1) output[i] = UnitYElement(s);
            else output[i] = UnitZElement(s);
        }

        public static void PointInterpolateKernel(
            Index1D index,
            ArrayView<V3> a,
            ArrayView<V3> b,
            ArrayView<double> t,
            int aCount,
            int bCount,
            int tCount,
            ArrayView<V3> output)
        {
            int i = index;
            int ia = i < aCount ? i : aCount - 1;
            int ib = i < bCount ? i : bCount - 1;
            int it = i < tCount ? i : tCount - 1;
            output[i] = PointInterpolateElement(a[ia], b[ib], t[it]);
        }

        public static void ProjectVectorToPlaneKernel(
            Index1D index,
            ArrayView<V3> vectors,
            ArrayView<V3> normals,
            int vCount,
            int nCount,
            ArrayView<V3> output)
        {
            int i = index;
            int iv = i < vCount ? i : vCount - 1;
            int inorm = i < nCount ? i : nCount - 1;
            output[i] = ProjectVectorToPlaneElement(vectors[iv], normals[inorm]);
        }

        public static void PointProjectKernel(
            Index1D index,
            ArrayView<V3> points,
            ArrayView<V3> planeOrigins,
            ArrayView<V3> planeNormals,
            int pCount,
            int oCount,
            int nCount,
            ArrayView<V3> output)
        {
            int i = index;
            int ip = i < pCount ? i : pCount - 1;
            int io = i < oCount ? i : oCount - 1;
            int @in = i < nCount ? i : nCount - 1;
            output[i] = PointProjectElement(points[ip], planeOrigins[io], planeNormals[@in]);
        }

        public static void RotateVectorAxisKernel(
            Index1D index,
            ArrayView<V3> vectors,
            ArrayView<V3> axes,
            ArrayView<double> angles,
            int vCount,
            int aCount,
            int tCount,
            ArrayView<V3> output)
        {
            int i = index;
            int iv = i < vCount ? i : vCount - 1;
            int ia = i < aCount ? i : aCount - 1;
            int it = i < tCount ? i : tCount - 1;
            output[i] = RotateVectorAxisElement(vectors[iv], axes[ia], angles[it]);
        }

        public static void PointScaleKernel(
            Index1D index,
            ArrayView<V3> points,
            ArrayView<V3> centers,
            ArrayView<double> factors,
            int pCount,
            int cCount,
            int fCount,
            ArrayView<V3> output)
        {
            int i = index;
            int ip = i < pCount ? i : pCount - 1;
            int ic = i < cCount ? i : cCount - 1;
            int @if = i < fCount ? i : fCount - 1;
            output[i] = PointScaleElement(points[ip], centers[ic], factors[@if]);
        }

        public static void PointRotateAxisKernel(
            Index1D index,
            ArrayView<V3> points,
            ArrayView<V3> centers,
            ArrayView<V3> axes,
            ArrayView<double> angles,
            int pCount,
            int cCount,
            int aCount,
            int tCount,
            ArrayView<V3> output)
        {
            int i = index;
            int ip = i < pCount ? i : pCount - 1;
            int ic = i < cCount ? i : cCount - 1;
            int ia = i < aCount ? i : aCount - 1;
            int it = i < tCount ? i : tCount - 1;
            output[i] = PointRotateAxisElement(points[ip], centers[ic], axes[ia], angles[it]);
        }

        public static void PointMirrorKernel(
            Index1D index,
            ArrayView<V3> points,
            ArrayView<V3> planeOrigins,
            ArrayView<V3> planeNormals,
            int pCount,
            int oCount,
            int nCount,
            ArrayView<V3> output)
        {
            int i = index;
            int ip = i < pCount ? i : pCount - 1;
            int io = i < oCount ? i : oCount - 1;
            int @in = i < nCount ? i : nCount - 1;
            output[i] = PointMirrorElement(points[ip], planeOrigins[io], planeNormals[@in]);
        }
    }
}
