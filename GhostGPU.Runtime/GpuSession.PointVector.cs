using System;
using GhostGPU.Core;
using GhostGPU.Runtime.Kernels;
using GhostGPU.Runtime.Memory;
using ILGPU;

namespace GhostGPU.Runtime
{
    public sealed partial class GpuSession
    {
        // =====================================================================
        // Point Construction & Deconstruction
        // =====================================================================

        public GpuBuffer ConstructPoints(GpuBuffer x, GpuBuffer y, GpuBuffer z)
        {
            var bx = ValidateBuffer(x, BufferKind.Scalars);
            var by = ValidateBuffer(y, BufferKind.Scalars);
            var bz = ValidateBuffer(z, BufferKind.Scalars);
            var format = _options.DefaultVectorFormat;

            if (bx.Count == 0 || by.Count == 0 || bz.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format);

            int count = Math.Max(bx.Count, Math.Max(by.Count, bz.Count));

            if (CanRunOnHost(count, bx, by, bz))
            {
                var px = HostF64(bx);
                var py = HostF64(by);
                var pz = HostF64(bz);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.ConstructPointElement(
                    px[Rl(i, px.Length)],
                    py[Rl(i, py.Length)],
                    pz[Rl(i, pz.Length)]));
                return HostResult(BufferKind.Points, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<V3>>)PointVectorKernels.ConstructV3Kernel)
                    (new Index1D(count), DeviceF64(scope, bx), DeviceF64(scope, by), DeviceF64(scope, bz), bx.Count, by.Count, bz.Count, o)), bx, by, bz);
        }

        public (GpuBuffer X, GpuBuffer Y, GpuBuffer Z) DeconstructPoints(GpuBuffer points)
        {
            DeconstructPoints(points, out var x, out var y, out var z);
            return (x, y, z);
        }

        public void DeconstructPoints(GpuBuffer points, out GpuBuffer x, out GpuBuffer y, out GpuBuffer z)
        {
            var bp = ValidateBuffer(points, BufferKind.Points);
            var scalarFormat = ElementType.F64;

            if (bp.Count == 0)
            {
                x = CreateEmptyBuffer(BufferKind.Scalars, scalarFormat);
                y = CreateEmptyBuffer(BufferKind.Scalars, scalarFormat);
                z = CreateEmptyBuffer(BufferKind.Scalars, scalarFormat);
                return;
            }

            int count = bp.Count;

            if (CanRunOnHost(count, bp))
            {
                var pts = HostV3(bp);
                var rx = new double[count];
                var ry = new double[count];
                var rz = new double[count];
                HostFor(count, i =>
                {
                    rx[i] = PointVectorKernels.DeconstructXElement(pts[i]);
                    ry[i] = PointVectorKernels.DeconstructYElement(pts[i]);
                    rz[i] = PointVectorKernels.DeconstructZElement(pts[i]);
                });
                x = HostResult(BufferKind.Scalars, scalarFormat, default, rx);
                y = HostResult(BufferKind.Scalars, scalarFormat, default, ry);
                z = HostResult(BufferKind.Scalars, scalarFormat, default, rz);
                return;
            }

            GpuBuffer bx = null, by = null, bz = null;
            try
            {
                RunDevice(scope =>
                {
                    DeviceLease<double> leaseX = null;
                    DeviceLease<double> leaseY = null;
                    DeviceLease<double> leaseZ = null;
                    try
                    {
                        leaseX = Rent<double>(count);
                        leaseY = Rent<double>(count);
                        leaseZ = Rent<double>(count);

                        GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<double>, ArrayView<double>, ArrayView<double>>>(
                            (Action<Index1D, ArrayView<V3>, ArrayView<double>, ArrayView<double>, ArrayView<double>>)PointVectorKernels.DeconstructV3Kernel)
                            (new Index1D(count), DeviceV3(scope, bp), leaseX.View, leaseY.View, leaseZ.View);
                        CountDispatch();

                        bx = Adopt(leaseX, BufferKind.Scalars, ElementType.F64, count);
                        by = Adopt(leaseY, BufferKind.Scalars, ElementType.F64, count);
                        bz = Adopt(leaseZ, BufferKind.Scalars, ElementType.F64, count);
                        return 0;
                    }
                    catch
                    {
                        if (bx == null) leaseX?.Dispose();
                        if (by == null) leaseY?.Dispose();
                        if (bz == null) leaseZ?.Dispose();
                        throw;
                    }
                }, bp);
            }
            catch
            {
                bx?.Dispose();
                by?.Dispose();
                bz?.Dispose();
                throw;
            }

            x = bx;
            y = by;
            z = bz;
        }

        // =====================================================================
        // Vector Construction & Deconstruction
        // =====================================================================

        public GpuBuffer ConstructVectors(GpuBuffer x, GpuBuffer y, GpuBuffer z)
        {
            var bx = ValidateBuffer(x, BufferKind.Scalars);
            var by = ValidateBuffer(y, BufferKind.Scalars);
            var bz = ValidateBuffer(z, BufferKind.Scalars);
            var format = _options.DefaultVectorFormat;

            if (bx.Count == 0 || by.Count == 0 || bz.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = Math.Max(bx.Count, Math.Max(by.Count, bz.Count));

            if (CanRunOnHost(count, bx, by, bz))
            {
                var px = HostF64(bx);
                var py = HostF64(by);
                var pz = HostF64(bz);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.ConstructPointElement(
                    px[Rl(i, px.Length)],
                    py[Rl(i, py.Length)],
                    pz[Rl(i, pz.Length)]));
                return HostResult(BufferKind.Vectors, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<V3>>)PointVectorKernels.ConstructV3Kernel)
                    (new Index1D(count), DeviceF64(scope, bx), DeviceF64(scope, by), DeviceF64(scope, bz), bx.Count, by.Count, bz.Count, o)), bx, by, bz);
        }

        public (GpuBuffer X, GpuBuffer Y, GpuBuffer Z) DeconstructVectors(GpuBuffer vectors)
        {
            DeconstructVectors(vectors, out var x, out var y, out var z);
            return (x, y, z);
        }

        public void DeconstructVectors(GpuBuffer vectors, out GpuBuffer x, out GpuBuffer y, out GpuBuffer z)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var scalarFormat = ElementType.F64;

            if (bv.Count == 0)
            {
                x = CreateEmptyBuffer(BufferKind.Scalars, scalarFormat);
                y = CreateEmptyBuffer(BufferKind.Scalars, scalarFormat);
                z = CreateEmptyBuffer(BufferKind.Scalars, scalarFormat);
                return;
            }

            int count = bv.Count;

            if (CanRunOnHost(count, bv))
            {
                var vecs = HostV3(bv);
                var rx = new double[count];
                var ry = new double[count];
                var rz = new double[count];
                HostFor(count, i =>
                {
                    rx[i] = PointVectorKernels.DeconstructXElement(vecs[i]);
                    ry[i] = PointVectorKernels.DeconstructYElement(vecs[i]);
                    rz[i] = PointVectorKernels.DeconstructZElement(vecs[i]);
                });
                x = HostResult(BufferKind.Scalars, scalarFormat, default, rx);
                y = HostResult(BufferKind.Scalars, scalarFormat, default, ry);
                z = HostResult(BufferKind.Scalars, scalarFormat, default, rz);
                return;
            }

            GpuBuffer bx = null, by = null, bz = null;
            try
            {
                RunDevice(scope =>
                {
                    DeviceLease<double> leaseX = null;
                    DeviceLease<double> leaseY = null;
                    DeviceLease<double> leaseZ = null;
                    try
                    {
                        leaseX = Rent<double>(count);
                        leaseY = Rent<double>(count);
                        leaseZ = Rent<double>(count);

                        GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<double>, ArrayView<double>, ArrayView<double>>>(
                            (Action<Index1D, ArrayView<V3>, ArrayView<double>, ArrayView<double>, ArrayView<double>>)PointVectorKernels.DeconstructV3Kernel)
                            (new Index1D(count), DeviceV3(scope, bv), leaseX.View, leaseY.View, leaseZ.View);
                        CountDispatch();

                        bx = Adopt(leaseX, BufferKind.Scalars, ElementType.F64, count);
                        by = Adopt(leaseY, BufferKind.Scalars, ElementType.F64, count);
                        bz = Adopt(leaseZ, BufferKind.Scalars, ElementType.F64, count);
                        return 0;
                    }
                    catch
                    {
                        if (bx == null) leaseX?.Dispose();
                        if (by == null) leaseY?.Dispose();
                        if (bz == null) leaseZ?.Dispose();
                        throw;
                    }
                }, bv);
            }
            catch
            {
                bx?.Dispose();
                by?.Dispose();
                bz?.Dispose();
                throw;
            }

            x = bx;
            y = by;
            z = bz;
        }

        // =====================================================================
        // Cylindrical & Spherical Construction
        // =====================================================================

        public GpuBuffer ConstructCylindricalPoints(GpuBuffer radius, GpuBuffer theta, GpuBuffer z)
        {
            var br = ValidateBuffer(radius, BufferKind.Scalars);
            var bt = ValidateBuffer(theta, BufferKind.Scalars);
            var bz = ValidateBuffer(z, BufferKind.Scalars);
            var format = _options.DefaultVectorFormat;

            if (br.Count == 0 || bt.Count == 0 || bz.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format);

            int count = Math.Max(br.Count, Math.Max(bt.Count, bz.Count));

            if (CanRunOnHost(count, br, bt, bz))
            {
                var pr = HostF64(br);
                var pt = HostF64(bt);
                var pz = HostF64(bz);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.ConstructCylindricalPointElement(
                    pr[Rl(i, pr.Length)],
                    pt[Rl(i, pt.Length)],
                    pz[Rl(i, pz.Length)]));
                return HostResult(BufferKind.Points, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<V3>>)PointVectorKernels.ConstructCylindricalKernel)
                    (new Index1D(count), DeviceF64(scope, br), DeviceF64(scope, bt), DeviceF64(scope, bz), br.Count, bt.Count, bz.Count, o)), br, bt, bz);
        }

        public GpuBuffer ConstructSphericalPoints(GpuBuffer radius, GpuBuffer theta, GpuBuffer phi)
        {
            var br = ValidateBuffer(radius, BufferKind.Scalars);
            var bt = ValidateBuffer(theta, BufferKind.Scalars);
            var bp = ValidateBuffer(phi, BufferKind.Scalars);
            var format = _options.DefaultVectorFormat;

            if (br.Count == 0 || bt.Count == 0 || bp.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format);

            int count = Math.Max(br.Count, Math.Max(bt.Count, bp.Count));

            if (CanRunOnHost(count, br, bt, bp))
            {
                var pr = HostF64(br);
                var pt = HostF64(bt);
                var pp = HostF64(bp);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.ConstructSphericalPointElement(
                    pr[Rl(i, pr.Length)],
                    pt[Rl(i, pt.Length)],
                    pp[Rl(i, pp.Length)]));
                return HostResult(BufferKind.Points, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int, ArrayView<V3>>)PointVectorKernels.ConstructSphericalKernel)
                    (new Index1D(count), DeviceF64(scope, br), DeviceF64(scope, bt), DeviceF64(scope, bp), br.Count, bt.Count, bp.Count, o)), br, bt, bp);
        }

        // =====================================================================
        // Vector Algebra
        // =====================================================================

        public GpuBuffer VectorAdd(GpuBuffer a, GpuBuffer b)
        {
            var ba = ValidateBuffer(a, BufferKind.Vectors);
            var bb = ValidateBuffer(b, BufferKind.Vectors);
            var format = WidestV3(ba, bb);
            V3 origin = format == ElementType.V3F ? PointVectorKernels.VectorAddElement(ba.Origin, bb.Origin) : default;

            if (ba.Count == 0 || bb.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format, origin);

            int count = Math.Max(ba.Count, bb.Count);

            if (CanRunOnHost(count, ba, bb))
            {
                var pa = HostV3(ba);
                var pb = HostV3(bb);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorAddElement(pa[Rl(i, pa.Length)], pb[Rl(i, pb.Length)]));
                return HostResult(BufferKind.Vectors, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>)PointVectorKernels.VectorBinaryOpKernel)
                    (new Index1D(count), DeviceV3(scope, ba), DeviceV3(scope, bb), ba.Count, bb.Count, 0, o)), ba, bb);
        }

        public GpuBuffer VectorSubtract(GpuBuffer a, GpuBuffer b)
        {
            var ba = ValidateBuffer(a, BufferKind.Vectors);
            var bb = ValidateBuffer(b, BufferKind.Vectors);
            var format = WidestV3(ba, bb);
            V3 origin = format == ElementType.V3F ? PointVectorKernels.VectorSubtractElement(ba.Origin, bb.Origin) : default;

            if (ba.Count == 0 || bb.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format, origin);

            int count = Math.Max(ba.Count, bb.Count);

            if (CanRunOnHost(count, ba, bb))
            {
                var pa = HostV3(ba);
                var pb = HostV3(bb);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorSubtractElement(pa[Rl(i, pa.Length)], pb[Rl(i, pb.Length)]));
                return HostResult(BufferKind.Vectors, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>)PointVectorKernels.VectorBinaryOpKernel)
                    (new Index1D(count), DeviceV3(scope, ba), DeviceV3(scope, bb), ba.Count, bb.Count, 1, o)), ba, bb);
        }

        public GpuBuffer VectorReverse(GpuBuffer vectors)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var format = bv.ElementType;
            V3 origin = format == ElementType.V3F ? PointVectorKernels.VectorReverseElement(bv.Origin) : default;

            if (bv.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format, origin);

            int count = bv.Count;

            if (CanRunOnHost(count, bv))
            {
                var pv = HostV3(bv);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorReverseElement(pv[i]));
                return HostResult(BufferKind.Vectors, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>>)PointVectorKernels.VectorReverseKernel)
                    (new Index1D(count), DeviceV3(scope, bv), o)), bv);
        }

        public GpuBuffer VectorLength(GpuBuffer vectors)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);

            if (bv.Count == 0)
                return CreateEmptyBuffer(BufferKind.Scalars, ElementType.F64);

            int count = bv.Count;

            if (CanRunOnHost(count, bv))
            {
                var pv = HostV3(bv);
                var r = new double[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorLengthElement(pv[i]));
                return HostResult(BufferKind.Scalars, ElementType.F64, default, r);
            }

            return RunDevice(scope => DeviceScalars(scope, ElementType.F64, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<double>>)PointVectorKernels.VectorLengthKernel)
                    (new Index1D(count), DeviceV3(scope, bv), o)), bv);
        }

        public GpuBuffer VectorScale(GpuBuffer vectors, GpuBuffer factors)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var bf = ValidateBuffer(factors, BufferKind.Scalars);
            var format = bv.ElementType;

            if (bv.Count == 0 || bf.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = Math.Max(bv.Count, bf.Count);

            if (CanRunOnHost(count, bv, bf))
            {
                var pv = HostV3(bv);
                var pf = HostF64(bf);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorScaleElement(pv[Rl(i, pv.Length)], pf[Rl(i, pf.Length)]));
                return HostResult(BufferKind.Vectors, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<double>, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<double>, int, int, ArrayView<V3>>)PointVectorKernels.VectorScaleKernel)
                    (new Index1D(count), DeviceV3(scope, bv), DeviceF64(scope, bf), bv.Count, bf.Count, o)), bv, bf);
        }

        public GpuBuffer VectorDivide(GpuBuffer vectors, GpuBuffer divisors)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var bd = ValidateBuffer(divisors, BufferKind.Scalars);
            var format = bv.ElementType;

            if (bv.Count == 0 || bd.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = Math.Max(bv.Count, bd.Count);

            if (CanRunOnHost(count, bv, bd))
            {
                var pv = HostV3(bv);
                var pd = HostF64(bd);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorDivideElement(pv[Rl(i, pv.Length)], pd[Rl(i, pd.Length)]));
                return HostResult(BufferKind.Vectors, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<double>, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<double>, int, int, ArrayView<V3>>)PointVectorKernels.VectorDivideKernel)
                    (new Index1D(count), DeviceV3(scope, bv), DeviceF64(scope, bd), bv.Count, bd.Count, o)), bv, bd);
        }

        public GpuBuffer VectorDotProduct(GpuBuffer a, GpuBuffer b)
        {
            var ba = ValidateBuffer(a, BufferKind.Vectors);
            var bb = ValidateBuffer(b, BufferKind.Vectors);

            if (ba.Count == 0 || bb.Count == 0)
                return CreateEmptyBuffer(BufferKind.Scalars, ElementType.F64);

            int count = Math.Max(ba.Count, bb.Count);

            if (CanRunOnHost(count, ba, bb))
            {
                var pa = HostV3(ba);
                var pb = HostV3(bb);
                var r = new double[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorDotElement(pa[Rl(i, pa.Length)], pb[Rl(i, pb.Length)]));
                return HostResult(BufferKind.Scalars, ElementType.F64, default, r);
            }

            return RunDevice(scope => DeviceScalars(scope, ElementType.F64, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<double>>)PointVectorKernels.VectorDotKernel)
                    (new Index1D(count), DeviceV3(scope, ba), DeviceV3(scope, bb), ba.Count, bb.Count, o)), ba, bb);
        }

        public GpuBuffer VectorCrossProduct(GpuBuffer a, GpuBuffer b)
        {
            var ba = ValidateBuffer(a, BufferKind.Vectors);
            var bb = ValidateBuffer(b, BufferKind.Vectors);
            var format = WidestV3(ba, bb);

            if (ba.Count == 0 || bb.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = Math.Max(ba.Count, bb.Count);

            if (CanRunOnHost(count, ba, bb))
            {
                var pa = HostV3(ba);
                var pb = HostV3(bb);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorCrossElement(pa[Rl(i, pa.Length)], pb[Rl(i, pb.Length)]));
                return HostResult(BufferKind.Vectors, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>)PointVectorKernels.VectorBinaryOpKernel)
                    (new Index1D(count), DeviceV3(scope, ba), DeviceV3(scope, bb), ba.Count, bb.Count, 2, o)), ba, bb);
        }

        public GpuBuffer VectorAngle(GpuBuffer a, GpuBuffer b)
        {
            var ba = ValidateBuffer(a, BufferKind.Vectors);
            var bb = ValidateBuffer(b, BufferKind.Vectors);

            if (ba.Count == 0 || bb.Count == 0)
                return CreateEmptyBuffer(BufferKind.Scalars, ElementType.F64);

            int count = Math.Max(ba.Count, bb.Count);

            if (CanRunOnHost(count, ba, bb))
            {
                var pa = HostV3(ba);
                var pb = HostV3(bb);
                var r = new double[count];
                HostFor(count, i => r[i] = PointVectorKernels.VectorAngleElement(pa[Rl(i, pa.Length)], pb[Rl(i, pb.Length)]));
                return HostResult(BufferKind.Scalars, ElementType.F64, default, r);
            }

            return RunDevice(scope => DeviceScalars(scope, ElementType.F64, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<double>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<double>>)PointVectorKernels.VectorAngleKernel)
                    (new Index1D(count), DeviceV3(scope, ba), DeviceV3(scope, bb), ba.Count, bb.Count, o)), ba, bb);
        }

        // =====================================================================
        // Unit Axis Vectors
        // =====================================================================

        public GpuBuffer UnitX(GpuBuffer factors = null) => UnitAxisInternal(factors, 0);
        public GpuBuffer UnitY(GpuBuffer factors = null) => UnitAxisInternal(factors, 1);
        public GpuBuffer UnitZ(GpuBuffer factors = null) => UnitAxisInternal(factors, 2);

        private GpuBuffer UnitAxisInternal(GpuBuffer factors, int axis)
        {
            var format = _options.DefaultVectorFormat;

            if (factors == null)
            {
                var val = axis == 0 ? new V3(1, 0, 0) : (axis == 1 ? new V3(0, 1, 0) : new V3(0, 0, 1));
                return UploadVectors(new[] { val }, format, Residency.Auto);
            }

            var bf = ValidateBuffer(factors, BufferKind.Scalars);
            if (bf.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = bf.Count;

            if (CanRunOnHost(count, bf))
            {
                var pf = HostF64(bf);
                var r = new V3[count];
                HostFor(count, i =>
                {
                    double s = pf[i];
                    r[i] = axis == 0 ? PointVectorKernels.UnitXElement(s) :
                          (axis == 1 ? PointVectorKernels.UnitYElement(s) : PointVectorKernels.UnitZElement(s));
                });
                return HostResult(BufferKind.Vectors, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<double>, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<double>, int, int, ArrayView<V3>>)PointVectorKernels.UnitAxisKernel)
                    (new Index1D(count), DeviceF64(scope, bf), bf.Count, axis, o)), bf);
        }

        // =====================================================================
        // Point Interpolation & Projection
        // =====================================================================

        public GpuBuffer PointInterpolate(GpuBuffer a, GpuBuffer b, GpuBuffer t)
        {
            var ba = ValidateBuffer(a, BufferKind.Points);
            var bb = ValidateBuffer(b, BufferKind.Points);
            var bt = ValidateBuffer(t, BufferKind.Scalars);
            var format = WidestV3(ba, bb);
            V3 origin = format == ElementType.V3F ? ba.Origin : default;

            if (ba.Count == 0 || bb.Count == 0 || bt.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format, origin);

            int count = Math.Max(ba.Count, Math.Max(bb.Count, bt.Count));

            if (CanRunOnHost(count, ba, bb, bt))
            {
                var pa = HostV3(ba);
                var pb = HostV3(bb);
                var pt = HostF64(bt);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.PointInterpolateElement(
                    pa[Rl(i, pa.Length)],
                    pb[Rl(i, pb.Length)],
                    pt[Rl(i, pt.Length)]));
                return HostResult(BufferKind.Points, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<double>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<double>, int, int, int, ArrayView<V3>>)PointVectorKernels.PointInterpolateKernel)
                    (new Index1D(count), DeviceV3(scope, ba), DeviceV3(scope, bb), DeviceF64(scope, bt), ba.Count, bb.Count, bt.Count, o)), ba, bb, bt);
        }

        public GpuBuffer ProjectVectorToPlane(GpuBuffer vectors, GpuBuffer planeNormals)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var bn = ValidateBuffer(planeNormals, BufferKind.Vectors);
            var format = bv.ElementType;

            if (bv.Count == 0 || bn.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = Math.Max(bv.Count, bn.Count);

            if (CanRunOnHost(count, bv, bn))
            {
                var pv = HostV3(bv);
                var pn = HostV3(bn);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.ProjectVectorToPlaneElement(pv[Rl(i, pv.Length)], pn[Rl(i, pn.Length)]));
                return HostResult(BufferKind.Vectors, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, int, int, ArrayView<V3>>)PointVectorKernels.ProjectVectorToPlaneKernel)
                    (new Index1D(count), DeviceV3(scope, bv), DeviceV3(scope, bn), bv.Count, bn.Count, o)), bv, bn);
        }

        public GpuBuffer PointProject(GpuBuffer points, GpuBuffer planeOrigins, GpuBuffer planeNormals)
        {
            var bp = ValidateBuffer(points, BufferKind.Points);
            var bo = ValidateBuffer(planeOrigins, BufferKind.Points);
            var bn = ValidateBuffer(planeNormals, BufferKind.Vectors);
            var format = WidestV3(bp, bo);
            V3 origin = format == ElementType.V3F ? bp.Origin : default;

            if (bp.Count == 0 || bo.Count == 0 || bn.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format, origin);

            int count = Math.Max(bp.Count, Math.Max(bo.Count, bn.Count));

            if (CanRunOnHost(count, bp, bo, bn))
            {
                var pp = HostV3(bp);
                var po = HostV3(bo);
                var pn = HostV3(bn);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.PointProjectElement(
                    pp[Rl(i, pp.Length)],
                    po[Rl(i, po.Length)],
                    pn[Rl(i, pn.Length)]));
                return HostResult(BufferKind.Points, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>)PointVectorKernels.PointProjectKernel)
                    (new Index1D(count), DeviceV3(scope, bp), DeviceV3(scope, bo), DeviceV3(scope, bn), bp.Count, bo.Count, bn.Count, o)), bp, bo, bn);
        }

        public GpuBuffer RotateVectorAxis(GpuBuffer vectors, GpuBuffer axisVectors, GpuBuffer angles)
        {
            var bv = ValidateBuffer(vectors, BufferKind.Vectors);
            var ba = ValidateBuffer(axisVectors, BufferKind.Vectors);
            var bt = ValidateBuffer(angles, BufferKind.Scalars);
            var format = bv.ElementType;

            if (bv.Count == 0 || ba.Count == 0 || bt.Count == 0)
                return CreateEmptyBuffer(BufferKind.Vectors, format);

            int count = Math.Max(bv.Count, Math.Max(ba.Count, bt.Count));

            if (CanRunOnHost(count, bv, ba, bt))
            {
                var pv = HostV3(bv);
                var pa = HostV3(ba);
                var pt = HostF64(bt);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.RotateVectorAxisElement(
                    pv[Rl(i, pv.Length)],
                    pa[Rl(i, pa.Length)],
                    pt[Rl(i, pt.Length)]));
                return HostResult(BufferKind.Vectors, format, default, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Vectors, format, default, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<double>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<double>, int, int, int, ArrayView<V3>>)PointVectorKernels.RotateVectorAxisKernel)
                    (new Index1D(count), DeviceV3(scope, bv), DeviceV3(scope, ba), DeviceF64(scope, bt), bv.Count, ba.Count, bt.Count, o)), bv, ba, bt);
        }

        // =====================================================================
        // Affine Transforms
        // =====================================================================

        public GpuBuffer PointScale(GpuBuffer points, GpuBuffer centers, GpuBuffer factors)
        {
            var bp = ValidateBuffer(points, BufferKind.Points);
            var bc = ValidateBuffer(centers, BufferKind.Points);
            var bf = ValidateBuffer(factors, BufferKind.Scalars);
            var format = WidestV3(bp, bc);
            V3 origin = format == ElementType.V3F ? bp.Origin : default;

            if (bp.Count == 0 || bc.Count == 0 || bf.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format, origin);

            int count = Math.Max(bp.Count, Math.Max(bc.Count, bf.Count));

            if (CanRunOnHost(count, bp, bc, bf))
            {
                var pp = HostV3(bp);
                var pc = HostV3(bc);
                var pf = HostF64(bf);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.PointScaleElement(
                    pp[Rl(i, pp.Length)],
                    pc[Rl(i, pc.Length)],
                    pf[Rl(i, pf.Length)]));
                return HostResult(BufferKind.Points, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<double>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<double>, int, int, int, ArrayView<V3>>)PointVectorKernels.PointScaleKernel)
                    (new Index1D(count), DeviceV3(scope, bp), DeviceV3(scope, bc), DeviceF64(scope, bf), bp.Count, bc.Count, bf.Count, o)), bp, bc, bf);
        }

        public GpuBuffer PointRotateAxis(GpuBuffer points, GpuBuffer centers, GpuBuffer axisVectors, GpuBuffer angles)
        {
            var bp = ValidateBuffer(points, BufferKind.Points);
            var bc = ValidateBuffer(centers, BufferKind.Points);
            var ba = ValidateBuffer(axisVectors, BufferKind.Vectors);
            var bt = ValidateBuffer(angles, BufferKind.Scalars);
            var format = WidestV3(bp, bc);
            V3 origin = format == ElementType.V3F ? bp.Origin : default;

            if (bp.Count == 0 || bc.Count == 0 || ba.Count == 0 || bt.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format, origin);

            int count = Math.Max(Math.Max(bp.Count, bc.Count), Math.Max(ba.Count, bt.Count));

            if (CanRunOnHost(count, bp, bc, ba, bt))
            {
                var pp = HostV3(bp);
                var pc = HostV3(bc);
                var pa = HostV3(ba);
                var pt = HostF64(bt);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.PointRotateAxisElement(
                    pp[Rl(i, pp.Length)],
                    pc[Rl(i, pc.Length)],
                    pa[Rl(i, pa.Length)],
                    pt[Rl(i, pt.Length)]));
                return HostResult(BufferKind.Points, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, ArrayView<double>, int, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, ArrayView<double>, int, int, int, int, ArrayView<V3>>)PointVectorKernels.PointRotateAxisKernel)
                    (new Index1D(count), DeviceV3(scope, bp), DeviceV3(scope, bc), DeviceV3(scope, ba), DeviceF64(scope, bt), bp.Count, bc.Count, ba.Count, bt.Count, o)), bp, bc, ba, bt);
        }

        public GpuBuffer PointMirror(GpuBuffer points, GpuBuffer planeOrigins, GpuBuffer planeNormals)
        {
            var bp = ValidateBuffer(points, BufferKind.Points);
            var bo = ValidateBuffer(planeOrigins, BufferKind.Points);
            var bn = ValidateBuffer(planeNormals, BufferKind.Vectors);
            var format = WidestV3(bp, bo);
            V3 origin = format == ElementType.V3F ? bp.Origin : default;

            if (bp.Count == 0 || bo.Count == 0 || bn.Count == 0)
                return CreateEmptyBuffer(BufferKind.Points, format, origin);

            int count = Math.Max(bp.Count, Math.Max(bo.Count, bn.Count));

            if (CanRunOnHost(count, bp, bo, bn))
            {
                var pp = HostV3(bp);
                var po = HostV3(bo);
                var pn = HostV3(bn);
                var r = new V3[count];
                HostFor(count, i => r[i] = PointVectorKernels.PointMirrorElement(
                    pp[Rl(i, pp.Length)],
                    po[Rl(i, po.Length)],
                    pn[Rl(i, pn.Length)]));
                return HostResult(BufferKind.Points, format, origin, r);
            }

            return RunDevice(scope => DeviceVectors(scope, BufferKind.Points, format, origin, count, o =>
                GetKernel<Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>>(
                    (Action<Index1D, ArrayView<V3>, ArrayView<V3>, ArrayView<V3>, int, int, int, ArrayView<V3>>)PointVectorKernels.PointMirrorKernel)
                    (new Index1D(count), DeviceV3(scope, bp), DeviceV3(scope, bo), DeviceV3(scope, bn), bp.Count, bo.Count, bn.Count, o)), bp, bo, bn);
        }
    }
}
