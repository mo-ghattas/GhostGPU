using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GhostGPU.Core;
using GhostGPU.Runtime;
using GhostGPU.Validation.Models;
using GhostGPU.Validation.Reference;

namespace GhostGPU.Validation.Tests
{
    /// <summary>
    /// v2 runtime checks that the earlier suites could not catch: compact formats on every op and both
    /// paths, host/device bit equality, chunked staging on the real backend, grid ties, NaN/Inf bounds,
    /// streaming, extension-API validation and lock-order safety under concurrency.
    /// </summary>
    public static class V2FormatAndConcurrencyTests
    {
        private const string Category = "RuntimeV2";

        public static void RunAll(IGpuSession session, Action<TestResultItem> record)
        {
            foreach (var res in new[] { Residency.Host, Residency.Device })
            {
                Run($"V2_CompactFormats_AllOps_Vs_F64_{res}", record, () => CompactFormatsVsF64(session, res));
                Run($"V2_OutputFormat_Rules_{res}", record, () => OutputFormatRules(session, res));
            }
            foreach (var (vf, sf) in new[] { (ElementType.V3D, ElementType.F64), (ElementType.V3F, ElementType.F32) })
                Run($"V2_Host_Vs_Device_0ULP_AllOps_{vf}_{sf}", record, () => HostVsDeviceZeroUlp(session, vf, sf));
            Run("V2_Map_Sine_Accuracy_Vs_MathSin", record, () => SineAccuracy(session));
            Run("V2_U16_Indices_Device_RoundTrip_And_View_Validation", record, () => U16Indices(session));
            Run("V2_Staging_Chunking_Backend", record, () => StagingChunking(session));
            Run("V2_Grid_Ties_Nonzero_Distance_Exact_Index", record, () => GridTies(session));
            Run("V2_Bounds_NaN_Inf_Vs_Serial_Oracle", record, () => BoundsNaNInf(session));
            Run("V2_Streaming_ClosestPoints_V3F_And_V3D", record, () => StreamingClosestPoints(session));
            Run("V2_Pin_Promotes_And_Blocks_Spill", record, () => PinBlocksSpill(session));
            Run("V2_Concurrent_Rent_Promote_Spill_No_Deadlock", record, () => ConcurrentRentPromote(session));
        }

        private static void Run(string name, Action<TestResultItem> record, Func<string> body)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                string details = body();
                record(new TestResultItem { Name = name, Category = Category, Outcome = TestOutcome.Pass, DurationMs = sw.Elapsed.TotalMilliseconds, Details = details });
            }
            catch (Exception ex)
            {
                record(new TestResultItem { Name = name, Category = Category, Outcome = TestOutcome.Fail, DurationMs = sw.Elapsed.TotalMilliseconds, Message = ex.Message, Details = ex.ToString() });
            }
        }

        private static GpuBackend BackendOf(IGpuSession s) => s.IsHardwareAccelerated ? GpuBackend.CudaOnly : GpuBackend.CpuDebug;

        // ------------------------------------------------------------------ data

        private sealed class Inputs
        {
            public V3[] A = Array.Empty<V3>(), B = Array.Empty<V3>(), Vec = Array.Empty<V3>();
            public double[] Sa = Array.Empty<double>(), Sb = Array.Empty<double>(), Mag = Array.Empty<double>(), Unit = Array.Empty<double>();
        }

        private static Inputs MakeInputs(int n, int seed)
        {
            var r = new Random(seed);
            var d = new Inputs
            {
                A = new V3[n], B = new V3[n], Vec = new V3[n],
                Sa = new double[n], Sb = new double[n], Mag = new double[n], Unit = new double[n]
            };
            for (int i = 0; i < n; i++)
            {
                // Large-ish world coordinates so V3F origin shifting matters.
                d.A[i] = new V3(1000 + r.NextDouble() * 200, -500 + r.NextDouble() * 200, 30 + r.NextDouble() * 200);
                d.B[i] = new V3(1000 + r.NextDouble() * 200, -500 + r.NextDouble() * 200, 30 + r.NextDouble() * 200);
                d.Vec[i] = new V3(r.NextDouble() * 20 - 10, r.NextDouble() * 20 - 10, r.NextDouble() * 20 - 10);
                d.Sa[i] = r.NextDouble() * 200 - 100;
                d.Sb[i] = (r.NextDouble() * 9 + 1) * (r.Next(2) == 0 ? -1 : 1); // |b| >= 1: no blow-up in tolerance checks
                d.Mag[i] = r.NextDouble() * 50 - 25;
                d.Unit[i] = r.NextDouble() * 1.4 - 0.2;
            }
            // No exact zero vectors here: V3F stores offsets from the bbox centre, so (0,0,0) comes back as a
            // ~1e-7 vector and Unitize of it is ill-conditioned. Zero-vector semantics are covered on V3D in MathVectorTests.
            return d;
        }

        private sealed class Bufs : IDisposable
        {
            public GpuBuffer A = null!, B = null!, Vec = null!, Sa = null!, Sb = null!, Mag = null!, Unit = null!;
            public void Dispose() { A.Dispose(); B.Dispose(); Vec.Dispose(); Sa.Dispose(); Sb.Dispose(); Mag.Dispose(); Unit.Dispose(); }
        }

        private static Bufs Upload(IGpuSession s, Inputs d, ElementType vf, ElementType sf, Residency res) => new Bufs
        {
            A = s.UploadPoints(d.A, vf, res),
            B = s.UploadPoints(d.B, vf, res),
            Vec = s.UploadVectors(d.Vec, vf, res),
            Sa = s.UploadScalars(d.Sa, sf, res),
            Sb = s.UploadScalars(d.Sb, sf, res),
            Mag = s.UploadScalars(d.Mag, sf, res),
            Unit = s.UploadScalars(d.Unit, sf, res),
        };

        private static readonly (string Name, Func<IGpuSession, Bufs, GpuBuffer> Op)[] Ops =
        {
            ("Distance", (s, b) => s.Distance(b.A, b.B)),
            ("VectorBetween", (s, b) => s.VectorBetween(b.A, b.B)),
            ("Unitize", (s, b) => s.Unitize(b.Vec)),
            ("Amplitude", (s, b) => s.Amplitude(b.Vec, b.Mag)),
            ("Move", (s, b) => s.Move(b.A, b.Vec)),
            ("Add", (s, b) => s.ScalarMath(b.Sa, b.Sb, ScalarOperation.Add)),
            ("Subtract", (s, b) => s.ScalarMath(b.Sa, b.Sb, ScalarOperation.Subtract)),
            ("Multiply", (s, b) => s.ScalarMath(b.Sa, b.Sb, ScalarOperation.Multiply)),
            ("Divide", (s, b) => s.ScalarMath(b.Sa, b.Sb, ScalarOperation.Divide)),
            ("Bounds", (s, b) => s.Bounds(b.Sa)),
            ("Remap", (s, b) => { using var bn = s.Bounds(b.Sa); return s.Remap(b.Sa, bn, 0, 1, true); }),
            ("MapLinear", (s, b) => s.Map(b.Unit, MapFunction.Linear)),
            ("MapSmoothStep", (s, b) => s.Map(b.Unit, MapFunction.SmoothStep)),
            ("MapSine", (s, b) => s.Map(b.Sa, MapFunction.Sine)),
            ("ColourRamp", (s, b) => s.ColourRamp(b.Unit, 0xFF102030u, 0xFFF0E0D0u)),
            ("ClosestPoints.Points", (s, b) => { var r = s.ClosestPoints(b.A, b.B); r.Indices.Dispose(); r.Distances.Dispose(); return r.Points; }),
            ("ClosestPoints.Indices", (s, b) => { var r = s.ClosestPoints(b.A, b.B); r.Points.Dispose(); r.Distances.Dispose(); return r.Indices; }),
            ("ClosestPoints.Distances", (s, b) => { var r = s.ClosestPoints(b.A, b.B); r.Points.Dispose(); r.Indices.Dispose(); return r.Distances; }),
        };

        /// <summary>Downloads any buffer as a flat double array (V3 -> xyz, colours -> uint, indices -> int).</summary>
        private static double[] Flat(IGpuSession s, GpuBuffer b)
        {
            switch (b.Kind)
            {
                case BufferKind.Points: return s.DownloadPoints(b).SelectMany(p => new[] { p.X, p.Y, p.Z }).ToArray();
                case BufferKind.Vectors: return s.DownloadVectors(b).SelectMany(p => new[] { p.X, p.Y, p.Z }).ToArray();
                case BufferKind.Scalars: return s.DownloadScalars(b);
                case BufferKind.Indices: return s.DownloadIndices(b).Select(i => (double)i).ToArray();
                case BufferKind.Colours: return s.DownloadColours(b).Select(c => (double)c).ToArray();
                default: throw new InvalidOperationException(b.Kind.ToString());
            }
        }

        private static void ExpectResidency(GpuBuffer b, Residency res, string what)
        {
            // Host inputs below HostWorkThreshold must stay on the host path; device inputs must stay on the device.
            if (b.Residency != res)
                throw new Exception($"{what}: expected {res}-path result, got {b.Residency}");
        }

        // ------------------------------------------------------------------ tests

        private static string CompactFormatsVsF64(IGpuSession s, Residency res)
        {
            // Host: 250 x 250 stays on the brute host path. Device: 301 x 301 takes the grid path.
            var d = MakeInputs(res == Residency.Host ? 250 : 301, 7);
            using var wide = Upload(s, d, ElementType.V3D, ElementType.F64, res);
            using var compact = Upload(s, d, ElementType.V3F, ElementType.F32, res);

            foreach (var (name, op) in Ops)
            {
                using var w = op(s, wide);
                using var c = op(s, compact);
                ExpectResidency(c, res, name);
                var fw = Flat(s, w);
                var fc = Flat(s, c);
                if (fw.Length != fc.Length) throw new Exception($"{name}: length {fc.Length} vs {fw.Length}");

                for (int i = 0; i < fw.Length; i++)
                {
                    double tol;
                    if (c.Kind == BufferKind.Colours)
                    {
                        uint x = (uint)fw[i], y = (uint)fc[i];
                        for (int sh = 0; sh < 32; sh += 8)
                            if (Math.Abs((int)((x >> sh) & 0xFF) - (int)((y >> sh) & 0xFF)) > 1)
                                throw new Exception($"{name}[{i}]: colour {y:X8} vs {x:X8}");
                        continue;
                    }
                    if (c.Kind == BufferKind.Indices) continue; // near-ties may legitimately flip under float storage; distances checked below
                    if (c.Kind == BufferKind.Points || c.Kind == BufferKind.Vectors || name.StartsWith("Distance") || name.StartsWith("ClosestPoints"))
                        tol = 1e-4 * (1 + Math.Abs(fw[i]) / 1000); // V3F: float offsets over a ~200 extent
                    else
                        tol = 2e-5 * (1 + Math.Abs(fw[i])); // F32 storage: ~6e-8 relative per value, amplified by sine/remap
                    AssertHelper.AssertClose(fc[i], fw[i], tol, $"{name}[{i}] {res} V3F/F32 vs V3D/F64");
                }
            }
            return $"All {Ops.Length} op outputs on V3F/F32 inputs ({res} path) match V3D/F64 within storage tolerance.";
        }

        private static string OutputFormatRules(IGpuSession s, Residency res)
        {
            var d = MakeInputs(40, 3);
            using var f = Upload(s, d, ElementType.V3F, ElementType.F32, res);
            using var w = Upload(s, d, ElementType.V3D, ElementType.F64, res);

            void Expect(GpuBuffer b, ElementType t, string what)
            {
                using (b)
                {
                    if (b.ElementType != t) throw new Exception($"{what}: ElementType {b.ElementType}, expected {t}");
                    if (b.ByteSize != (long)b.Count * ElementTypes.SizeOf(t)) throw new Exception($"{what}: ByteSize mismatch");
                    Flat(s, b); // payload type must match ElementType (InvalidCast otherwise)
                }
            }

            Expect(s.Distance(f.A, f.B), ElementType.F64, "Distance(V3F,V3F)");
            Expect(s.VectorBetween(f.A, f.B), ElementType.V3F, "VectorBetween(V3F,V3F)");
            Expect(s.VectorBetween(f.A, w.B), ElementType.V3D, "VectorBetween(V3F,V3D)");
            Expect(s.Move(w.A, f.Vec), ElementType.V3D, "Move(V3D,V3F)");
            Expect(s.Move(f.A, f.Vec), ElementType.V3F, "Move(V3F,V3F)");
            Expect(s.Unitize(f.Vec), ElementType.V3F, "Unitize(V3F)");
            Expect(s.Amplitude(f.Vec, w.Mag), ElementType.V3F, "Amplitude(V3F,F64)");
            Expect(s.Amplitude(w.Vec, f.Mag), ElementType.V3D, "Amplitude(V3D,F32)");
            Expect(s.ScalarMath(f.Sa, f.Sb, ScalarOperation.Add), ElementType.F32, "ScalarMath(F32,F32)");
            Expect(s.ScalarMath(f.Sa, w.Sb, ScalarOperation.Add), ElementType.F64, "ScalarMath(F32,F64)");
            Expect(s.Bounds(f.Sa), ElementType.F32, "Bounds(F32)");
            Expect(s.Map(f.Sa, MapFunction.Sine), ElementType.F32, "Map(F32)");
            using (var bf = s.Bounds(f.Sa))
            using (var bw = s.Bounds(w.Sa))
            {
                Expect(s.Remap(f.Sa, bf, 0, 1, false), ElementType.F32, "Remap(F32,F32)");
                Expect(s.Remap(f.Sa, bw, 0, 1, false), ElementType.F64, "Remap(F32,F64 bounds)");
            }
            Expect(s.ColourRamp(f.Unit, 0, 0xFFFFFFFF), ElementType.U32, "ColourRamp(F32)");
            using (var r = s.ClosestPoints(f.A, f.B))
            {
                if (r.Points.ElementType != ElementType.V3F || r.Indices.ElementType != ElementType.I32 || r.Distances.ElementType != ElementType.F64)
                    throw new Exception($"ClosestPoints(V3F,V3F) formats {r.Points.ElementType}/{r.Indices.ElementType}/{r.Distances.ElementType}");
                Flat(s, r.Points);
            }
            using (var r = s.ClosestPoints(f.A, w.B))
                if (r.Points.ElementType != ElementType.V3D) throw new Exception("ClosestPoints(V3F,V3D) points must be V3D");
            return "Output ElementType follows contract section 1 and payload type matches it.";
        }

        private static string HostVsDeviceZeroUlp(IGpuSession s, ElementType vf, ElementType sf)
        {
            var d = MakeInputs(255, 11); // 255^2 <= 65536: ClosestPoints uses brute on both paths
            using var host = Upload(s, d, vf, sf, Residency.Host);
            using var dev = Upload(s, d, vf, sf, Residency.Device);
            var report = new List<string>();

            foreach (var (name, op) in Ops)
            {
                using var h = op(s, host);
                using var g = op(s, dev);
                ExpectResidency(h, Residency.Host, name + " host");
                ExpectResidency(g, Residency.Device, name + " device");
                if (h.ElementType != g.ElementType) throw new Exception($"{name}: host {h.ElementType} vs device {g.ElementType}");
                var fh = Flat(s, h);
                var fg = Flat(s, g);
                for (int i = 0; i < fh.Length; i++)
                {
                    long bh = BitConverter.DoubleToInt64Bits(fh[i]);
                    long bg = BitConverter.DoubleToInt64Bits(fg[i]);
                    if (bh != bg)
                        throw new Exception($"{name}[{i}] ({vf}/{sf}): host {fh[i]:R} != device {fg[i]:R} ({Math.Abs(bh - bg)} ulp)");
                }
                report.Add(name);
            }
            return $"0-ULP host/device equality for {report.Count} op outputs ({vf}/{sf}).";
        }

        private static string SineAccuracy(IGpuSession s)
        {
            var r = new Random(23);
            var xs = new List<double> { 0, -0.0, 1e-300, 1e-8, Math.PI / 4, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI, 710.5, -12345.678, 1e5, 1.5e6 };
            for (int i = 0; i < 20000; i++) xs.Add((r.NextDouble() - 0.5) * Math.Pow(10, r.Next(-6, 7)));
            var x = xs.ToArray();
            double worst = 0;
            double[]? hostY = null;
            foreach (var res in new[] { Residency.Host, Residency.Device })
            {
                using var b = s.UploadScalars(x, ElementType.F64, res);
                using var m = s.Map(b, MapFunction.Sine);
                var y = s.DownloadScalars(m);
                if (hostY == null) hostY = y;
                else
                    for (int i = 0; i < x.Length; i++)
                        if (BitConverter.DoubleToInt64Bits(hostY[i]) != BitConverter.DoubleToInt64Bits(y[i]))
                            throw new Exception($"Sine({x[i]:R}): host {hostY[i]:R} != device {y[i]:R}");
                for (int i = 0; i < x.Length; i++)
                {
                    double e = Math.Sin(x[i]);
                    double ulp = Math.Max(Math.BitIncrement(Math.Abs(e)) - Math.Abs(e), double.Epsilon);
                    double err = Math.Abs(y[i] - e) / ulp;
                    worst = Math.Max(worst, err);
                    if (err > 2) throw new Exception($"Sine({x[i]:R}) {res} = {y[i]:R}, Math.Sin = {e:R} ({err:F1} ulp)");
                }
            }
            return $"Shared sine within {worst:F2} ulp of Math.Sin over {x.Length} arguments up to 1.5e6, both paths.";
        }

        private static string U16Indices(IGpuSession s)
        {
            int n = 70000; // >= HostWorkThreshold: Auto goes to the device
            var idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = (i * 7919) % 65536;
            using var b = s.UploadIndices(idx);
            if (b.ElementType != ElementType.U16) throw new Exception($"Expected U16, got {b.ElementType}");
            if (b.Residency != Residency.Device) throw new Exception($"Expected device residency for {n} indices, got {b.Residency}");
            var back = s.DownloadIndices(b);
            for (int i = 0; i < n; i++)
                if (back[i] != idx[i]) throw new Exception($"U16 device roundtrip mismatch at {i}: {back[i]} vs {idx[i]}");

            var gs = (GpuSession)s;
            var view = gs.DeviceView<ushort>(b);
            if (view.Length != n) throw new Exception($"DeviceView<ushort> length {view.Length}");
            bool threw = false;
            try { gs.DeviceView<int>(b); } catch (ArgumentException) { threw = true; }
            if (!threw) throw new Exception("DeviceView<int> on a U16 buffer must throw (element size mismatch).");

            using var v3f = s.UploadPoints(new[] { new V3(1, 2, 3), new V3(4, 5, 6) }, ElementType.V3F, Residency.Host);
            threw = false;
            try { gs.DeviceView<V3>(v3f); } catch (ArgumentException) { threw = true; }
            if (!threw) throw new Exception("DeviceView<V3> on a V3F buffer must throw (24 vs 12 bytes).");
            var fv = gs.DeviceView<V3F>(v3f);
            if (fv.Length != 2 || v3f.Residency != Residency.Device) throw new Exception("DeviceView<V3F> must promote and expose 2 elements.");
            return "U16 indices roundtrip through the device; DeviceView rejects mismatched element sizes and promotes host buffers.";
        }

        private static string StagingChunking(IGpuSession main)
        {
            var opt = new SessionOptions
            {
                Backend = BackendOf(main),
                StagingSlots = 2,
                StagingSlotBytes = 64 * 1024,
                MemoryLimitBytes = 64L * 1024 * 1024
            };
            using var s = SessionFactory.CreateSession(opt);

            int count = 43691; // 1,048,584 B of V3D = 16+ slots, last chunk partial
            var pts = new V3[count];
            for (int i = 0; i < count; i++) pts[i] = new V3(i, i * 0.5, -i);

            using var buf = s.UploadPoints(pts, ElementType.V3D, Residency.Device);
            long staged = s.Metrics.StagedBytes;
            if (staged != (long)count * 24)
                throw new Exception($"Upload StagedBytes {staged}, expected {(long)count * 24}");

            using var f32 = s.UploadPoints(pts, ElementType.V3F, Residency.Device); // 12 B elements, chunk boundary mid-element
            using var one = s.UploadVectors(new[] { new V3(1, 1, 1) }, ElementType.V3D, Residency.Device);
            using var moved = s.Move(buf, one);

            var down = s.DownloadPoints(buf);
            var downF = s.DownloadPoints(f32);
            var downM = s.DownloadPoints(moved);
            for (int i = 0; i < count; i++)
            {
                if (down[i].X != pts[i].X || down[i].Y != pts[i].Y || down[i].Z != pts[i].Z)
                    throw new Exception($"Chunked V3D roundtrip mismatch at {i}");
                if (downM[i].X != pts[i].X + 1 || downM[i].Z != pts[i].Z + 1)
                    throw new Exception($"Kernel read of chunked upload wrong at {i} (upload not fenced?)");
                AssertHelper.AssertClose(downF[i], pts[i], 5e-3, $"Chunked V3F roundtrip {i}");
            }
            if (s.Metrics.StagedBytes < staged + (long)count * (12 + 24 + 12 + 24))
                throw new Exception($"Downloads did not go through the ring: StagedBytes {s.Metrics.StagedBytes}");
            return $"{count * 24} B through 64 KiB slots on {s.DeviceName}: exact roundtrip, fenced kernel reads.";
        }

        private static string GridTies(IGpuSession s)
        {
            // 1000 random targets live in x in [60,100]; tie queries sit in x in [0,40] so nothing random is closer.
            var r = new Random(99);
            int tCount = 1000, qCount = 200;
            var targets = new V3[tCount];
            for (int i = 0; i < tCount; i++) targets[i] = new V3(60 + r.NextDouble() * 40, r.NextDouble() * 100, r.NextDouble() * 100);
            var queries = new V3[qCount];
            for (int i = 0; i < qCount; i++) queries[i] = new V3(70 + r.NextDouble() * 20, r.NextDouble() * 100, r.NextDouble() * 100);

            // Exact ties at distance 2 (and 1.5): lower index placed at the far side so grid search order meets the higher index first.
            var ties = new List<(int q, int lo, int hi, double dist)>();
            int slot = 0;
            for (int k = 0; k < 20; k++)
            {
                var c = new V3(5 + k * 1.75, 10 + k * 4, 50 - k);
                double dd = k % 2 == 0 ? 2.0 : 1.5;
                int lo = 3 + slot * 2;         // low indices
                int hi = tCount - 1 - slot * 2; // high indices
                slot++;
                targets[hi] = new V3(c.X + dd, c.Y, c.Z);
                targets[lo] = new V3(c.X - dd, c.Y, c.Z);
                if (k % 3 == 0) targets[lo] = new V3(c.X, c.Y, c.Z - dd); // tie along another axis
                queries[k] = c;
                ties.Add((k, lo, hi, dd));
            }

            using var qb = s.UploadPoints(queries, ElementType.V3D, Residency.Device);
            using var tb = s.UploadPoints(targets, ElementType.V3D, Residency.Device);
            using var res = s.ClosestPoints(qb, tb); // 200k pairs, 1000 targets -> grid path
            var gi = s.DownloadIndices(res.Indices);
            var gd = s.DownloadScalars(res.Distances);
            var reference = NaiveCpuReference.ClosestPoints(queries, targets);

            foreach (var (q, lo, hi, dd) in ties)
            {
                if (gd[q] != dd) throw new Exception($"Tie query {q}: distance {gd[q]:R}, expected exactly {dd}");
                if (gi[q] != reference.Indices[q] || gi[q] != lo)
                    throw new Exception($"Tie query {q}: gpu index {gi[q]}, reference {reference.Indices[q]}, expected lowest {lo} (other {hi})");
            }
            for (int i = 0; i < qCount; i++)
            {
                if (gi[i] != reference.Indices[i]) throw new Exception($"Query {i}: gpu index {gi[i]} != reference {reference.Indices[i]}");
                AssertHelper.AssertClose(gd[i], reference.Distances[i], 1e-9, $"Grid distance {i}");
            }
            return $"{ties.Count} exact nonzero-distance ties resolved to the lowest index on the grid path; all {qCount} indices equal the reference.";
        }

        /// <summary>v1 serial bounds rule (same as ReductionKernels.SerialBoundsKernel).</summary>
        private static double[] SerialBounds(double[] v)
        {
            double min = double.PositiveInfinity, max = double.NegativeInfinity;
            foreach (var x in v)
            {
                if (double.IsNaN(x)) continue;
                if (x < min) min = x;
                if (x > max) max = x;
            }
            if (double.IsPositiveInfinity(min)) min = double.NaN;
            if (double.IsNegativeInfinity(max)) max = double.NaN;
            return new[] { min, max };
        }

        private static string BoundsNaNInf(IGpuSession s)
        {
            // Uploads reject non-finite values, so NaN/Inf come from ScalarMath divide: a / 0.
            var cases = new List<(string name, double[] num, double[] den)>();
            var r = new Random(5);
            foreach (int n in new[] { 1, 3, 300, 5000, 100003 })
            {
                var num = new double[n]; var den = new double[n];
                for (int i = 0; i < n; i++) { num[i] = r.NextDouble() * 1000 - 500; den[i] = 1; }
                for (int i = 0; i < n; i += 7) { num[i] = 0; den[i] = 0; }       // NaN
                for (int i = 3; i < n; i += 97) { num[i] = 1; den[i] = 0; }      // +Inf
                for (int i = 5; i < n; i += 131) { num[i] = -1; den[i] = 0; }    // -Inf
                cases.Add(($"mixed{n}", num, den));
                cases.Add(($"allNaN{n}", new double[n], new double[n]));
                cases.Add(($"posInfOnly{n}", Enumerable.Repeat(1.0, n).ToArray(), new double[n]));
                cases.Add(($"negInfAndFinite{n}", Enumerable.Range(0, n).Select(i => i % 2 == 0 ? -1.0 : i).ToArray(), Enumerable.Range(0, n).Select(i => i % 2 == 0 ? 0.0 : 1.0).ToArray()));
            }

            int checkedCount = 0;
            foreach (var (name, num, den) in cases)
            {
                var expectedInput = num.Select((x, i) => x / den[i]).ToArray();
                var oracle = SerialBounds(expectedInput);
                foreach (var res in new[] { Residency.Host, Residency.Device })
                foreach (var sf in new[] { ElementType.F64, ElementType.F32 })
                {
                    using var a = s.UploadScalars(num, sf, res);
                    using var b = s.UploadScalars(den, sf, res);
                    using var q = s.ScalarMath(a, b, ScalarOperation.Divide);
                    using var bnd = s.Bounds(q);
                    var got = s.DownloadScalars(bnd);
                    var exp = sf == ElementType.F32 ? SerialBounds(s.DownloadScalars(q)) : oracle;
                    for (int k = 0; k < 2; k++)
                        if (BitConverter.DoubleToInt64Bits(got[k]) != BitConverter.DoubleToInt64Bits(exp[k]) && !(double.IsNaN(got[k]) && double.IsNaN(exp[k])))
                            throw new Exception($"Bounds {name} {res} {sf} [{k}]: got {got[k]:R}, serial oracle {exp[k]:R}");
                    checkedCount++;
                }
            }
            using (var empty = s.UploadScalars(Array.Empty<double>()))
            using (var eb = s.Bounds(empty))
            {
                var e = s.DownloadScalars(eb);
                if (e.Length != 2 || !double.IsNaN(e[0]) || !double.IsNaN(e[1])) throw new Exception("Bounds(empty) must be [NaN, NaN].");
            }
            return $"{checkedCount} NaN/Inf bounds cases (host+device, F64+F32, up to 100003 values) match the v1 serial oracle.";
        }

        private static string StreamingClosestPoints(IGpuSession main)
        {
            var opt = new SessionOptions { Backend = BackendOf(main), StagingSlotBytes = 64 * 1024, MemoryLimitBytes = 64L * 1024 * 1024 };
            using var s = SessionFactory.CreateSession(opt);
            var r = new Random(17);
            int qn = 150001, tn = 300;
            var q = new V3[qn];
            for (int i = 0; i < qn; i++) q[i] = new V3(500 + r.NextDouble() * 100, r.NextDouble() * 100, r.NextDouble() * 100);
            var t = new V3[tn];
            for (int i = 0; i < tn; i++) t[i] = new V3(500 + r.NextDouble() * 100, r.NextDouble() * 100, r.NextDouble() * 100);

            foreach (var fmt in new[] { ElementType.V3D, ElementType.V3F })
            {
                using var qb = s.UploadPoints(q, fmt, Residency.Host);
                using var tb = s.UploadPoints(t, fmt, Residency.Auto);

                using var res = s.ClosestPoints(qb, tb);
                if (qb.Residency != Residency.Host) throw new Exception($"{fmt}: streamed queries were promoted to the device");
                if (res.Points.ElementType != fmt) throw new Exception($"{fmt}: output points {res.Points.ElementType}");
                if (s.Metrics.PeakBytes >= (long)qn * ElementTypes.SizeOf(fmt))
                    throw new Exception($"{fmt}: device peak {s.Metrics.PeakBytes} B >= full query payload; queries were not streamed");

                var qw = s.DownloadPoints(qb); // stored (possibly V3F-rounded) coordinates
                var tw = s.DownloadPoints(tb);
                var reference = NaiveCpuReference.ClosestPoints(qw, tw);
                var gi = s.DownloadIndices(res.Indices);
                var gd = s.DownloadScalars(res.Distances);
                var gp = s.DownloadPoints(res.Points);
                for (int i = 0; i < qn; i++)
                {
                    AssertHelper.AssertClose(gd[i], reference.Distances[i], 1e-9, $"{fmt} streamed distance {i}");
                    if (gi[i] != reference.Indices[i] && Math.Abs(gd[i] - reference.Distances[i]) > 1e-12)
                        throw new Exception($"{fmt} streamed index {i}: {gi[i]} vs {reference.Indices[i]}");
                    AssertHelper.AssertClose(gp[i], tw[gi[i]], fmt == ElementType.V3F ? 1e-4 : 0, $"{fmt} streamed point {i}");
                }
            }
            return $"{qn} host-resident queries streamed in 64 KiB chunks against {tn} grid targets (V3D and V3F) match the reference.";
        }

        private static string PinBlocksSpill(IGpuSession main)
        {
            var opt = new SessionOptions { Backend = BackendOf(main), MemoryLimitBytes = 3072, PoolCacheBytes = 0, Spill = SpillPolicy.SpillToHost };
            using var s = (GpuSession)SessionFactory.CreateSession(opt);
            var pts = Enumerable.Range(0, 40).Select(i => new V3(i, 2 * i, 3 * i)).ToArray(); // 960 B -> 1 KiB class
            using var a = s.UploadPoints(pts, ElementType.V3D, Residency.Host);
            using var b = s.UploadPoints(pts, ElementType.V3D, Residency.Device);
            using (s.Pin(a))
            {
                if (a.Residency != Residency.Device) throw new Exception("Pin must promote a host buffer to the device.");
                // 2 KiB more: budget 3 KiB, live 2 KiB -> must spill b (unpinned), never a.
                using var lease = s.Rent<double>(256);
                if (a.Residency != Residency.Device) throw new Exception("Pinned buffer was spilled.");
                if (b.Residency != Residency.Host) throw new Exception($"Unpinned buffer should have spilled, residency {b.Residency}");
                bool threw = false;
                try { using var l2 = s.Rent<double>(256); } catch (InvalidOperationException) { threw = true; }
                if (!threw) throw new Exception("Budget exhausted by pinned buffer + lease should throw, not spill the pin.");
            }
            var back = s.DownloadPoints(b);
            if (back[39].Z != 117) throw new Exception("Spilled buffer content changed.");
            return "Pin promotes, pinned buffers are never spilled, unpinned LRU buffers are.";
        }

        private static string ConcurrentRentPromote(IGpuSession main)
        {
            // Tight budget: every Pin promotes (Rent under the storage lock) and every Rent may spill other
            // storages (needs their locks). The old Promote(storage->pool) vs Rent(pool->spill->storage) order
            // deadlocked here; a bounded wait turns a deadlock into a failure instead of a hang.
            var opt = new SessionOptions { Backend = BackendOf(main), MemoryLimitBytes = 48 * 1024, PoolCacheBytes = 8 * 1024, Spill = SpillPolicy.SpillToHost };
            var s = (GpuSession)SessionFactory.CreateSession(opt);
            int threads = 8, iterations = 150;
            long budgetFailures = 0, ops = 0;
            var errors = new System.Collections.Concurrent.ConcurrentQueue<string>();

            var shared = Enumerable.Range(0, 6).Select(k =>
                s.UploadPoints(Enumerable.Range(0, 100).Select(i => new V3(k, i, -i)).ToArray(), ElementType.V3D, Residency.Device)).ToArray();

            var tasks = Enumerable.Range(0, threads).Select(tid => Task.Factory.StartNew(() =>
            {
                var rnd = new Random(tid);
                for (int it = 0; it < iterations; it++)
                {
                    try
                    {
                        var mine = Enumerable.Range(0, 50 + rnd.Next(100)).Select(i => new V3(tid, it, i)).ToArray();
                        using var hb = s.UploadPoints(mine, ElementType.V3D, Residency.Host);
                        var sb = shared[rnd.Next(shared.Length)];
                        using (s.Pin(hb))
                        using (s.Pin(sb))
                        using (var lease = s.Rent<double>(64 + rnd.Next(512)))
                        {
                            var v = s.DownloadPoints(hb);
                            if (v[^1].Z != mine[^1].Z || v[0].X != tid) errors.Enqueue($"t{tid} it{it}: pinned content wrong");
                        }
                        using (var d = s.Distance(hb, sb))
                        {
                            var dv = s.DownloadScalars(d);
                            if (dv.Length != Math.Max(mine.Length, 100)) errors.Enqueue($"t{tid} it{it}: distance length {dv.Length}");
                        }
                        Interlocked.Increment(ref ops);
                    }
                    catch (InvalidOperationException ex) when (ex.Message.Contains("memory limit"))
                    {
                        Interlocked.Increment(ref budgetFailures); // legitimate: everything pinned, nothing spillable
                    }
                    catch (Exception ex)
                    {
                        errors.Enqueue($"t{tid} it{it}: {ex.GetType().Name}: {ex.Message}");
                    }
                }
            }, TaskCreationOptions.LongRunning)).ToArray();

            if (!Task.WaitAll(tasks, TimeSpan.FromSeconds(120)))
                throw new Exception($"Concurrent Rent/Promote/Spill did not finish in 120 s (deadlock?) after {Interlocked.Read(ref ops)} ops. Session leaked deliberately.");

            // Promotion storm: many threads promote at once, so a promoting thread (holding its storage lock)
            // spills while others hold theirs. With blocking spill locks this is the A->B / B->A deadlock.
            long promotions = 0;
            var storm = Enumerable.Range(0, 16).Select(tid => Task.Factory.StartNew(() =>
            {
                var pts = Enumerable.Range(0, 100).Select(i => new V3(tid, i, 0)).ToArray(); // 2400 B -> 4 KiB class
                for (int it = 0; it < 300; it++)
                {
                    try
                    {
                        using var hb = s.UploadPoints(pts, ElementType.V3D, Residency.Host);
                        using (s.Pin(hb)) Interlocked.Increment(ref promotions);
                    }
                    catch (InvalidOperationException ex) when (ex.Message.Contains("memory limit"))
                    {
                        Interlocked.Increment(ref budgetFailures);
                    }
                    catch (Exception ex)
                    {
                        errors.Enqueue($"storm t{tid} it{it}: {ex.GetType().Name}: {ex.Message}");
                    }
                }
            }, TaskCreationOptions.LongRunning)).ToArray();
            if (!Task.WaitAll(storm, TimeSpan.FromSeconds(120)))
                throw new Exception($"Promotion storm did not finish in 120 s (deadlock?) after {Interlocked.Read(ref promotions)} promotions. Session leaked deliberately.");

            foreach (var b in shared) b.Dispose();
            var m = s.Metrics;
            s.Dispose();
            if (!errors.IsEmpty) throw new Exception($"{errors.Count} errors, first: {errors.First()}");
            // Refused allocations (everything pinned) must not leak: with every buffer disposed nothing is live.
            if (m.ActiveBytes != 0 || m.LiveAllocations != 0)
                throw new Exception($"Leak after stress: ActiveBytes {m.ActiveBytes}, LiveAllocations {m.LiveAllocations}");
            if (ops < threads) throw new Exception($"Only {ops} of {threads * iterations} iterations completed ({budgetFailures} budget failures).");
            if (m.SpilledBytes == 0) throw new Exception("Stress never spilled: budget too loose to exercise the Rent->spill vs Promote lock path.");
            return $"{ops} concurrent pin/rent/op iterations on {threads} threads + {promotions} storm promotions on 16 threads, {m.SpilledBytes} B spilled, {budgetFailures} clean budget refusals, no deadlock.";
        }
    }
}
