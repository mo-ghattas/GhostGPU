using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Kernel
{
    /// <summary>
    /// M2: exact Brep-Brep surface-surface intersection.
    /// Device (or host below <see cref="SsiBatch.MinDeviceBatch"/>): Bezier patch-pair hull culling with breadth-first
    /// subdivision, and batched Newton seeding from the leaf boxes. Host: per-face-pair marching (Parallel.For over face
    /// pairs), seam/trim cutting, duplicate removal and the cubic fits.
    /// Output: one SsiBranch per face pair piece. Branches are split at face trims and at surface seams, so a curve
    /// that Rhino returns joined (e.g. around a box) arrives here as several branches sharing endpoints. IsClosed means
    /// the 3D curve is closed; a closed branch crossing a seam once keeps open UV pullbacks that start and end on it.
    /// </summary>
    public static class Ssi
    {
        /// <summary>Phase timings of the last call on this thread (ms): subdivide, seed, march, fit.</summary>
        [ThreadStatic] public static double[] LastTimings;

        public static SsiResult Intersect(KBrep a, KBrep b, double tolerance, GpuSession session = null)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (!(tolerance > 0) || double.IsInfinity(tolerance)) throw new ArgumentOutOfRangeException(nameof(tolerance));
            var res = new SsiResult();
            var sw = Stopwatch.StartNew();
            var timings = new double[4];
            LastTimings = timings;
            if (a.Faces.Count == 0 || b.Faces.Count == 0) return res;

            var fa = new SsiFace[a.Faces.Count]; var fb = new SsiFace[b.Faces.Count];
            Parallel.For(0, fa.Length + fb.Length, i =>
            {
                if (i < fa.Length) fa[i] = SsiFace.Build(a, i); else fb[i - fa.Length] = SsiFace.Build(b, i - fa.Length);
            });

            V3 amin = fa[0].BoxMin, amax = fa[0].BoxMax, bmin = fb[0].BoxMin, bmax = fb[0].BoxMax;
            foreach (var f in fa) { amin = V3.Min(amin, f.BoxMin); amax = V3.Max(amax, f.BoxMax); }
            foreach (var f in fb) { bmin = V3.Min(bmin, f.BoxMin); bmax = V3.Max(bmax, f.BoxMax); }
            if (!Overlap(amin, amax, bmin, bmax, tolerance)) return res;
            V3 omin = V3.Max(amin, bmin), omax = V3.Min(amax, bmax);
            double odiag = new V3(Math.Max(0, omax.X - omin.X), Math.Max(0, omax.Y - omin.Y), Math.Max(0, omax.Z - omin.Z)).Length;
            double leafSize = Math.Max(odiag / 64, 10 * tolerance);

            // coincident planar faces: no transversal curve; their overlap boundary comes from neighbouring face pairs
            var skip = new bool[fa.Length, fb.Length];
            for (int i = 0; i < fa.Length; i++)
                for (int j = 0; j < fb.Length; j++)
                {
                    if (!fa[i].Planar || !fb[j].Planar) continue;
                    double c = V3.Dot(fa[i].PlaneN, fb[j].PlaneN);
                    if (Math.Abs(c) < 1 - 1e-12) continue;
                    if (Math.Abs(fa[i].PlaneD - Math.Sign(c) * fb[j].PlaneD) > tolerance) continue;
                    if (!Overlap(fa[i].BoxMin, fa[i].BoxMax, fb[j].BoxMin, fb[j].BoxMax, tolerance)) continue;
                    skip[i, j] = true;
                    res.Warnings.Add($"faces A{i}/B{j} are coplanar (coincident or touching); no transversal curve, overlap boundary is reported by the adjacent face pairs");
                }

            // patch tables
            var patches = new List<BezierPatch>(); var patchFace = new List<int>();
            int nA = 0;
            foreach (var f in fa) foreach (var p in f.Patches) { patches.Add(p); patchFace.Add(f.Index); }
            nA = patches.Count;
            foreach (var f in fb) foreach (var p in f.Patches) { patches.Add(p); patchFace.Add(f.Index); }
            var meta = new int[patches.Count * 3];
            var cvList = new List<double>();
            for (int i = 0; i < patches.Count; i++)
            {
                meta[3 * i] = cvList.Count / 4; meta[3 * i + 1] = patches[i].DegreeU; meta[3 * i + 2] = patches[i].DegreeV;
                cvList.AddRange(patches[i].Cvs);
            }
            var init = new List<SsiPair>();
            for (int i = 0; i < nA; i++)
                for (int j = nA; j < patches.Count; j++)
                {
                    if (skip[patchFace[i], patchFace[j]]) continue;
                    if (!Overlap(patches[i].BoxMin, patches[i].BoxMax, patches[j].BoxMin, patches[j].BoxMax, tolerance)) continue;
                    init.Add(new SsiPair { A = i, B = j, Au0 = 0, Au1 = 1, Av0 = 0, Av1 = 1, Bu0 = 0, Bu1 = 1, Bv0 = 0, Bv1 = 1 });
                }

            List<SsiPair> leaves; SsiSeed[] seeds;
            using (var batch = new SsiBatch(session, cvList.ToArray(), meta))
            {
                leaves = batch.Subdivide(init, tolerance, leafSize, 24, 1 << 21, res.Warnings);
                timings[0] = sw.Elapsed.TotalMilliseconds;
                seeds = batch.Newton(leaves.ToArray());
                timings[1] = sw.Elapsed.TotalMilliseconds;
            }

            // seeds per face pair, in surface parameters
            var bucket = new SortedDictionary<long, List<double[]>>();
            for (int i = 0; i < leaves.Count; i++)
            {
                var s = seeds[i];
                if (!(s.Residual < leafSize)) continue;
                BezierPatch pa = patches[leaves[i].A], pb = patches[leaves[i].B];
                long key = (long)patchFace[leaves[i].A] * fb.Length + patchFace[leaves[i].B];
                if (!bucket.TryGetValue(key, out var l)) bucket[key] = l = new List<double[]>();
                l.Add(new[] { pa.U.ParameterAt(s.Sa), pa.V.ParameterAt(s.Ta), pb.U.ParameterAt(s.Sb), pb.V.ParameterAt(s.Tb), s.Residual });
            }

            // boundary seeds: points of each face's trim boundary near the other face (open branches ending on trims)
            for (int i = 0; i < fa.Length; i++)
                for (int j = 0; j < fb.Length; j++)
                {
                    if (skip[i, j] || !Overlap(fa[i].BoxMin, fa[i].BoxMax, fb[j].BoxMin, fb[j].BoxMax, tolerance)) continue;
                    var bs = BoundarySeeds(fa[i], fb[j], true, 2 * leafSize);
                    bs.AddRange(BoundarySeeds(fb[j], fa[i], false, 2 * leafSize));
                    if (bs.Count == 0) continue;
                    long key = (long)i * fb.Length + j;
                    if (!bucket.TryGetValue(key, out var l)) bucket[key] = l = new List<double[]>();
                    l.AddRange(bs);
                }

            // march each face pair (host, parallel over face pairs)
            var keys = bucket.Keys.ToArray();
            var tracers = new SsiTracer[keys.Length];
            var piecesPer = new List<SsiPiece>[keys.Length];
            Parallel.For(0, keys.Length, k =>
            {
                int i = (int)(keys[k] / fb.Length), j = (int)(keys[k] % fb.Length);
                double hmax = Math.Max(10 * tolerance, Math.Min(odiag, Math.Min(fa[i].Diag, fb[j].Diag)) / 10);
                var tr = new SsiTracer(fa[i], fb[j], tolerance, hmax);
                var list = bucket[keys[k]];
                list.Sort((x, y) => x[4].CompareTo(y[4]));
                foreach (var s in list) tr.TraceSeed(new[] { s[0], s[1], s[2], s[3] });
                var pcs = new List<SsiPiece>();
                foreach (var t in tr.Traces) pcs.AddRange(tr.Cut(t));
                tracers[k] = tr; piecesPer[k] = pcs;
            });
            timings[2] = sw.Elapsed.TotalMilliseconds;

            // duplicates across face pairs (curves along shared edges / coincident faces)
            var all = new List<(SsiPiece pc, SsiTracer tr)>();
            for (int k = 0; k < keys.Length; k++)
            {
                res.Warnings.AddRange(tracers[k].Warnings.Distinct());
                foreach (var p in piecesPer[k]) all.Add((p, tracers[k]));
            }
            var order = Enumerable.Range(0, all.Count).OrderByDescending(i => all[i].pc.Length).ThenBy(i => i).ToList();
            var keep = new List<int>();
            foreach (int i in order)
            {
                bool dup = false;
                foreach (int j in keep)
                    if (Covered(all[i].pc, all[j].pc, 2 * tolerance)) { dup = true; break; }
                if (!dup) keep.Add(i);
            }
            keep.Sort();

            var branches = new SsiBranch[keep.Count];
            var fitWarnings = new List<string>();
            Parallel.For(0, keep.Count, k =>
            {
                var w = new List<string>();
                branches[k] = SsiFit.Fit(all[keep[k]].tr, all[keep[k]].pc, tolerance, w);
                if (w.Count > 0) lock (fitWarnings) fitWarnings.AddRange(w);
            });
            res.Branches.AddRange(branches);
            res.Warnings.AddRange(fitWarnings);
            timings[3] = sw.Elapsed.TotalMilliseconds;
            return res;
        }

        /// <summary>
        /// Trim-boundary points of <paramref name="from"/> lying within radius of a coarse 16x16 sample grid of
        /// <paramref name="to"/>; returned as unpolished seeds (uA,vA,uB,vB,priority) for TraceSeed's Newton polish.
        /// Limitation: grid start guess, ~256 boundary samples per face; a true curve/surface intersection would be exact.
        /// </summary>
        static List<double[]> BoundarySeeds(SsiFace from, SsiFace to, bool fromIsA, double radius)
        {
            var res = new List<double[]>();
            if (from.SegA.Length == 0) return res;
            const int G = 16;
            var grid = new V3[(G + 1) * (G + 1)];
            for (int i = 0; i <= G; i++)
                for (int j = 0; j <= G; j++)
                    grid[i * (G + 1) + j] = NurbsEval.SurfacePoint(to.Srf, to.U0 + (to.U1 - to.U0) * i / G, to.V0 + (to.V1 - to.V0) * j / G);
            int stride = Math.Max(1, from.SegA.Length / 256);
            for (int s = 0; s < from.SegA.Length; s += stride)
            {
                V2 uv = from.SegA[s];
                V3 p = NurbsEval.SurfacePoint(from.Srf, from.WrapU(uv.X), from.WrapV(uv.Y));
                int bi = -1; double bd = radius;
                for (int g = 0; g < grid.Length; g++) { double d = (grid[g] - p).Length; if (d < bd) { bd = d; bi = g; } }
                if (bi < 0) continue;
                double u = to.U0 + (to.U1 - to.U0) * (bi / (G + 1)) / G, v = to.V0 + (to.V1 - to.V0) * (bi % (G + 1)) / G;
                res.Add(fromIsA ? new[] { uv.X, uv.Y, u, v, double.MaxValue } : new[] { u, v, uv.X, uv.Y, double.MaxValue });
            }
            return res;
        }

        static bool Overlap(V3 amin, V3 amax, V3 bmin, V3 bmax, double e) =>
            !(amin.X > bmax.X + e || bmin.X > amax.X + e || amin.Y > bmax.Y + e || bmin.Y > amax.Y + e
              || amin.Z > bmax.Z + e || bmin.Z > amax.Z + e);

        /// <summary>True when every node of x lies within dist of y's polyline.</summary>
        static bool Covered(SsiPiece x, SsiPiece y, double dist)
        {
            foreach (var p in x.P)
            {
                bool near = false;
                for (int s = 0; s + 1 < y.P.Count && !near; s++)
                {
                    V3 a = y.P[s], e = y.P[s + 1] - a;
                    double l2 = V3.Dot(e, e);
                    double t = l2 > 0 ? Math.Max(0, Math.Min(1, V3.Dot(p - a, e) / l2)) : 0;
                    near = (p - (a + e * t)).Length <= dist;
                }
                if (!near) return false;
            }
            return true;
        }
    }
}
