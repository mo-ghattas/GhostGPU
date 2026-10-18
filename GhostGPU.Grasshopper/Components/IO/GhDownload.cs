using System;
using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GhostGPU.Core;
using GhostGPU.Grasshopper.Parameters;

namespace GhostGPU.Grasshopper.Components.IO
{
    public class GhDownload : GhostGpuComponentBase
    {
        public GhDownload()
            : base("Download GPU", "Down",
                   "Explicit readback boundary. Downloads resident GPU buffers or meshes into host Grasshopper geometry. Preserves exact branch paths, element counts, double precision vertices, and rejects disposed handles.",
                   SubCategoryIO)
        {
        }

        public override Guid ComponentGuid => new Guid("35E00A63-958C-4FA9-A6CE-7B4D228DF118");

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddParameter(new Param_GpuBuffer(), "Buffer", "B", "Resident GPU buffer(s) to download", GH_ParamAccess.tree);
            pManager.AddParameter(new Param_GpuMesh(), "Mesh", "M", "Resident GPU mesh(es) to download", GH_ParamAccess.tree);
            pManager[0].Optional = true;
            pManager[1].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddPointParameter("Points", "P", "Downloaded points", GH_ParamAccess.tree);
            pManager.AddVectorParameter("Vectors", "V", "Downloaded vectors", GH_ParamAccess.tree);
            pManager.AddNumberParameter("Numbers", "N", "Downloaded numbers", GH_ParamAccess.tree);
            pManager.AddIntegerParameter("Indices", "i", "Downloaded indices", GH_ParamAccess.tree);
            pManager.AddColourParameter("Colours", "C", "Downloaded colours", GH_ParamAccess.tree);
            pManager.AddMeshParameter("Meshes", "M", "Downloaded meshes with updated double-precision vertices/colours and original topology", GH_ParamAccess.tree);
        }

        protected override void Solve(IGH_DataAccess DA)
        {
            var session = GetGpuSession();

            var outPts = new GH_Structure<GH_Point>();
            var outVecs = new GH_Structure<GH_Vector>();
            var outNums = new GH_Structure<GH_Number>();
            var outIdxs = new GH_Structure<GH_Integer>();
            var outCols = new GH_Structure<GH_Colour>();
            var outMeshes = new GH_Structure<GH_Mesh>();

            // Process Buffers
            if (DA.GetDataTree(0, out GH_Structure<GH_GpuBuffer> bufferTree) && bufferTree.PathCount > 0)
            {
                foreach (GH_Path path in bufferTree.Paths)
                {
                    var branch = bufferTree[path];
                    if (branch == null || branch.Count == 0)
                    {
                        outPts.EnsurePath(path);
                        outVecs.EnsurePath(path);
                        outNums.EnsurePath(path);
                        outIdxs.EnsurePath(path);
                        outCols.EnsurePath(path);
                        continue;
                    }

                    foreach (var goo in branch)
                    {
                        if (goo == null || goo.Value == null || goo.Value.IsDisposed)
                        {
                            throw new InvalidOperationException($"Branch {path} contains a null or disposed GPU buffer handle. Cannot download.");
                        }

                        var buf = goo.Value;
                        switch (buf.Kind)
                        {
                            case BufferKind.Points:
                                outPts.EnsurePath(path);
                                var pts = session.DownloadPoints(buf);
                                foreach (var p in pts)
                                    outPts.Append(new GH_Point(new Point3d(p.X, p.Y, p.Z)), path);
                                break;

                            case BufferKind.Vectors:
                                outVecs.EnsurePath(path);
                                var vecs = session.DownloadVectors(buf);
                                foreach (var v in vecs)
                                    outVecs.Append(new GH_Vector(new Vector3d(v.X, v.Y, v.Z)), path);
                                break;

                            case BufferKind.Scalars:
                                outNums.EnsurePath(path);
                                var nums = session.DownloadScalars(buf);
                                foreach (var n in nums)
                                    outNums.Append(new GH_Number(n), path);
                                break;

                            case BufferKind.Indices:
                                outIdxs.EnsurePath(path);
                                var idxs = session.DownloadIndices(buf);
                                foreach (var i in idxs)
                                    outIdxs.Append(new GH_Integer(i), path);
                                break;

                            case BufferKind.Colours:
                                outCols.EnsurePath(path);
                                var argbs = session.DownloadColours(buf);
                                foreach (var c in argbs)
                                    outCols.Append(new GH_Colour(Color.FromArgb((int)c)), path);
                                break;
                        }
                    }
                }
            }

            // Process Meshes
            if (DA.GetDataTree(1, out GH_Structure<GH_GpuMesh> meshTree) && meshTree.PathCount > 0)
            {
                foreach (GH_Path path in meshTree.Paths)
                {
                    var branch = meshTree[path];
                    if (branch == null || branch.Count == 0)
                    {
                        outMeshes.EnsurePath(path);
                        continue;
                    }

                    foreach (var goo in branch)
                    {
                        if (goo == null || goo.Value == null || goo.Value.VertexPositions == null || goo.Value.VertexPositions.IsDisposed)
                        {
                            throw new InvalidOperationException($"Branch {path} contains a null or disposed GPU mesh handle. Cannot download.");
                        }

                        var payload = goo.Value;
                        var v3s = session.DownloadPoints(payload.VertexPositions);
                        int expectedVertices = payload.TopologyTemplate.Vertices.Count;

                        if (v3s.Length != expectedVertices)
                        {
                            throw new InvalidOperationException(
                                $"Downloaded vertex count ({v3s.Length}) does not match topology template vertex count ({expectedVertices}).");
                        }

                        var reconstructed = payload.TopologyTemplate.DuplicateMesh();
                        reconstructed.Vertices.UseDoublePrecisionVertices = true;

                        for (int i = 0; i < v3s.Length; i++)
                        {
                            reconstructed.Vertices.SetVertex(i, v3s[i].X, v3s[i].Y, v3s[i].Z);
                        }

                        if (payload.VertexColours != null && !payload.VertexColours.IsDisposed)
                        {
                            var argbs = session.DownloadColours(payload.VertexColours);
                            if (argbs.Length != expectedVertices)
                            {
                                throw new InvalidOperationException(
                                    $"Downloaded colour count ({argbs.Length}) does not match mesh vertex count ({expectedVertices}).");
                            }

                            reconstructed.VertexColors.Clear();
                            for (int i = 0; i < argbs.Length; i++)
                            {
                                reconstructed.VertexColors.Add(Color.FromArgb((int)argbs[i]));
                            }
                        }

                        reconstructed.Normals.ComputeNormals();
                        outMeshes.Append(new GH_Mesh(reconstructed), path);
                    }
                }
            }

            DA.SetDataTree(0, outPts);
            DA.SetDataTree(1, outVecs);
            DA.SetDataTree(2, outNums);
            DA.SetDataTree(3, outIdxs);
            DA.SetDataTree(4, outCols);
            DA.SetDataTree(5, outMeshes);
        }
    }
}
