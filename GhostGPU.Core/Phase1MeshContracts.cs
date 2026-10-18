using System;
using System.Collections.Generic;

namespace GhostGPU.Core
{
    /// <summary>
    /// Quad or triangle face index descriptor.
    /// When C == D, the face represents a triangle (A, B, C).
    /// When C != D, the face represents a quad (A, B, C, D).
    /// </summary>
    public readonly struct MeshFaceIndices : IEquatable<MeshFaceIndices>
    {
        public readonly int A;
        public readonly int B;
        public readonly int C;
        public readonly int D;

        public bool IsTriangle => C == D;
        public bool IsQuad => C != D;

        public MeshFaceIndices(int a, int b, int c, int d)
        {
            A = a;
            B = b;
            C = c;
            D = d;
        }

        public MeshFaceIndices(int a, int b, int c)
        {
            A = a;
            B = b;
            C = c;
            D = c;
        }

        public bool Equals(MeshFaceIndices other) =>
            A == other.A && B == other.B && C == other.C && D == other.D;

        public override bool Equals(object obj) =>
            obj is MeshFaceIndices other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = A;
                hash = (hash * 397) ^ B;
                hash = (hash * 397) ^ C;
                hash = (hash * 397) ^ D;
                return hash;
            }
        }

        public static bool operator ==(MeshFaceIndices left, MeshFaceIndices right) => left.Equals(right);
        public static bool operator !=(MeshFaceIndices left, MeshFaceIndices right) => !left.Equals(right);

        public override string ToString() =>
            IsTriangle ? $"Tri[{A}, {B}, {C}]" : $"Quad[{A}, {B}, {C}, {D}]";
    }

    /// <summary>
    /// Weighting scheme for aggregating incident face normals into vertex normals.
    /// </summary>
    public enum MeshNormalWeighting : byte
    {
        /// <summary>Sum of unitized adjacent face normals.</summary>
        Unweighted = 0,
        /// <summary>Sum of unnormalized face cross-product vectors (magnitude proportional to face area).</summary>
        AreaWeighted = 1
    }

    /// <summary>
    /// Compressed Sparse Row (CSR) adjacency data representing vertex-to-incident-faces connectivity.
    /// Defensively snapshots input arrays upon construction and validates offsets and indices.
    /// </summary>
    public sealed class MeshVertexAdjacency
    {
        private readonly int[] _vertexOffsets;
        private readonly int[] _adjacentFaces;

        public int VertexCount { get; }
        public IReadOnlyList<int> VertexOffsets => _vertexOffsets;
        public IReadOnlyList<int> AdjacentFaces => _adjacentFaces;
        public int TotalIncidences => _adjacentFaces.Length;

        public int[] GetOffsetsSnapshot() => (int[])_vertexOffsets.Clone();
        public int[] GetFacesSnapshot() => (int[])_adjacentFaces.Clone();

        public MeshVertexAdjacency(int vertexCount, int[] vertexOffsets, int[] adjacentFaces)
        {
            if (vertexCount < 0) throw new ArgumentOutOfRangeException(nameof(vertexCount), "Vertex count must be non-negative.");
            if (vertexOffsets == null) throw new ArgumentNullException(nameof(vertexOffsets));
            if (adjacentFaces == null) throw new ArgumentNullException(nameof(adjacentFaces));

            if (vertexOffsets.Length != vertexCount + 1)
            {
                throw new ArgumentException($"VertexOffsets length must be vertexCount + 1 ({vertexCount + 1}), but was {vertexOffsets.Length}.", nameof(vertexOffsets));
            }

            VertexCount = vertexCount;
            _vertexOffsets = (int[])vertexOffsets.Clone();
            _adjacentFaces = (int[])adjacentFaces.Clone();

            ValidateOffsetsInternal();
        }

        private void ValidateOffsetsInternal()
        {
            if (_vertexOffsets[0] != 0)
                throw new InvalidOperationException("First vertex offset must be 0.");
            if (_vertexOffsets[VertexCount] != _adjacentFaces.Length)
                throw new InvalidOperationException($"Final vertex offset ({_vertexOffsets[VertexCount]}) must equal AdjacentFaces length ({_adjacentFaces.Length}).");

            for (int i = 0; i < VertexCount; i++)
            {
                int start = _vertexOffsets[i];
                int end = _vertexOffsets[i + 1];
                if (start > end)
                    throw new InvalidOperationException($"Non-monotonic CSR offset at vertex {i}: start {start} > end {end}.");
                if (start < 0 || end > _adjacentFaces.Length)
                    throw new InvalidOperationException($"CSR offset at vertex {i} out of bounds: [{start}, {end}] in length {_adjacentFaces.Length}.");
            }
        }

        public void ValidateFaces(int maxFaceIndex)
        {
            for (int i = 0; i < VertexCount; i++)
            {
                int start = _vertexOffsets[i];
                int end = _vertexOffsets[i + 1];
                for (int k = start; k < end; k++)
                {
                    int f = _adjacentFaces[k];
                    if (f < 0 || f >= maxFaceIndex)
                        throw new IndexOutOfRangeException($"Adjacent face index {f} at vertex {i} is outside valid range [0, {maxFaceIndex - 1}].");
                }
            }
        }
    }

    /// <summary>
    /// Immutable topology descriptor containing face connectivity and optional precomputed CSR vertex adjacency.
    /// Defensively snapshots input arrays upon construction and performs strict index bounds checking.
    /// </summary>
    public sealed class MeshTopologyDescriptor
    {
        private readonly MeshFaceIndices[] _faces;

        public int VertexCount { get; }
        public int FaceCount => _faces.Length;
        public MeshVertexAdjacency Adjacency { get; }

        public IReadOnlyList<MeshFaceIndices> Faces => _faces;
        public MeshFaceIndices[] GetFacesSnapshot() => (MeshFaceIndices[])_faces.Clone();

        public MeshFaceIndices this[int index] => _faces[index];

        public MeshTopologyDescriptor(int vertexCount, MeshFaceIndices[] faces, MeshVertexAdjacency adjacency = null)
        {
            if (vertexCount < 0) throw new ArgumentOutOfRangeException(nameof(vertexCount));
            VertexCount = vertexCount;
            _faces = faces != null ? (MeshFaceIndices[])faces.Clone() : Array.Empty<MeshFaceIndices>();
            Adjacency = adjacency;

            Validate();
        }

        public static MeshTopologyDescriptor CreateValidated(int vertexCount, MeshFaceIndices[] faces, MeshVertexAdjacency adjacency = null)
        {
            return new MeshTopologyDescriptor(vertexCount, faces, adjacency);
        }

        public void Validate()
        {
            for (int i = 0; i < _faces.Length; i++)
            {
                var f = _faces[i];
                if (VertexCount == 0)
                {
                    throw new IndexOutOfRangeException($"Mesh has 0 vertices but face {i} references vertices ({f.A}, {f.B}, {f.C}, {f.D}).");
                }
                if (f.A < 0 || f.A >= VertexCount ||
                    f.B < 0 || f.B >= VertexCount ||
                    f.C < 0 || f.C >= VertexCount ||
                    f.D < 0 || f.D >= VertexCount)
                {
                    throw new IndexOutOfRangeException(
                        $"Mesh face {i} ({f}) references vertex out of valid range [0, {VertexCount - 1}].");
                }
            }

            if (Adjacency != null)
            {
                if (Adjacency.VertexCount != VertexCount)
                    throw new ArgumentException($"Adjacency vertex count ({Adjacency.VertexCount}) does not match topology vertex count ({VertexCount}).");
                Adjacency.ValidateFaces(FaceCount);
            }
        }
    }

    /// <summary>
    /// Holds resident GPU buffers produced by unified face analysis (centers, normals, areas).
    /// </summary>
    public sealed class FaceAnalysisResult : IDisposable
    {
        public GpuBuffer Centers { get; }
        public GpuBuffer Normals { get; }
        public GpuBuffer Areas { get; }

        public FaceAnalysisResult(GpuBuffer centers, GpuBuffer normals, GpuBuffer areas)
        {
            Centers = centers;
            Normals = normals;
            Areas = areas;
        }

        public void Dispose()
        {
            Centers?.Dispose();
            Normals?.Dispose();
            Areas?.Dispose();
        }
    }
}
