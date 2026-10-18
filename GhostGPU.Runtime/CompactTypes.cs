using System;
using System.Runtime.InteropServices;
using GhostGPU.Core;

namespace GhostGPU.Runtime
{
    /// <summary>
    /// Compact single-precision 3D vector / point offset (12 bytes).
    /// Used with a per-buffer double-precision Origin so that world coordinates maintain precision.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct V3F : IEquatable<V3F>
    {
        public readonly float X;
        public readonly float Y;
        public readonly float Z;

        public V3F(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public V3 ToV3(V3 origin) => new V3(origin.X + (double)X, origin.Y + (double)Y, origin.Z + (double)Z);

        public static V3F FromV3(V3 v, V3 origin) =>
            new V3F((float)(v.X - origin.X), (float)(v.Y - origin.Y), (float)(v.Z - origin.Z));

        public bool Equals(V3F other) => X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z);
        public override bool Equals(object obj) => obj is V3F other && Equals(other);
        public override int GetHashCode() => unchecked((X.GetHashCode() * 397 ^ Y.GetHashCode()) * 397 ^ Z.GetHashCode());
        public override string ToString() => $"({X}, {Y}, {Z})";
    }

    internal static class CompactTypeHelper
    {
        public static V3 ComputeOrigin(V3[] values)
        {
            if (values == null || values.Length == 0)
                return default;

            double minX = double.PositiveInfinity, minY = double.PositiveInfinity, minZ = double.PositiveInfinity;
            double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity, maxZ = double.NegativeInfinity;

            for (int i = 0; i < values.Length; i++)
            {
                var v = values[i];
                if (v.X < minX) minX = v.X;
                if (v.X > maxX) maxX = v.X;
                if (v.Y < minY) minY = v.Y;
                if (v.Y > maxY) maxY = v.Y;
                if (v.Z < minZ) minZ = v.Z;
                if (v.Z > maxZ) maxZ = v.Z;
            }

            if (double.IsInfinity(minX) || double.IsInfinity(maxX))
                return default;

            return new V3((minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5);
        }
    }
}
