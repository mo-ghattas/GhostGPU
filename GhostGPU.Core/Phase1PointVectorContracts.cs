using System;

namespace GhostGPU.Core
{
    /// <summary>
    /// Explicit plane representation defined by an origin point and normal vector.
    /// Used for vector/point projection and point reflection operations.
    /// </summary>
    public readonly struct Plane3D : IEquatable<Plane3D>
    {
        public readonly V3 Origin;
        public readonly V3 Normal;

        public Plane3D(V3 origin, V3 normal)
        {
            Origin = origin;
            Normal = normal;
        }

        public static Plane3D WorldXY => new Plane3D(new V3(0, 0, 0), new V3(0, 0, 1));
        public static Plane3D WorldYZ => new Plane3D(new V3(0, 0, 0), new V3(1, 0, 0));
        public static Plane3D WorldZX => new Plane3D(new V3(0, 0, 0), new V3(0, 1, 0));

        public bool Equals(Plane3D other) => Origin.Equals(other.Origin) && Normal.Equals(other.Normal);
        public override bool Equals(object obj) => obj is Plane3D other && Equals(other);
        public override int GetHashCode() => unchecked((Origin.GetHashCode() * 397) ^ Normal.GetHashCode());
        public static bool operator ==(Plane3D left, Plane3D right) => left.Equals(right);
        public static bool operator !=(Plane3D left, Plane3D right) => !left.Equals(right);
    }

    /// <summary>
    /// Constants and precision tolerances for Phase 1 point and vector math.
    /// </summary>
    public static class PointVectorConstants
    {
        public const double ZeroTolerance = 1e-12;
        public const double DegenerateTolerance = 1e-14;
    }
}
