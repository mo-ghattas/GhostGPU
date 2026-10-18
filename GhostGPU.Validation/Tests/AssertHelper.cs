using System;
using GhostGPU.Core;

namespace GhostGPU.Validation.Tests
{
    public static class AssertHelper
    {
        public static void AssertClose(double actual, double expected, double tolerance, string message)
        {
            if (double.IsNaN(expected))
            {
                if (!double.IsNaN(actual))
                    throw new Exception($"{message}: Expected NaN, but got {actual}");
                return;
            }

            if (double.IsPositiveInfinity(expected))
            {
                if (!double.IsPositiveInfinity(actual))
                    throw new Exception($"{message}: Expected +Infinity, but got {actual}");
                return;
            }

            if (double.IsNegativeInfinity(expected))
            {
                if (!double.IsNegativeInfinity(actual))
                    throw new Exception($"{message}: Expected -Infinity, but got {actual}");
                return;
            }

            // Expected is finite: actual MUST be finite (detect NaN/Inf explicitly)
            if (double.IsNaN(actual))
                throw new Exception($"{message}: Expected finite value {expected}, but got NaN!");

            if (double.IsInfinity(actual))
                throw new Exception($"{message}: Expected finite value {expected}, but got {actual} (Infinity)!");

            double diff = Math.Abs(actual - expected);
            if (diff > tolerance)
                throw new Exception($"{message}: Expected {expected}, got {actual} (difference {diff} > tolerance {tolerance})");
        }

        public static void AssertClose(V3 actual, V3 expected, double tolerance, string message)
        {
            if (!actual.Finite)
                throw new Exception($"{message}: Expected finite V3 ({expected.X},{expected.Y},{expected.Z}), but actual contains nonfinite coords: ({actual.X},{actual.Y},{actual.Z})");

            AssertClose(actual.X, expected.X, tolerance, $"{message} [X]");
            AssertClose(actual.Y, expected.Y, tolerance, $"{message} [Y]");
            AssertClose(actual.Z, expected.Z, tolerance, $"{message} [Z]");
        }

        public static void AssertFinite(double value, string message)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
                throw new Exception($"{message}: Expected finite double, got {value}");
        }

        public static void AssertFinite(V3 value, string message)
        {
            if (!value.Finite)
                throw new Exception($"{message}: Expected finite V3, got ({value.X},{value.Y},{value.Z})");
        }
    }
}
