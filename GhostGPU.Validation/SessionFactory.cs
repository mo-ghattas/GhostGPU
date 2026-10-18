using System;
using GhostGPU.Core;
using GhostGPU.Runtime;
using GhostGPU.Validation.Reference;

namespace GhostGPU.Validation
{
    public static class SessionFactory
    {
        public static IGpuSession CreateSession(SessionOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);

            var session = new GpuSession(options);
            if (options.Backend == GpuBackend.CudaOnly && !session.IsHardwareAccelerated)
            {
                session.Dispose();
                throw new InvalidOperationException("CudaOnly was requested, but session reported IsHardwareAccelerated == false. Silent CPU fallback is prohibited.");
            }

            return session;
        }

        public static IGpuSession CreateNaiveCpuReference(SessionOptions? options = null)
        {
            return new NaiveCpuSession(options ?? new SessionOptions { Backend = GpuBackend.CpuDebug });
        }
    }
}
