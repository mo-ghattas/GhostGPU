using System;
using System.Collections.Concurrent;
using System.Reflection;
using ILGPU.Runtime;

namespace GhostGPU.Runtime.Kernels
{
    /// <summary>
    /// Thread-safe accelerator kernel cache, caching LoadAutoGroupedStreamKernel delegates per method.
    /// </summary>
    internal sealed class KernelCache
    {
        private readonly Accelerator _accelerator;
        private readonly ConcurrentDictionary<MethodInfo, Delegate> _cache = new ConcurrentDictionary<MethodInfo, Delegate>();

        public KernelCache(Accelerator accelerator)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        }

        public TDelegate GetKernel<TDelegate>(Delegate kernelMethod) where TDelegate : Delegate
        {
            if (kernelMethod == null) throw new ArgumentNullException(nameof(kernelMethod));
            var method = kernelMethod.Method;
            if (_cache.TryGetValue(method, out var cached))
            {
                return (TDelegate)cached;
            }

            var del = CompileKernel<TDelegate>(kernelMethod);
            _cache[method] = del;
            return del;
        }

        private TDelegate CompileKernel<TDelegate>(Delegate kernelMethod) where TDelegate : Delegate
        {
            var genArgs = typeof(TDelegate).GetGenericArguments();
            var methods = typeof(KernelLoaders).GetMethods(BindingFlags.Public | BindingFlags.Static);
            MethodInfo targetMethod = null;
            foreach (var m in methods)
            {
                if (m.Name == "LoadAutoGroupedStreamKernel" && m.IsGenericMethodDefinition)
                {
                    if (m.GetGenericArguments().Length == genArgs.Length && m.GetParameters().Length == 2)
                    {
                        targetMethod = m;
                        break;
                    }
                }
            }
            if (targetMethod == null)
                throw new InvalidOperationException($"No LoadAutoGroupedStreamKernel found on KernelLoaders for {genArgs.Length} arguments.");

            var generic = targetMethod.MakeGenericMethod(genArgs);
            return (TDelegate)generic.Invoke(null, new object[] { _accelerator, kernelMethod });
        }
    }
}
