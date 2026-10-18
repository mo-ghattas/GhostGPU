using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Reflection;
using Grasshopper.Kernel;

namespace GhostGPU.Grasshopper.Icons
{
    /// <summary>
    /// Safely resolves and extracts native Grasshopper component icons at runtime
    /// from the active ComponentServer or directly from installed Rhino 8 Grasshopper assemblies.
    /// Incorporates anti-recursion guards, thread-safe serialized caching, exact GUID validation,
    /// and dynamic Rhino installation resolution.
    /// Source icons borrowed from Grasshopper are never disposed.
    /// </summary>
    public static class NativeIconExtractor
    {
        private static readonly object _extractLock = new object();
        private static readonly Dictionary<Guid, Bitmap> _sourceIconCache = new Dictionary<Guid, Bitmap>();

        private static readonly object _assemblyLock = new object();
        private static readonly Dictionary<string, Assembly> _assemblyCache = new Dictionary<string, Assembly>(StringComparer.OrdinalIgnoreCase);
        private static readonly HashSet<string> _failedAssemblies = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Attempts to extract the native Grasshopper icon for the given mapping.
        /// Thread-safe and serialized.
        /// </summary>
        /// <param name="mapping">The native mapping information.</param>
        /// <returns>Extracted native Bitmap or null if unavailable.</returns>
        public static Bitmap ExtractNativeIcon(NativeMapping mapping)
        {
            if (mapping == null || !mapping.HasNativeCounterpart) return null;

            lock (_extractLock)
            {
                if (_sourceIconCache.TryGetValue(mapping.NativeGuid, out Bitmap cached))
                {
                    return cached;
                }

                Bitmap icon = TryExtractFromComponentServer(mapping.NativeGuid);

                if (icon == null)
                {
                    icon = TryExtractFromAssembly(mapping);
                }

                if (icon != null)
                {
                    _sourceIconCache[mapping.NativeGuid] = icon;
                }

                return icon;
            }
        }

        private static Bitmap TryExtractFromComponentServer(Guid nativeGuid)
        {
            try
            {
                if (!global::Grasshopper.Instances.IsComponentServer) return null;

                var server = global::Grasshopper.Instances.ComponentServer;
                if (server == null) return null;

                var proxy = server.EmitObjectProxy(nativeGuid);
                if (proxy == null) return null;

                // Validate proxy GUID strictly
                if (proxy.Guid != nativeGuid) return null;

                // Anti-recursion guard: never resolve from GhostGPU's own proxies
                if (proxy.Type != null && proxy.Type.Assembly != null)
                {
                    string asmName = proxy.Type.Assembly.GetName().Name;
                    if (asmName != null && asmName.StartsWith("GhostGPU", StringComparison.OrdinalIgnoreCase))
                    {
                        return null;
                    }
                }

                // Borrow icon from Grasshopper proxy (never dispose)
                return proxy.Icon;
            }
            catch
            {
                // ComponentServer may be uninitialized in headless or unit-test environments
                return null;
            }
        }

        private static Bitmap TryExtractFromAssembly(NativeMapping mapping)
        {
            if (mapping == null || string.IsNullOrEmpty(mapping.NativeTypeName) || string.IsNullOrEmpty(mapping.NativeAssemblyName))
            {
                return null;
            }

            try
            {
                Assembly asm = GetOrLoadAssembly(mapping.NativeAssemblyName);
                if (asm == null) return null;

                // Strict exact type resolution - no fuzzy short name matching to prevent wrong node icons
                Type compType = asm.GetType(mapping.NativeTypeName, false, false);
                if (compType == null || compType.IsAbstract) return null;

                // Create an isolated instance to read the protected/public Icon property
                object inst = Activator.CreateInstance(compType);
                if (inst is GH_Component ghComp)
                {
                    // Strict GUID validation: instantiated component Guid must exactly match expected native Guid
                    if (ghComp.ComponentGuid != mapping.NativeGuid)
                    {
                        return null;
                    }

                    // Try Icon_24x24 property first, otherwise Icon property
                    var iconProp = compType.GetProperty("Icon_24x24", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                                   ?? compType.GetProperty("Icon", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

                    if (iconProp != null)
                    {
                        return iconProp.GetValue(ghComp, null) as Bitmap;
                    }
                }
            }
            catch
            {
                // Silently fallback if assembly cannot be instantiated standalone
            }

            return null;
        }

        private static Assembly GetOrLoadAssembly(string assemblyFileName)
        {
            if (string.IsNullOrEmpty(assemblyFileName)) return null;

            lock (_assemblyLock)
            {
                if (_assemblyCache.TryGetValue(assemblyFileName, out Assembly cached))
                {
                    return cached;
                }

                if (_failedAssemblies.Contains(assemblyFileName))
                {
                    return null;
                }

                Assembly loaded = TryLoadAssemblyFromDisk(assemblyFileName);
                if (loaded != null)
                {
                    _assemblyCache[assemblyFileName] = loaded;
                }
                else
                {
                    // Do not store null in a dictionary; track failed lookups in set
                    _failedAssemblies.Add(assemblyFileName);
                }

                return loaded;
            }
        }

        private static Assembly TryLoadAssemblyFromDisk(string assemblyFileName)
        {
            try
            {
                // 1. Check if already loaded in AppDomain
                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    try
                    {
                        if (string.Equals(Path.GetFileName(asm.Location), assemblyFileName, StringComparison.OrdinalIgnoreCase))
                        {
                            return asm;
                        }
                    }
                    catch { }
                }

                // 2. Resolve components directory dynamically from Grasshopper SDK assembly location
                string ghAsmLoc = typeof(GH_Component).Assembly.Location;
                string componentsDir = null;
                if (!string.IsNullOrEmpty(ghAsmLoc))
                {
                    string ghDir = Path.GetDirectoryName(ghAsmLoc);
                    if (!string.IsNullOrEmpty(ghDir))
                    {
                        string candidateDir = Path.Combine(ghDir, "Components");
                        if (Directory.Exists(candidateDir))
                        {
                            componentsDir = candidateDir;
                        }
                    }
                }

                // Fallback to RhinoInstallDir environment variable if available
                if (componentsDir == null)
                {
                    string rhinoEnv = Environment.GetEnvironmentVariable("RhinoInstallDir");
                    if (!string.IsNullOrEmpty(rhinoEnv))
                    {
                        string candidate = Path.Combine(rhinoEnv, @"Plug-ins\Grasshopper\Components");
                        if (Directory.Exists(candidate)) componentsDir = candidate;
                    }
                }

                // Standard default fallback if needed
                if (componentsDir == null)
                {
                    string defaultDir = @"C:\Program Files\Rhino 8\Plug-ins\Grasshopper\Components";
                    if (Directory.Exists(defaultDir)) componentsDir = defaultDir;
                }

                if (componentsDir != null)
                {
                    string targetPath = Path.Combine(componentsDir, assemblyFileName);
                    if (File.Exists(targetPath))
                    {
                        return Assembly.LoadFrom(targetPath);
                    }
                }
            }
            catch { }

            return null;
        }

        public static void ClearCache()
        {
            lock (_extractLock)
            {
                _sourceIconCache.Clear();
            }
            lock (_assemblyLock)
            {
                _assemblyCache.Clear();
                _failedAssemblies.Clear();
            }
        }
    }
}
