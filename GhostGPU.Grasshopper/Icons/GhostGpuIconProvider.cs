using System;
using System.Collections.Generic;
using System.Drawing;
using Grasshopper.Kernel;

namespace GhostGPU.Grasshopper.Icons
{
    /// <summary>
    /// Primary runtime icon provider for all GhostGPU Grasshopper components.
    /// Provides automatic native counterpart icon discovery, high-fidelity GPU cyan/teal recoloring,
    /// consistent fallback generation for custom nodes, and serialized thread-safe caching.
    /// Caches one bitmap per component identity during ordinary component use.
    /// </summary>
    public static class GhostGpuIconProvider
    {
        private static readonly object _syncLock = new object();
        private static readonly Dictionary<Guid, Bitmap> _cachedByGuid = new Dictionary<Guid, Bitmap>();
        private static readonly Dictionary<string, Bitmap> _cachedByName = new Dictionary<string, Bitmap>(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Retrieves the recolored or fallback icon for a GhostGPU component instance.
        /// </summary>
        /// <param name="component">The Grasshopper component.</param>
        /// <returns>A 24x24 32bpp ARGB icon Bitmap.</returns>
        public static Bitmap GetIcon(GH_Component component)
        {
            if (component == null) return FallbackIconGenerator.CreateFallbackIcon("GPU");

            Guid guid = component.ComponentGuid;
            string typeName = component.GetType().Name;
            string nickname = component.NickName;

            return GetIcon(guid, typeName, nickname);
        }

        /// <summary>
        /// Retrieves the recolored or fallback icon for a component by its GUID and optional type/nickname.
        /// Fully serialized to prevent race-condition duplicate allocations and GDI+ handle leaks.
        /// </summary>
        /// <param name="componentGuid">GhostGPU component GUID.</param>
        /// <param name="componentTypeName">Optional component class name for fallback lookup.</param>
        /// <param name="nickname">Optional component nickname for fallback label.</param>
        /// <returns>A 24x24 32bpp ARGB icon Bitmap.</returns>
        public static Bitmap GetIcon(Guid componentGuid, string componentTypeName = null, string nickname = null)
        {
            if (componentGuid == Guid.Empty && string.IsNullOrEmpty(componentTypeName))
            {
                return FallbackIconGenerator.CreateFallbackIcon("GPU");
            }

            lock (_syncLock)
            {
                if (componentGuid != Guid.Empty && _cachedByGuid.TryGetValue(componentGuid, out Bitmap cachedByGuid))
                {
                    return cachedByGuid;
                }

                if (componentGuid == Guid.Empty && !string.IsNullOrEmpty(componentTypeName) && _cachedByName.TryGetValue(componentTypeName, out Bitmap cachedByName))
                {
                    return cachedByName;
                }

                Bitmap resolved = ResolveIcon(componentGuid, componentTypeName, nickname);

                if (resolved != null)
                {
                    if (componentGuid != Guid.Empty)
                    {
                        _cachedByGuid[componentGuid] = resolved;
                    }
                    if (!string.IsNullOrEmpty(componentTypeName))
                    {
                        _cachedByName[componentTypeName] = resolved;
                    }
                }

                return resolved;
            }
        }

        private static Bitmap ResolveIcon(Guid componentGuid, string componentTypeName, string nickname)
        {
            NativeMapping mapping = null;

            if (componentGuid != Guid.Empty)
            {
                NativeIconCatalog.TryGetMapping(componentGuid, out mapping);
            }

            if (mapping == null && !string.IsNullOrEmpty(componentTypeName))
            {
                NativeIconCatalog.TryGetMapping(componentTypeName, out mapping);
            }

            if (mapping != null && mapping.HasNativeCounterpart)
            {
                Bitmap nativeSource = NativeIconExtractor.ExtractNativeIcon(mapping);
                if (nativeSource != null)
                {
                    Bitmap recolored = IconRecolorEngine.Recolor(nativeSource);
                    if (recolored != null)
                    {
                        return recolored;
                    }
                }
            }

            // Fallback generation for custom nodes or when native icon extraction fails
            string label = mapping?.FallbackLabel ?? nickname ?? componentTypeName ?? "GPU";
            return FallbackIconGenerator.CreateFallbackIcon(label);
        }

        /// <summary>
        /// Recolors an arbitrary source icon bitmap to the GhostGPU cyan/teal brand palette.
        /// </summary>
        public static Bitmap Recolor(Bitmap sourceBitmap)
        {
            return IconRecolorEngine.Recolor(sourceBitmap);
        }

        /// <summary>
        /// Generates a procedural 24x24 fallback icon with the specified monogram or label.
        /// </summary>
        public static Bitmap GetFallbackIcon(string label = null)
        {
            return FallbackIconGenerator.CreateFallbackIcon(label);
        }

        /// <summary>
        /// Clears cached icon dictionary entries.
        /// Note: Existing references already returned to Grasshopper canvas renderers remain valid.
        /// Discarded bitmaps are released when all UI references are collected by GC; callers holding
        /// external bitmap handles remain responsible for their own lifecycle.
        /// </summary>
        public static void ClearCache()
        {
            lock (_syncLock)
            {
                _cachedByGuid.Clear();
                _cachedByName.Clear();
            }
            FallbackIconGenerator.ClearCache();
            NativeIconExtractor.ClearCache();
        }
    }
}
