using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using GhostGPU.Grasshopper.Components;
using GhostGPU.Grasshopper.Icons;
using Grasshopper.Kernel;

internal static class ComponentChecks
{
    internal static void Run(string install)
    {
        var components = typeof(GhostGpuComponentBase).Assembly.GetTypes()
            .Where(t => !t.IsAbstract && typeof(GhostGpuComponentBase).IsAssignableFrom(t))
            .Select(t => (GH_Component)Activator.CreateInstance(t)).OrderBy(c => c.SubCategory).ThenBy(c => c.Name).ToArray();
        Require(components.Select(c => c.ComponentGuid).Distinct().Count() == components.Length, "Duplicate GhostGPU component GUID.");
        var rows = new List<object>();
        int mapped = 0, changedPixels = 0;
        using var sheet = new Bitmap(1320, ((components.Length + 2) / 3) * 64 + 40);
        using var graphics = Graphics.FromImage(sheet); graphics.Clear(Color.FromArgb(246, 248, 250));
        using var font = new Font("Segoe UI", 9); using var title = new Font("Segoe UI", 12, FontStyle.Bold);
        graphics.DrawString("GhostGPU: native icon / recolored GPU icon", title, Brushes.Black, 10, 8);
        for (int index = 0; index < components.Length; index++)
        {
            var component = components[index]; var icon = component.Icon_24x24;
            Require(icon != null && icon.Width == 24 && icon.Height == 24, component.Name + " missing 24px icon.");
            int x = (index % 3) * 440 + 12, y = (index / 3) * 64 + 42;
            Guid nativeGuid = Guid.Empty;
            if (NativeIconCatalog.TryGetMapping(component.ComponentGuid, out var mapping) && mapping.HasNativeCounterpart)
            {
                nativeGuid = mapping.NativeGuid;
                var assembly = Assembly.LoadFrom(Path.Combine(install, "Plug-ins", "Grasshopper", "Components", mapping.NativeAssemblyName));
                var native = (GH_Component)Activator.CreateInstance(assembly.GetType(mapping.NativeTypeName, true));
                Require(native.ComponentGuid == nativeGuid, component.Name + " native GUID/type mismatch.");
                var source = native.Icon_24x24;
                Require(source != null && source.Size == icon.Size, component.Name + " native icon size mismatch.");
                int changed = 0;
                for (int iy = 0; iy < source.Height; iy++) for (int ix = 0; ix < source.Width; ix++)
                {
                    Color a = source.GetPixel(ix, iy), b = icon.GetPixel(ix, iy);
                    Require(a.A == b.A, component.Name + " native alpha/shape was changed.");
                    if (a.A != 0 && (a.R != b.R || a.G != b.G || a.B != b.B)) changed++;
                }
                Require(changed > 0, component.Name + " icon was not recolored.");
                changedPixels += changed; mapped++;
                graphics.DrawImageUnscaled(source, x, y);
            }
            graphics.DrawImageUnscaled(icon, x + 34, y);
            graphics.DrawString(component.Name, font, Brushes.Black, x + 70, y + 3);
            rows.Add(new { component.Name, component.NickName, Guid = component.ComponentGuid, component.SubCategory, NativeGuid = nativeGuid == Guid.Empty ? (Guid?)null : nativeGuid,
                Inputs = component.Params.Input.Select(p => new { p.Name, Access = p.Access.ToString(), p.Optional }).ToArray(),
                Outputs = component.Params.Output.Select(p => new { p.Name, Access = p.Access.ToString() }).ToArray() });
        }
        Directory.CreateDirectory("artifacts/phase1");
        sheet.Save("artifacts/phase1/icons-contact-sheet.png", ImageFormat.Png);
        File.WriteAllText("artifacts/phase1/components.json", JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"PASS {components.Length} component constructors, GUIDs, ports and icons; {mapped} exact native alpha matches, {changedPixels} recolored pixels");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}
