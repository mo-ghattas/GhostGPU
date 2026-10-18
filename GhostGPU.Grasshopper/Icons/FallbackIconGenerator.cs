using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;

namespace GhostGPU.Grasshopper.Icons
{
    /// <summary>
    /// Procedural icon generator for GhostGPU components that do not have a native Grasshopper counterpart
    /// or when running in environments where native assemblies/proxies are inaccessible.
    /// Creates 24x24 pixel-perfect icons consistent with standard Grasshopper styling and the GhostGPU cyan/teal brand.
    /// Uses serialized locking to guarantee no duplicate bitmap creation or GDI+ leaks.
    /// </summary>
    public static class FallbackIconGenerator
    {
        private static readonly object _syncLock = new object();
        private static readonly Dictionary<string, Bitmap> _cache = new Dictionary<string, Bitmap>(StringComparer.OrdinalIgnoreCase);

        // Palette
        private static readonly Color BackgroundColor = Color.FromArgb(255, 32, 38, 46);
        private static readonly Color BorderColor = Color.FromArgb(255, 60, 70, 82);
        private static readonly Color CyanAccentColor = Color.FromArgb(255, 0, 229, 255);
        private static readonly Color TealDarkAccent = Color.FromArgb(255, 0, 168, 198);
        private static readonly Color TextColor = Color.FromArgb(255, 240, 246, 252);

        /// <summary>
        /// Gets or generates a cached 24x24 fallback icon with an optional label.
        /// Thread-safe and serialized to avoid race-condition duplicate allocations.
        /// </summary>
        /// <param name="label">Short 2-4 character monogram or nickname.</param>
        /// <returns>A 24x24 32-bit ARGB Bitmap.</returns>
        public static Bitmap CreateFallbackIcon(string label = null)
        {
            string key = string.IsNullOrWhiteSpace(label) ? "GPU" : label.Trim().ToUpperInvariant();
            if (key.Length > 4) key = key.Substring(0, 4);

            lock (_syncLock)
            {
                if (_cache.TryGetValue(key, out Bitmap cached))
                {
                    return cached;
                }

                Bitmap generated = GenerateIcon(key);
                _cache[key] = generated;
                return generated;
            }
        }

        private static Bitmap GenerateIcon(string text)
        {
            const int width = 24;
            const int height = 24;

            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);

            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

                g.Clear(Color.Transparent);

                // Rounded badge rectangle
                var badgeRect = new Rectangle(1, 1, 21, 21);
                using (var path = CreateRoundedRectanglePath(badgeRect, 3))
                {
                    // Gradient fill (subtle dark slate to deeper slate)
                    using (var fillBrush = new LinearGradientBrush(
                        badgeRect,
                        BackgroundColor,
                        Color.FromArgb(255, 20, 24, 30),
                        LinearGradientMode.Vertical))
                    {
                        g.FillPath(fillBrush, path);
                    }

                    // Border
                    using (var borderPen = new Pen(BorderColor, 1.0f))
                    {
                        g.DrawPath(borderPen, path);
                    }
                }

                // Cyan/teal top accent line
                using (var accentPen = new Pen(CyanAccentColor, 1.5f))
                {
                    g.DrawLine(accentPen, 4, 3, 19, 3);
                }

                // Cyan GPU indicator dot in bottom right
                using (var cyanBrush = new SolidBrush(CyanAccentColor))
                {
                    g.FillEllipse(cyanBrush, 17, 17, 3, 3);
                }

                // Render monogram text
                float fontSize = text.Length <= 2 ? 8.0f : (text.Length == 3 ? 6.5f : 5.5f);
                using (var font = new Font(FontFamily.GenericSansSerif, fontSize, FontStyle.Bold, GraphicsUnit.Pixel))
                using (var textBrush = new SolidBrush(TextColor))
                using (var sf = new StringFormat
                {
                    Alignment = StringAlignment.Center,
                    LineAlignment = StringAlignment.Center,
                    FormatFlags = StringFormatFlags.NoWrap
                })
                {
                    var textRect = new RectangleF(1, 4, 21, 16);
                    g.DrawString(text, font, textBrush, textRect, sf);
                }
            }

            return bmp;
        }

        private static GraphicsPath CreateRoundedRectanglePath(Rectangle rect, int radius)
        {
            var path = new GraphicsPath();
            int diameter = radius * 2;
            var arc = new Rectangle(rect.X, rect.Y, diameter, diameter);

            // Top-left
            path.AddArc(arc, 180, 90);

            // Top-right
            arc.X = rect.Right - diameter;
            path.AddArc(arc, 270, 90);

            // Bottom-right
            arc.Y = rect.Bottom - diameter;
            path.AddArc(arc, 0, 90);

            // Bottom-left
            arc.X = rect.Left;
            path.AddArc(arc, 90, 90);

            path.CloseFigure();
            return path;
        }

        public static void ClearCache()
        {
            lock (_syncLock)
            {
                _cache.Clear();
            }
        }
    }
}
