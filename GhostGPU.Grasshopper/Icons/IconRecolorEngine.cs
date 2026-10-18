using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace GhostGPU.Grasshopper.Icons
{
    /// <summary>
    /// High-performance pixel transformation engine that recolors native Grasshopper component icons
    /// to the GhostGPU cyan/teal brand palette while strictly preserving native alpha channel byte-for-byte
    /// and silhouette geometry.
    /// Uses safe managed memory operations without requiring unsafe compilation flags.
    /// </summary>
    public static class IconRecolorEngine
    {
        /// <summary>
        /// Target hue in degrees (186° is GhostGPU electric cyan/teal).
        /// </summary>
        public const float TargetHue = 186.0f;

        /// <summary>
        /// Recolors a 24x24 or arbitrary-size 32bpp ARGB bitmap to GhostGPU cyan/teal identity.
        /// Preserves exact alpha transparency and antialiasing byte-for-byte.
        /// </summary>
        /// <param name="source">Source icon bitmap.</param>
        /// <returns>A new recolored Bitmap instance.</returns>
        public static Bitmap Recolor(Bitmap source)
        {
            if (source == null) return null;

            int width = source.Width;
            int height = source.Height;

            var result = new Bitmap(width, height, PixelFormat.Format32bppArgb);

            BitmapData srcData = null;
            BitmapData dstData = null;

            try
            {
                var rect = new Rectangle(0, 0, width, height);
                srcData = source.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                dstData = result.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);

                int pixelCount = width * height;
                int[] pixels = new int[pixelCount];
                Marshal.Copy(srcData.Scan0, pixels, 0, pixelCount);

                int coloredCount = 0;

                // First pass: detect if the icon contains colored elements (chroma)
                for (int i = 0; i < pixelCount; i++)
                {
                    uint pixel = (uint)pixels[i];
                    uint a = (pixel >> 24) & 0xFF;
                    if (a > 30)
                    {
                        byte r = (byte)((pixel >> 16) & 0xFF);
                        byte g = (byte)((pixel >> 8) & 0xFF);
                        byte b = (byte)(pixel & 0xFF);

                        float dummyH, s, dummyL;
                        RgbToHsl(r, g, b, out dummyH, out s, out dummyL);
                        if (s > 0.12f)
                        {
                            coloredCount++;
                        }
                    }
                }

                bool hasColor = coloredCount > 0;
                int[] outputPixels = new int[pixelCount];

                // Second pass: apply precise hue translation while preserving alpha byte-for-byte
                for (int i = 0; i < pixelCount; i++)
                {
                    uint pixel = (uint)pixels[i];
                    uint a = (pixel >> 24) & 0xFF;

                    if (a == 0)
                    {
                        // Strictly zero out transparent pixels (RGB irrelevant/cleared)
                        outputPixels[i] = 0;
                        continue;
                    }

                    byte r = (byte)((pixel >> 16) & 0xFF);
                    byte g = (byte)((pixel >> 8) & 0xFF);
                    byte b = (byte)(pixel & 0xFF);

                    float h, s, l;
                    RgbToHsl(r, g, b, out h, out s, out l);

                    byte outR, outG, outB;

                    if (hasColor)
                    {
                        // If icon has colored elements, shift hue of saturated pixels to cyan/teal
                        if (s > 0.08f)
                        {
                            float newSat = Math.Max(s, 0.50f);
                            HslToRgb(TargetHue, newSat, l, out outR, out outG, out outB);
                        }
                        else
                        {
                            // Preserve neutral outlines, pure blacks, and pure white highlights
                            outR = r;
                            outG = g;
                            outB = b;
                        }
                    }
                    else
                    {
                        // Fully monochrome icon (e.g. Construct Point, Deconstruct Point):
                        // Transform black/dark stroke pixels into visible teal while keeping white highlights.
                        if (l > 0.95f)
                        {
                            // Preserve pure white highlights
                            outR = r;
                            outG = g;
                            outB = b;
                        }
                        else
                        {
                            // Shift dark/black strokes and midtones to visible teal
                            float strokeSat = 0.75f;
                            float strokeLightness = Math.Max(l, 0.38f);
                            HslToRgb(TargetHue, strokeSat, strokeLightness, out outR, out outG, out outB);
                        }
                    }

                    // Write back with identical alpha byte
                    uint resultPixel = ((uint)a << 24) | ((uint)outR << 16) | ((uint)outG << 8) | (uint)outB;
                    outputPixels[i] = (int)resultPixel;
                }

                Marshal.Copy(outputPixels, 0, dstData.Scan0, pixelCount);
            }
            finally
            {
                if (srcData != null) source.UnlockBits(srcData);
                if (dstData != null) result.UnlockBits(dstData);
            }

            return result;
        }

        public static void RgbToHsl(byte r, byte g, byte b, out float h, out float s, out float l)
        {
            float rf = r / 255.0f;
            float gf = g / 255.0f;
            float bf = b / 255.0f;

            float max = Math.Max(rf, Math.Max(gf, bf));
            float min = Math.Min(rf, Math.Min(gf, bf));
            l = (max + min) / 2.0f;
            float delta = max - min;

            if (delta < 0.00001f)
            {
                h = 0.0f;
                s = 0.0f;
                return;
            }

            s = l > 0.5f ? delta / (2.0f - max - min) : delta / (max + min);

            if (rf >= max)
                h = (gf - bf) / delta + (gf < bf ? 6.0f : 0.0f);
            else if (gf >= max)
                h = (bf - rf) / delta + 2.0f;
            else
                h = (rf - gf) / delta + 4.0f;

            h *= 60.0f;
        }

        public static void HslToRgb(float h, float s, float l, out byte r, out byte g, out byte b)
        {
            if (s < 0.00001f)
            {
                byte v = (byte)Math.Round(Math.Max(0.0f, Math.Min(255.0f, l * 255.0f)));
                r = g = b = v;
                return;
            }

            float q = l < 0.5f ? l * (1.0f + s) : l + s - (l * s);
            float p = 2.0f * l - q;
            float hk = h / 360.0f;

            r = (byte)Math.Round(Math.Max(0.0f, Math.Min(255.0f, HueToRgb(p, q, hk + 1.0f / 3.0f) * 255.0f)));
            g = (byte)Math.Round(Math.Max(0.0f, Math.Min(255.0f, HueToRgb(p, q, hk) * 255.0f)));
            b = (byte)Math.Round(Math.Max(0.0f, Math.Min(255.0f, HueToRgb(p, q, hk - 1.0f / 3.0f) * 255.0f)));
        }

        private static float HueToRgb(float p, float q, float t)
        {
            if (t < 0.0f) t += 1.0f;
            if (t > 1.0f) t -= 1.0f;
            if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
            if (t < 1.0f / 2.0f) return q;
            if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
            return p;
        }
    }
}
