using System;
using GhostGPU.Core;
using GhostGPU.Validation.Models;

namespace GhostGPU.Validation.Tests
{
    public static class UploadDownloadTests
    {
        public static void RunAll(IGpuSession session, Action<TestResultItem> recordResult)
        {
            TestPointsUploadDownload(session, recordResult);
            TestVectorsUploadDownload(session, recordResult);
            TestScalarsUploadDownload(session, recordResult);
            TestIndicesUploadDownload(session, recordResult);
            TestColoursUploadDownload(session, recordResult);
            TestNonfiniteCoordinatesRejection(session, recordResult);
            TestNonfiniteScalarsRejection(session, recordResult);
            TestEmptyBuffers(session, recordResult);
        }

        private static void TestPointsUploadDownload(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var input = new[]
                {
                    new V3(1.0, 2.0, 3.0),
                    new V3(-10.5, 20.25, -30.125),
                    new V3(0, 0, 0),
                    new V3(1e10, 2e10, 3e10)
                };

                using var buf = session.UploadPoints(input);
                if (buf.Kind != BufferKind.Points || buf.Count != 4)
                    throw new Exception($"Buffer kind or count invalid: {buf.Kind}, count {buf.Count}");

                var downloaded = session.DownloadPoints(buf);
                if (downloaded.Length != input.Length)
                    throw new Exception($"Downloaded length mismatch: expected {input.Length}, got {downloaded.Length}");

                for (int i = 0; i < input.Length; i++)
                {
                    if (downloaded[i].X != input[i].X || downloaded[i].Y != input[i].Y || downloaded[i].Z != input[i].Z)
                        throw new Exception($"Downloaded point at {i} differs: expected ({input[i].X},{input[i].Y},{input[i].Z}), got ({downloaded[i].X},{downloaded[i].Y},{downloaded[i].Z})");
                }

                record(new TestResultItem
                {
                    Name = "Points_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Exact match across 4 points with varying scales."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Points_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message,
                    Details = ex.ToString()
                });
            }
        }

        private static void TestVectorsUploadDownload(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var input = new[] { new V3(0, 1, 0), new V3(-5, -5, 5), new V3(0, 0, 0) };
                using var buf = session.UploadVectors(input);
                if (buf.Kind != BufferKind.Vectors || buf.Count != 3)
                    throw new Exception("Buffer kind or count mismatch.");

                var downloaded = session.DownloadVectors(buf);
                if (downloaded.Length != 3)
                    throw new Exception("Downloaded length mismatch.");

                for (int i = 0; i < input.Length; i++)
                {
                    if (downloaded[i].X != input[i].X || downloaded[i].Y != input[i].Y || downloaded[i].Z != input[i].Z)
                        throw new Exception($"Mismatch at index {i}");
                }

                record(new TestResultItem
                {
                    Name = "Vectors_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Vectors match bitwise."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Vectors_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestScalarsUploadDownload(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var input = new[] { 0.0, -100.5, 3.141592653589793, 1e20, 1e-20 };
                using var buf = session.UploadScalars(input);
                if (buf.Kind != BufferKind.Scalars || buf.Count != input.Length)
                    throw new Exception("Buffer metadata mismatch.");

                var downloaded = session.DownloadScalars(buf);
                for (int i = 0; i < input.Length; i++)
                {
                    if (downloaded[i] != input[i])
                        throw new Exception($"Scalar at {i} differs: expected {input[i]}, got {downloaded[i]}");
                }

                record(new TestResultItem
                {
                    Name = "Scalars_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Double-precision scalars preserved exact bitwise."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Scalars_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestIndicesUploadDownload(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var input = new[] { 0, 1, 2, 42, 999999, -1 };
                using var buf = session.UploadIndices(input);
                var downloaded = session.DownloadIndices(buf);

                for (int i = 0; i < input.Length; i++)
                {
                    if (downloaded[i] != input[i])
                        throw new Exception($"Index mismatch at {i}");
                }

                record(new TestResultItem
                {
                    Name = "Indices_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Indices_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestColoursUploadDownload(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var input = new uint[] { 0xFF000000, 0xFFFFFFFF, 0xFFFF0000, 0x0000FF00, 0x12345678 };
                using var buf = session.UploadColours(input);
                var downloaded = session.DownloadColours(buf);

                for (int i = 0; i < input.Length; i++)
                {
                    if (downloaded[i] != input[i])
                        throw new Exception($"Colour mismatch at {i}: expected 0x{input[i]:X8}, got 0x{downloaded[i]:X8}");
                }

                record(new TestResultItem
                {
                    Name = "Colours_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Colours_Upload_Download_Integrity",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestNonfiniteCoordinatesRejection(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                bool threwNaN = false;
                try
                {
                    session.UploadPoints(new[] { new V3(0, double.NaN, 0) });
                }
                catch (ArgumentException)
                {
                    threwNaN = true;
                }

                bool threwInf = false;
                try
                {
                    session.UploadVectors(new[] { new V3(double.PositiveInfinity, 0, 0) });
                }
                catch (ArgumentException)
                {
                    threwInf = true;
                }

                if (!threwNaN || !threwInf)
                    throw new Exception($"Expected ArgumentException for nonfinite coordinates. ThrewNaN={threwNaN}, ThrewInf={threwInf}");

                record(new TestResultItem
                {
                    Name = "Nonfinite_Coordinates_Rejection",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Nonfinite coordinates correctly rejected with ArgumentException."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Nonfinite_Coordinates_Rejection",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestNonfiniteScalarsRejection(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                bool threw = false;
                try
                {
                    session.UploadScalars(new[] { 1.0, double.NaN, 3.0 });
                }
                catch (ArgumentException)
                {
                    threw = true;
                }

                if (!threw)
                    throw new Exception("Expected ArgumentException when uploading NaN scalar.");

                record(new TestResultItem
                {
                    Name = "Nonfinite_Scalars_Rejection",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Nonfinite_Scalars_Rejection",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }

        private static void TestEmptyBuffers(IGpuSession session, Action<TestResultItem> record)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                using var pts = session.UploadPoints(Array.Empty<V3>());
                using var vec = session.UploadVectors(Array.Empty<V3>());
                using var scl = session.UploadScalars(Array.Empty<double>());
                using var idx = session.UploadIndices(Array.Empty<int>());
                using var col = session.UploadColours(Array.Empty<uint>());

                if (pts.Count != 0 || vec.Count != 0 || scl.Count != 0 || idx.Count != 0 || col.Count != 0)
                    throw new Exception("Empty buffers must have Count == 0.");

                if (pts.IsDisposed || vec.IsDisposed || scl.IsDisposed || idx.IsDisposed || col.IsDisposed)
                    throw new Exception("Empty buffers must not be marked IsDisposed.");

                if (session.DownloadPoints(pts).Length != 0 ||
                    session.DownloadVectors(vec).Length != 0 ||
                    session.DownloadScalars(scl).Length != 0 ||
                    session.DownloadIndices(idx).Length != 0 ||
                    session.DownloadColours(col).Length != 0)
                    throw new Exception("Downloading empty buffers must yield empty arrays.");

                record(new TestResultItem
                {
                    Name = "Empty_Buffers_ValidHandles",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Pass,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Details = "Zero-length buffers are valid handles and produce zero-length downloads."
                });
            }
            catch (Exception ex)
            {
                record(new TestResultItem
                {
                    Name = "Empty_Buffers_ValidHandles",
                    Category = "UploadDownload",
                    Outcome = TestOutcome.Fail,
                    DurationMs = sw.Elapsed.TotalMilliseconds,
                    Message = ex.Message
                });
            }
        }
    }
}
