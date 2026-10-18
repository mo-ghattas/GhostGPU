using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace GhostGPU.Validation.Models
{
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum TestOutcome
    {
        Pass,
        Fail,
        Skipped
    }

    public sealed class TestResultItem
    {
        public string Name { get; set; } = string.Empty;
        public string Category { get; set; } = string.Empty;
        public TestOutcome Outcome { get; set; }
        public double DurationMs { get; set; }
        public string? Message { get; set; }
        public string? Details { get; set; }
    }

    public sealed class BenchmarkResultItem
    {
        public string Operation { get; set; } = string.Empty;
        public int ElementCount { get; set; }
        public int TargetCount { get; set; }
        public double DeviceDurationMs { get; set; }
        public double NaiveCpuDurationMs { get; set; }
        public double Speedup { get; set; }
        public double MaxAbsoluteDifference { get; set; }
        public string CpuReferenceDescription { get; set; } = "Explicit naive unvectorized O(N) or O(N*M) CPU reference loop; no KD-tree or SIMD optimization.";
    }

    public sealed class MetricsSnapshot
    {
        public long UploadedBytes { get; set; }
        public long DownloadedBytes { get; set; }
        public long ActiveBytes { get; set; }
        public long PeakBytes { get; set; }
        public long LiveAllocations { get; set; }
        public long KernelDispatches { get; set; }
        public long Synchronizations { get; set; }
    }

    public sealed class EnvironmentInfo
    {
        public string MachineName { get; set; } = Environment.MachineName;
        public string OSVersion { get; set; } = Environment.OSVersion.ToString();
        public string ClrVersion { get; set; } = Environment.Version.ToString();
        public int ProcessorCount { get; set; } = Environment.ProcessorCount;
        public bool Is64BitProcess { get; set; } = Environment.Is64BitProcess;
    }

    public sealed class ValidationReport
    {
        public DateTime TimestampUtc { get; set; } = DateTime.UtcNow;
        public string BackendRequested { get; set; } = string.Empty;
        public string BackendActual { get; set; } = string.Empty;
        public string DeviceName { get; set; } = string.Empty;
        public bool IsHardwareAccelerated { get; set; }
        public bool StrictCudaModeRequired { get; set; }
        public int TotalTests { get; set; }
        public int Passed { get; set; }
        public int Failed { get; set; }
        public int Skipped { get; set; }
        public double TotalDurationMs { get; set; }
        public EnvironmentInfo Environment { get; set; } = new();
        public List<TestResultItem> TestResults { get; set; } = new();
        public List<BenchmarkResultItem> BenchmarkResults { get; set; } = new();
        public MetricsSnapshot? FinalMetrics { get; set; }
    }
}
