using System;
using System.Collections.Generic;

namespace GhostGPU.Kernel.Validation
{
    public enum TestOutcome
    {
        Pass,
        Fail,
        NotRun,
        Skipped
    }

    public sealed class TestCaseResult
    {
        public string Name { get; set; }
        public string Milestone { get; set; }
        public string Backend { get; set; }
        public TestOutcome Outcome { get; set; }
        public double DurationMs { get; set; }
        public double MaxError { get; set; }
        public string Message { get; set; }
        public string Details { get; set; }
    }

    public sealed class MilestoneReport
    {
        public string Milestone { get; set; }
        public string Status { get; set; }
        public int TotalTests { get; set; }
        public int Passed { get; set; }
        public int Failed { get; set; }
        public int NotRun { get; set; }
        public int Skipped { get; set; }
        public List<TestCaseResult> Tests { get; set; } = new List<TestCaseResult>();
    }

    public sealed class ValidationReport
    {
        public string TimestampUtc { get; set; }
        public string BackendRequested { get; set; }
        public string BackendActual { get; set; }
        public string DeviceName { get; set; }
        public bool IsHardwareAccelerated { get; set; }
        public int TotalTests { get; set; }
        public int Passed { get; set; }
        public int Failed { get; set; }
        public int NotRun { get; set; }
        public int Skipped { get; set; }
        public double TotalDurationMs { get; set; }
        public List<MilestoneReport> Milestones { get; set; } = new List<MilestoneReport>();
        public List<TestCaseResult> TestResults { get; set; } = new List<TestCaseResult>();
    }
}
