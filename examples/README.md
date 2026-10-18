# Example definitions

Install the built GhostGPU plug-in before opening these definitions in Rhino 8/Grasshopper. Saved handles are recomputed; no live device allocation is embedded in a file.

| Definition | Purpose |
| --- | --- |
| GhostGPU-Attractors.gh | Start here: connected point/mesh attractor workflow |
| GhostGPU-CPU-vs-GPU.gh | GPU and ordinary CPU nodes with shared inputs |
| GhostGPU-Benchmark-GPU-vs-CPU.gh | Larger comparison definition from development |
| GhostGPU-MemoryConfig.gh | Session budget, residency and format options |
| GhostGPU-MeshBooleans.gh | Mesh Boolean examples |
| GhostGPU-SolidBooleans.gh | Brep Boolean examples |
| GhostGPU-Showcase.gh | Combined component showcase |
| GhostGPU-Showcase-Clusters.gh | Clustered showcase; sessions are scoped to inner GH documents |

Timing panels contain previous development measurements. Recompute and measure your own workflow before drawing performance conclusions. Full-path timings should include uploads, downloads and output geometry creation. Component profiling alone does not account for all asynchronous GPU work.

The definitions are development examples, not an automated regression suite. See [known issues](../docs/KNOWN-ISSUES.md), particularly cached handles and mesh adapter precision. Test with copies of important models.
