# Third-party notices

The root MIT license applies to GhostGPU project code. Dependencies keep their own licenses and notices.

- **ILGPU 1.5.3** and **ILGPU.Algorithms 1.5.3**: restored through NuGet. Project and license: https://github.com/m4rs-mt/ILGPU. Release staging copies available package license/notice files and NuGet metadata into `licenses/`.
- **RhinoCommon, Grasshopper and GH_IO**: referenced from an installed Rhino 8 for interoperability. They are not redistributed in this source repository or the staged plug-in package. Their use remains subject to the applicable vendor terms.
- **.NET and other transitive NuGet packages**: restored through the SDK/NuGet; their package metadata and licenses remain authoritative. Release staging includes available notices for the plug-in's restored packages.

## fdlibm mathematical routines

`GhostGPU.Runtime/Kernels/ScalarKernels.cs` uses fdlibm-style sine/cosine polynomials and argument reduction. The upstream fdlibm notice is retained here for attribution:

```text
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
```

Upstream source: https://www.netlib.org/fdlibm/.

## Test data and contributions

`tests/fixtures/golden` contains numeric geometry and expected results exported through RhinoCommon. The exporter is `scripts/ExportGolden.csx`. These are test fixtures, not vendor source or executable assemblies.

Only submit code, models and other material that you have the right to contribute under this project's license. Keep required third-party attribution with any contributed code. Do not add proprietary vendor source, decompiled listings, application binaries, activation files or private customer geometry.
