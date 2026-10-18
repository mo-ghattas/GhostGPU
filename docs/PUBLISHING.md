# Publish from your personal account

This folder is a new source snapshot with fresh Git history. The original development repository remains separate.

1. Review README.md, LICENSE and docs/KNOWN-ISSUES.md.
2. Create an empty repository named **GhostGPU** on your personal GitHub account. Choose public visibility when you are ready. Do not initialize a second README, license or gitignore on GitHub.
3. From this folder, connect and push, substituting your username:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/GhostGPU.git
git push -u origin main
```

If the initial commit has not been created, first run `git add .` and `git commit -m "Prepare public GhostGPU source"` using your normal Git identity. Never copy the old `.git` directory into this folder.

Suggested description: **Experimental GPU-assisted Rhino 8 / Grasshopper components using ILGPU, with resident data buffers and mesh/Brep geometry workflows.**

Suggested topics: `grasshopper`, `rhino3d`, `gpu-computing`, `ilgpu`, `cuda`, `csharp`, `computational-geometry`.

After pushing, check the Actions result. Create issues from K1–K5 in docs/KNOWN-ISSUES.md and link them from contributor discussions. Publish compiled releases separately after CUDA and in-Rhino verification; CI currently validates standalone CPU paths only.
