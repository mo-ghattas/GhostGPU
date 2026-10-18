using System;
using System.Collections.Concurrent;
using Grasshopper;
using Grasshopper.Kernel;
using GhostGPU.Core;
using GhostGPU.Runtime;

namespace GhostGPU.Grasshopper.Session
{
    /// <summary>
    /// Scopes IGpuSession lifetimes to Grasshopper documents.
    /// Strictly isolates sessions per GH_Document and rejects null/ambient document sharing.
    /// Cleans up device memory when documents close or are removed from the server.
    /// Enforces CUDA default execution and explicit error reporting if CUDA is unavailable.
    /// </summary>
    public static class GpuSessionManager
    {
        private static readonly ConcurrentDictionary<Guid, IGpuSession> DocumentSessions = new ConcurrentDictionary<Guid, IGpuSession>();
        private static readonly ConcurrentDictionary<Guid, ConcurrentBag<IDisposable>> DocumentTransientLeases = new ConcurrentDictionary<Guid, ConcurrentBag<IDisposable>>();
        private static readonly ConcurrentDictionary<Guid, bool> HookedDocumentEvents = new ConcurrentDictionary<Guid, bool>();
        private static readonly object LockObj = new object();
        private static bool _eventsHooked;

        public static GpuBackend PreferredBackend { get; set; } = GpuBackend.CudaOnly;

        static GpuSessionManager()
        {
            EnsureEventsHooked();
        }

        public static void EnsureEventsHooked()
        {
            if (_eventsHooked) return;
            lock (LockObj)
            {
                if (_eventsHooked) return;
                try
                {
                    if (Instances.DocumentServer != null)
                    {
                        Instances.DocumentServer.DocumentRemoved += OnDocumentRemoved;
                        _eventsHooked = true;
                    }
                }
                catch
                {
                    // Running headless or in test harness without UI DocumentServer
                }
            }
        }

        private static void HookDocumentEvents(GH_Document doc)
        {
            if (doc == null) return;
            if (HookedDocumentEvents.TryAdd(doc.DocumentID, true))
            {
                doc.SolutionEnd += OnDocumentSolutionEnd;
            }
        }

        private static void OnDocumentSolutionEnd(object sender, GH_SolutionEventArgs e)
        {
            if (sender is GH_Document doc && DocumentTransientLeases.TryRemove(doc.DocumentID, out var bag))
            {
                foreach (var item in bag)
                {
                    try { item?.Dispose(); } catch { }
                }
            }
        }

        /// <summary>
        /// Registers a transient lease (such as a duplicated GH_GpuBuffer) to be deterministically
        /// disposed at the end of the current document solution pass.
        /// </summary>
        public static void RegisterTransientLease(GH_Document doc, IDisposable lease)
        {
            if (doc == null || lease == null) return;
            HookDocumentEvents(doc);
            var bag = DocumentTransientLeases.GetOrAdd(doc.DocumentID, _ => new ConcurrentBag<IDisposable>());
            bag.Add(lease);
        }

        private static void OnDocumentRemoved(GH_DocumentServer server, GH_Document doc)
        {
            if (doc != null)
            {
                if (DocumentTransientLeases.TryRemove(doc.DocumentID, out var bag))
                {
                    foreach (var item in bag)
                    {
                        try { item?.Dispose(); } catch { }
                    }
                }
                HookedDocumentEvents.TryRemove(doc.DocumentID, out _);
                CloseSession(doc.DocumentID);
            }
        }

        /// <summary>
        /// Retrieves or creates an IGpuSession associated strictly with the given GH_Document.
        /// Rejects null documents to prevent ambient or cross-document memory leakage.
        /// </summary>
        public static IGpuSession GetSession(GH_Document doc, SessionOptions options = null)
        {
            if (doc == null)
            {
                throw new ArgumentNullException(nameof(doc),
                    "[GhostGPU] Grasshopper document context is required for GPU session scoping. " +
                    "Components must belong to a valid GH_Document so device allocations can be deterministically cleaned up.");
            }

            EnsureEventsHooked();
            HookDocumentEvents(doc);
            Guid docId = doc.DocumentID;

            if (DocumentSessions.TryGetValue(docId, out var existing) && existing != null)
            {
                return existing;
            }

            lock (LockObj)
            {
                if (DocumentSessions.TryGetValue(docId, out existing) && existing != null)
                {
                    return existing;
                }

                var sessionOptions = options ?? new SessionOptions
                {
                    Backend = GetConfiguredBackend()
                };

                IGpuSession created;
                try
                {
                    created = new GpuSession(sessionOptions);
                }
                catch (Exception ex)
                {
                    if (sessionOptions.Backend == GpuBackend.CudaOnly)
                    {
                        throw new InvalidOperationException(
                            $"[GhostGPU] Real CUDA hardware acceleration is required by default, but initialization failed: {ex.Message}. " +
                            "No silent CPU fallback is allowed. For explicit testing/validation, set backend to CpuDebug explicitly.", ex);
                    }
                    throw;
                }

                DocumentSessions[docId] = created;
                return created;
            }
        }

        /// <summary>
        /// Recreates the IGpuSession for the specified GH_Document with the provided SessionOptions.
        /// Existing leases in the old session are disposed.
        /// </summary>
        public static IGpuSession RecreateSession(GH_Document doc, SessionOptions options)
        {
            if (doc == null)
            {
                throw new ArgumentNullException(nameof(doc),
                    "[GhostGPU] Grasshopper document context is required for GPU session scoping.");
            }

            EnsureEventsHooked();
            HookDocumentEvents(doc);
            Guid docId = doc.DocumentID;

            lock (LockObj)
            {
                if (DocumentSessions.TryRemove(docId, out var oldSession) && oldSession != null)
                {
                    try { oldSession.Dispose(); } catch { }
                }

                var sessionOptions = options ?? new SessionOptions
                {
                    Backend = GetConfiguredBackend()
                };

                IGpuSession created;
                try
                {
                    created = new GpuSession(sessionOptions);
                }
                catch (Exception ex)
                {
                    if (sessionOptions.Backend == GpuBackend.CudaOnly)
                    {
                        throw new InvalidOperationException(
                            $"[GhostGPU] Real CUDA hardware acceleration is required by default, but initialization failed: {ex.Message}. " +
                            "No silent CPU fallback is allowed. For explicit testing/validation, set backend to CpuDebug explicitly.", ex);
                    }
                    throw;
                }

                DocumentSessions[docId] = created;
                return created;
            }
        }

        private static GpuBackend GetConfiguredBackend()
        {
            string env = Environment.GetEnvironmentVariable("GHOSTGPU_BACKEND");
            if (!string.IsNullOrEmpty(env) && string.Equals(env, "CpuDebug", StringComparison.OrdinalIgnoreCase))
            {
                return GpuBackend.CpuDebug;
            }
            return PreferredBackend;
        }

        /// <summary>
        /// Closes and disposes the session associated with the specified document ID.
        /// </summary>
        public static void CloseSession(Guid docId)
        {
            if (DocumentSessions.TryRemove(docId, out var session) && session != null)
            {
                try
                {
                    session.Dispose();
                }
                catch
                {
                    // Suppress during document shutdown
                }
            }
        }

        /// <summary>
        /// Closes and disposes the session associated with the specified GH_Document.
        /// </summary>
        public static void CloseSession(GH_Document doc)
        {
            if (doc != null)
            {
                CloseSession(doc.DocumentID);
            }
        }

        /// <summary>
        /// Resets the session for the given document, invalidating all old handles.
        /// </summary>
        public static void ResetSession(GH_Document doc)
        {
            if (doc == null) return;
            if (DocumentSessions.TryGetValue(doc.DocumentID, out var session) && session != null)
            {
                session.Reset();
            }
        }

        /// <summary>
        /// Cleans up all active document sessions.
        /// </summary>
        public static void DisposeAll()
        {
            lock (LockObj)
            {
                foreach (var kvp in DocumentSessions)
                {
                    try
                    {
                        kvp.Value?.Dispose();
                    }
                    catch { }
                }
                DocumentSessions.Clear();
            }
        }
    }
}
