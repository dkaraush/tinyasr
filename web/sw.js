// Cross-origin isolation for a static host (GitHub Pages can't set headers): every response this page and its workers
// get comes back with COOP / COEP, so SharedArrayBuffer and WebAssembly threads work (index.html registers it).
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", ev => ev.waitUntil(self.clients.claim()));
self.addEventListener("fetch", ev => {
  const r = ev.request;
  if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;
  ev.respondWith(fetch(r).then(res => {
    if (res.status === 0) return res;  // opaque: as it is
    const h = new Headers(res.headers);
    h.set("Cross-Origin-Opener-Policy", "same-origin");
    h.set("Cross-Origin-Embedder-Policy", "require-corp");
    h.set("Cross-Origin-Resource-Policy", "cross-origin");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }));
});
