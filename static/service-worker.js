const CACHE_NAME = 'essential-v1';
const RUNTIME_CACHE = 'essential-runtime-v1';

// Essential resources to cache
const PRECACHE_URLS = [
  '/',
  '/manifest.json',
];

// Install event - precache essential resources
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[Service Worker] Precaching app shell');
        return cache.addAll(PRECACHE_URLS);
      })
      .catch((error) => {
        console.error('[Service Worker] Precaching failed:', error);
      })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  const currentCaches = [CACHE_NAME, RUNTIME_CACHE];
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (!currentCaches.includes(cacheName)) {
            console.log('[Service Worker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch event - network first strategy for Streamlit
self.addEventListener('fetch', (event) => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // Skip WebSocket connections (Streamlit uses these)
  if (event.request.url.includes('/_stcore/stream')) {
    return;
  }

  // Skip API calls to external services
  if (event.request.url.includes('api.ocr.space') || 
      event.request.url.includes('groq') ||
      event.request.url.includes('qdrant')) {
    return;
  }

  event.respondWith(
    // Network first, fall back to cache
    fetch(event.request)
      .then((response) => {
        // Don't cache if not successful
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }

        // Clone the response
        const responseToCache = response.clone();

        // Cache static assets
        if (event.request.url.includes('/static/') || 
            event.request.url.includes('.css') ||
            event.request.url.includes('.js') ||
            event.request.url.includes('.png') ||
            event.request.url.includes('.jpg')) {
          caches.open(RUNTIME_CACHE)
            .then((cache) => {
              cache.put(event.request, responseToCache);
            });
        }

        return response;
      })
      .catch(() => {
        // Network failed, try cache
        return caches.match(event.request)
          .then((cachedResponse) => {
            if (cachedResponse) {
              return cachedResponse;
            }

            // Return a custom offline page for navigation requests
            if (event.request.mode === 'navigate') {
              return new Response(
                `<!DOCTYPE html>
                <html>
                  <head>
                    <title>Essential - Offline</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                      body {
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                        background: #f0f2f6;
                        color: #262730;
                      }
                      .container {
                        text-align: center;
                        padding: 2rem;
                      }
                      h1 { color: #ff4b4b; }
                      .icon { font-size: 4rem; margin-bottom: 1rem; }
                    </style>
                  </head>
                  <body>
                    <div class="container">
                      <div class="icon">📖</div>
                      <h1>Essential is Offline</h1>
                      <p>Please check your internet connection.</p>
                      <p>Essential requires an active connection to process documents and generate responses.</p>
                      <button onclick="window.location.reload()" 
                              style="padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; background: #ff4b4b; color: white; border: none; border-radius: 4px;">
                        Try Again
                      </button>
                    </div>
                  </body>
                </html>`,
                {
                  headers: { 'Content-Type': 'text/html' }
                }
              );
            }

            return new Response('Offline', {
              status: 503,
              statusText: 'Service Unavailable'
            });
          });
      })
  );
});

// Handle messages from clients
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
