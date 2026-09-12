#!/usr/bin/env python3
"""Simple proxy server: serves dashboard static files + proxies /api to API server."""
import http.server
import json
import urllib.request
import os
import sys

PORT = 5174
API_URL = "http://localhost:8000"
DIST_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard", "dist")

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIST_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            api_path = self.path[4:]  # strip /api/
            url = f"{API_URL}/{api_path}"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                    self.send_response(resp.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(data)
            except Exception as e:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            # For SPA: if file doesn't exist, serve index.html
            path = self.translate_path(self.path)
            if not os.path.exists(path) or os.path.isdir(path):
                self.path = "/index.html"
            super().do_GET()

    def log_message(self, format, *args):
        pass  # suppress logs

if __name__ == "__main__":
    class ReusableHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True

server = ReusableHTTPServer(("0.0.0.0", PORT), ProxyHandler)
    print(f"Dashboard proxy running on http://localhost:{PORT}")
    print(f"Proxying /api/* to {API_URL}")
    server.serve_forever()
