import os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(OUT, exist_ok=True)


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Allow-Local-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        name = os.path.basename(self.path.strip("/")) or "payload.txt"
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(body)
        self.send_response(200)
        self._cors()
        self.end_headers()
        self.wfile.write(f"saved {name} {n} bytes".encode())
        sys.stderr.write(f"saved {name} {n} bytes\n")

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", 8765), H).serve_forever()
