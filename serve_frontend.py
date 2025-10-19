"""
Simple static file server for the frontend
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 3000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def serve_frontend():
    """Serve the frontend on port 3000"""
    os.chdir(Path(__file__).parent)
    
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🌐 Frontend server running at http://localhost:{PORT}")
        print("📱 Mobile Price Tracker UI is ready!")
        print("🔗 Open your browser and go to http://localhost:3000")
        
        # Try to open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass
            
        httpd.serve_forever()

if __name__ == "__main__":
    import os
    serve_frontend()
