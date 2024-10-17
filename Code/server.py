import http.server
import socketserver
import json

# Define the port on which the server will listen.
PORT = 8001

class Handler(http.server.SimpleHTTPRequestHandler):
    """
    Custom request handler to serve JSON data for specific endpoints.
    """

    def do_GET(self):
        """
        Handle GET requests. Serves JSON data from 'gauge_value.json'
        when the '/data' endpoint is accessed. For other requests, 
        it behaves as a default HTTP GET handler.
        """
        if self.path == '/data':
            # Send HTTP response status and headers.
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            try:
                # Read JSON data from 'gauge_value.json'.
                with open("gauge_value.json", "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                # Provide default data if the file is not found.
                data = {"value": 0}

            # Send the JSON response to the client.
            self.wfile.write(json.dumps(data).encode())
        else:
            # Default behavior for other paths.
            super().do_GET()

def run_server():
    """
    Initialize and run the HTTP server.
    """
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Run the server when the script is executed.
    run_server()
