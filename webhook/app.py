#!/usr/bin/env python3
"""
Flask application for Gatekeeper webhook receiver.
Receives webhook events from GitHub/GitLab for commit analysis.
"""

import os
import sys
from datetime import datetime

# Ensure the parent directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request
from flask_cors import CORS

# Handle both local development and Docker context
try:
    from webhook.routes.dashboard import dashboard_bp
except ImportError:
    from routes.dashboard import dashboard_bp

app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard

# Register blueprints
app.register_blueprint(dashboard_bp)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "gatekeeper-webhook"
    }), 200

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Webhook endpoint for receiving commit events.
    
    This is a placeholder that logs the received payload.
    Will be fully implemented in a later phase.
    """
    # Log the received payload
    payload = request.get_json(force=True)
    
    print(f"[{datetime.now().isoformat()}] Received webhook payload:")
    print(f"  Headers: {dict(request.headers)}")
    print(f"  Body: {payload}")
    
    # Return success response
    return jsonify({
        "status": "received",
        "message": "Webhook payload logged successfully"
    }), 200

@app.route("/", methods=["GET"])
def root():
    """Root endpoint with service information."""
    return jsonify({
        "service": "gatekeeper-webhook",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/webhook": "POST - Receive webhook events",
            "/issues": "GET/POST - List/Create issues",
            "/issues/<id>": "PATCH - Toggle issue status",
            "/issues/stats": "GET - Issue statistics"
        }
    }), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)