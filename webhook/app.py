#!/usr/bin/env python3
"""
Flask application for Gatekeeper webhook receiver.
Receives webhook events from GitHub/GitLab for commit analysis.
"""

import os
from datetime import datetime

from flask import Flask, jsonify, request

app = Flask(__name__)

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
            "/webhook": "POST - Receive webhook events"
        }
    }), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)