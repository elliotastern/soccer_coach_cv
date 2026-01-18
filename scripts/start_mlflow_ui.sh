#!/bin/bash
# Start MLflow UI for viewing training experiments

cd "$(dirname "$0")/.."

TRACKING_URI=${1:-"file:./mlruns"}
PORT=${2:-5000}

echo "Starting MLflow UI..."
echo "Tracking URI: $TRACKING_URI"
echo "Port: $PORT"
echo ""
echo "Access MLflow UI at: http://localhost:$PORT"
echo ""

mlflow ui --backend-store-uri "$TRACKING_URI" --port "$PORT"
