#!/bin/bash
# Start CVAT using Docker Compose

echo "Starting CVAT..."
cd "$(dirname "$0")"

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "Error: docker-compose or 'docker compose' not found"
    exit 1
fi

# Start CVAT
$COMPOSE_CMD -f docker-compose.cvat.yml up -d

echo ""
echo "Waiting for CVAT to start..."
sleep 10

# Check if CVAT is running
if curl -s http://localhost:8080/api/server/about > /dev/null 2>&1; then
    echo "✅ CVAT is running at http://localhost:8080"
    echo ""
    echo "Default credentials:"
    echo "  Username: admin"
    echo "  Password: admin"
    echo ""
    echo "To stop CVAT: docker-compose -f docker-compose.cvat.yml down"
else
    echo "⚠️  CVAT may still be starting. Check logs with:"
    echo "   docker-compose -f docker-compose.cvat.yml logs -f"
fi
