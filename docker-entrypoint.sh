#!/bin/sh

# Function to check if a host:port is available
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local timeout=${4:-30}  # Default timeout of 30 seconds
    local start_time=$(date +%s)

    echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for $service to be ready..."
    
    while ! nc -z "$host" "$port"; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') Timeout waiting for $service after ${timeout}s"
            exit 1
        fi
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for $service to be ready... (${elapsed}s)"
        sleep 2
    done
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') $service is ready"
}

# Function to validate port number
validate_port() {
    local port=$1
    if ! echo "$port" | grep -qE '^[0-9]+$' || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
        echo "Invalid port number: $port"
        exit 1
    fi
}

# Extract host and port from Redis URL
if [ -n "$REDIS_URL" ]; then
    # Handle both redis:// and rediss:// protocols
    REDIS_PROTOCOL=$(echo "$REDIS_URL" | grep -o "^redis\(s\)\?://")
    
    # Remove protocol prefix and /0 suffix if present
    REDIS_URL_STRIPPED=$(echo "$REDIS_URL" | sed -e "s|^$REDIS_PROTOCOL||" -e 's|/.*$||')
    
    # Handle authentication credentials if present
    if echo "$REDIS_URL_STRIPPED" | grep -q "@"; then
        # Extract host:port part after the @ symbol
        REDIS_HOST_PORT=$(echo "$REDIS_URL_STRIPPED" | sed -e 's|^.*@||')
    else
        REDIS_HOST_PORT="$REDIS_URL_STRIPPED"
    fi
    
    # Extract host and port
    REDIS_HOST=$(echo "$REDIS_HOST_PORT" | sed -e 's|:.*$||')
    REDIS_PORT=$(echo "$REDIS_HOST_PORT" | sed -e 's|^.*:||')
    
    if [ -z "$REDIS_HOST" ] || [ -z "$REDIS_PORT" ]; then
        echo "Invalid REDIS_URL format: $REDIS_URL"
        exit 1
    fi
    validate_port "$REDIS_PORT"
else
    REDIS_HOST="redis"
    REDIS_PORT="6379"
fi

# Extract host and port from Kafka bootstrap servers
if [ -n "$KAFKA_BOOTSTRAP_SERVERS" ]; then
    # Handle multiple bootstrap servers (comma-separated list)
    # We'll use the first server for connection checking
    FIRST_SERVER=$(echo "$KAFKA_BOOTSTRAP_SERVERS" | cut -d, -f1)
    
    KAFKA_HOST=$(echo "$FIRST_SERVER" | cut -d: -f1)
    KAFKA_PORT=$(echo "$FIRST_SERVER" | cut -d: -f2)
    
    if [ -z "$KAFKA_HOST" ] || [ -z "$KAFKA_PORT" ]; then
        echo "Invalid KAFKA_BOOTSTRAP_SERVERS format: $KAFKA_BOOTSTRAP_SERVERS"
        exit 1
    fi
    validate_port "$KAFKA_PORT"
else
    KAFKA_HOST="kafka"
    KAFKA_PORT="9092"
fi

# Check if netcat is available
if ! command -v nc >/dev/null 2>&1; then
    echo "Error: netcat is not installed"
    exit 1
fi

# Check Redis connection
wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis"

# Check Kafka connection
wait_for_service "$KAFKA_HOST" "$KAFKA_PORT" "Kafka"

# Execute the main command
exec "$@" 