# Mechafil Cache Updater System

A distributed caching system for mechafil-server that separates data fetching from computation, enabling serverless deployment patterns.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐
│ Cache Updater   │    │ Mechafil Server  │
│ (Always On)     │    │ (On Demand)      │
│                 │    │                  │
│ Fetches Data ──────┐ │ Reads Cache ──┐  │
└─────────────────┘  │ └──────────────────┘
                     │                  │
                     ▼                  ▼
              ┌─────────────────────────────┐
              │    Shared Volume           │
              │   /data/shared-cache       │
              │                           │
              │  ┌─────────────────────┐   │
              │  │ DiskCache files     │   │
              │  │ (pickle format)     │   │
              │  └─────────────────────┘   │
              └─────────────────────────────┘
```

### Components

- **Cache Updater**: Background service that fetches data from APIs and maintains cache
- **Mechafil Server**: Stateless web service that reads from cache and runs simulations
- **Shared Volume**: Persistent storage shared between both services

### Benefits

- ✅ **Serverless Ready**: Mechafil server can scale to zero
- ✅ **Cost Effective**: Single data fetch serves multiple server instances
- ✅ **No HTTP Serialization Issues**: Uses native DiskCache pickle format
- ✅ **Simple Architecture**: Direct file system access, no network calls

## 🚀 Local Development

### Prerequisites

- Docker and Docker Compose
- Spacescope API credentials (for data fetching)

### Quick Start

1. **Navigate to cache updater directory:**
   ```bash
   cd /Users/luca/programmi/cel/mechafil/programs/mechafil-cache-updater
   ```

2. **Set up environment variables:**
   ```bash
   # Create .env file with your credentials
   echo "SPACESCOPE_TOKEN=your_token_here" >> .env
   echo "RELOAD_TEST_MODE=true" >> .env
   ```

3. **Choose your deployment pattern:**

### 🚀 **Serverless Pattern (Recommended - AWS Lambda Equivalent)**

This pattern mimics AWS API Gateway + Lambda + S3 + CloudWatch Events:

**Option A: Lambda-like Cache Updater (Runs once and exits)**
```bash
# 🔄 Run volume updater locally (creates ./shared-cache directory)
SHARED_CACHE_DIR=./shared-cache poetry run python -m cache_updater.main --once

# 🔄 Run volume updater in Docker (updates shared volume)
docker-compose run --rm cache-updater python -m cache_updater.main --once

# 📅 Schedule with cron (production-like)
# Add to your crontab: 0 1 * * * cd /path/to/cache-updater && poetry run python -m cache_updater.main --once
```

**How the volume updater works:**
- Downloads historical data from Spacescope APIs
- Processes and stores data in DiskCache format (pickle)
- Creates cache entries for date ranges
- Volume persists between runs (equivalent to S3 bucket)
- Each run is completely independent (Lambda-like execution)

**Option B: Background Service (Traditional always-on)**
```bash
# Start cache updater as background service
docker-compose up cache-updater -d --build
```

**Start mechafil server on-demand:**
```bash
# 🚀 Start server when needed (API Gateway + Lambda equivalent)
docker-compose up mechafil-server

# 🌐 Use the server (accesses shared volume automatically)
curl http://localhost:8000/health
curl http://localhost:8000/historical-data

# 🛑 Stop server when done (Ctrl+C or in another terminal)
docker-compose stop mechafil-server

# 💾 Volume data persists (S3 equivalent)
```

**How mechafil-server accesses the volume:**
- Reads cache files directly from `/data/shared-cache` mount point
- Uses same DiskCache keys as cache updater
- Falls back to error if cache is missing (no direct API calls)
- Completely stateless - can scale to zero
- Volume remains accessible even when server is stopped

**🎯 Perfect AWS Serverless Equivalent:**
- **Cache Updater** = Lambda + CloudWatch Events (runs once daily)
- **Mechafil Server** = API Gateway + Lambda (on-demand execution)  
- **Shared Volume** = S3 Bucket (persistent data storage)

### 🛠️ **Development Pattern**

**Start both services together:**
```bash
docker-compose up --build
```

### 🏭 **Production Pattern**

**Cache only (for managed server deployments):**
```bash
docker-compose up cache-updater -d
```

4. **Test the volume updater and server access:**
   ```bash
   # 🔄 Test volume updater (Lambda-like execution)
   docker-compose run --rm cache-updater python -m cache_updater.main --once
   
   # 🚀 Start mechafil server (reads from volume)
   docker-compose up mechafil-server
   
   # 🌐 Test server endpoints (volume access)
   curl http://localhost:8000/health
   curl http://localhost:8000/historical-data
   
   # 🧮 Run simulation (uses cached data)
   curl -X POST http://localhost:8000/simulate \
     -H "Content-Type: application/json" \
     -d '{
       "start_date": "2024-01-01",
       "current_date": "2024-12-01", 
       "forecast_length": 30,
       "rbp": 0.5,
       "rr": 0.6,
       "fpr": 0.7
     }'
   
   # 📋 Check volume contents
   docker-compose exec mechafil-server ls -la /data/shared-cache
   ```

5. **Manage services:**
   ```bash
   # Stop server only (keep cache running)
   docker-compose stop mechafil-server
   
   # Stop everything
   docker-compose down
   
   # Check what's running
   docker-compose ps
   ```

### Environment Variables

**Cache Updater:**
- `SPACESCOPE_TOKEN`: API token for data access
- `SPACESCOPE_AUTH_FILE`: Alternative auth file path
- `RELOAD_TEST_MODE`: Set to `true` for 2-minute refresh (testing)
- `RELOAD_TRIGGER`: Daily refresh time in HH:MM format (default: "01:00" UTC)
- `SHARED_CACHE_DIR`: Cache directory path (default: "/data/shared-cache")

**Mechafil Server:**
- `USE_SHARED_CACHE`: Must be `true` to enable shared cache mode
- `SHARED_CACHE_DIR`: Cache directory path (default: "/data/shared-cache")

### Alternative: Testing Without API Credentials

If you don't have API credentials, you can use pre-cached test data:

```bash
# 1. Create test volume with existing cache data
docker volume create test_cache_vol
docker run --rm \
  -v test_cache_vol:/target \
  -v /Users/luca/programmi/cel/mechafil/programs/test-cache:/source \
  alpine cp -r /source/. /target/

# 2. Run only mechafil-server with test data
docker run -d \
  --name mechafil-server \
  -v test_cache_vol:/data/shared-cache \
  -p 8000:8000 \
  -e USE_SHARED_CACHE=true \
  -e SHARED_CACHE_DIR=/data/shared-cache \
  mechafil-server

# 3. Test
curl http://localhost:8000/health
```

## 📋 Quick Reference

### Volume Updater Commands

```bash
# 🔄 LAMBDA-LIKE VOLUME UPDATES
docker-compose run --rm cache-updater python -m cache_updater.main --once  # Update once and exit
poetry run python -m cache_updater.main --once                             # Update locally

# 📅 SCHEDULE VOLUME UPDATES
# Crontab entry: 0 1 * * * cd /path/to/cache-updater && poetry run python -m cache_updater.main --once

# 🔍 INSPECT VOLUME
docker-compose exec mechafil-server ls -la /data/shared-cache              # Check volume contents
docker volume inspect mechafil-cache-updater_shared_cache                   # Volume details
```

### Server Access Commands

```bash
# 🚀 ON-DEMAND SERVER (reads from volume)
docker-compose up mechafil-server             # Start server (reads volume)
curl http://localhost:8000/health             # Test volume access
curl http://localhost:8000/historical-data    # View cached data
docker-compose stop mechafil-server           # Stop server (volume persists)

# 🛠️ DEVELOPMENT MODE  
docker-compose up --build                     # Start both services
docker-compose down                           # Stop everything

# 📊 MONITORING
docker-compose ps                             # Check running services
docker-compose logs cache-updater             # Cache updater logs
docker-compose logs mechafil-server           # Server logs

# 🧹 CLEANUP
docker-compose down                           # Stop and remove containers
docker-compose down -v                        # Stop and remove volumes too
```

### Service Architecture

| Service | Purpose | When Running | Restart Policy |
|---------|---------|--------------|----------------|
| `cache-updater` | Data fetching & caching | Always (24/7) | `unless-stopped` |
| `mechafil-server` | API & simulations | On-demand | Manual |

## ☁️ Fly.io Deployment

### Overview

Deploy both services to Fly.io with a shared persistent volume:

```
┌─────────────────────────────────────┐
│         Fly.io Machine              │
│                                     │
│  ┌─────────────────┐                │
│  │ Cache Updater   │                │
│  │ (Always On)     │                │
│  └─────────────────┘                │
│           │                         │
│  ┌─────────────────┐                │
│  │ Mechafil Server │                │
│  │ (Auto Scale)    │                │
│  └─────────────────┘                │
│           │                         │
│  ┌─────────────────────────────────┐ │
│  │    Fly.io Volume               │ │
│  │    /data/shared-cache          │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Prerequisites

1. **Install Fly CLI:**
   ```bash
   # macOS
   brew install flyctl
   
   # Or download from https://fly.io/docs/hands-on/install-flyctl/
   ```

2. **Login to Fly.io:**
   ```bash
   fly auth login
   ```

### Step 1: Deploy Cache Updater Service (Lambda-like)

1. **Create cache updater app:**
   ```bash
   cd /Users/luca/programmi/cel/mechafil/programs/mechafil-cache-updater
   fly apps create mechafil-cache-updater
   ```

2. **Create persistent volume:**
   ```bash
   fly volumes create shared_cache --size 1 --region ord
   ```

3. **Set secrets:**
   ```bash
   fly secrets set SPACESCOPE_TOKEN=your_token_here
   ```

4. **Deploy Lambda-like scheduled cache updater:**
   ```bash
   # Build the Lambda-like image
   fly deploy -c fly.lambda.toml
   
   # Create scheduled machine (runs daily at 1:00 UTC)
   fly machine create \
     --app mechafil-cache-updater \
     --config fly.lambda.toml \
     --schedule="0 1 * * *" \
     --restart=no \
     --mount=shared_cache:/data/shared-cache
   ```

**How the volume updater runs on Fly.io:**
- Scheduled machine starts daily at 1:00 UTC
- Executes `python -m cache_updater.main --once` and exits
- Volume remains persistent and accessible
- Zero cost when not running (serverless pattern)

**Benefits of Lambda-like approach:**
- ✅ **Runs only when needed** (daily at 1:00 UTC)
- ✅ **Zero cost when idle** (no running machines)
- ✅ **Volume persists** between runs
- ✅ **True serverless** pattern

### Alternative: Background Service (Always On)

If you prefer a traditional always-on service:

```bash
# Use the regular Dockerfile instead
cat > fly.toml << 'EOF'
app = "mechafil-cache-updater"
primary_region = "ord"

[build]
  dockerfile = "Dockerfile"

[mounts]
  source = "shared_cache"
  destination = "/data/shared-cache"

[env]
  SHARED_CACHE_DIR = "/data/shared-cache"
  RELOAD_TRIGGER = "01:00"
  RELOAD_TEST_MODE = "false"

[processes]
  cache_updater = "python -m cache_updater.main --service"
EOF

fly deploy
```

### Step 2: Deploy Mechafil Server

1. **Create mechafil server app:**
   ```bash
   cd /Users/luca/programmi/cel/mechafil/programs/mechafil-server
   fly apps create mechafil-server
   ```

2. **Create fly.toml for mechafil server:**
   ```bash
   cat > fly.toml << 'EOF'
   app = "mechafil-server"
   primary_region = "ord"
   
   [build]
     dockerfile = "Dockerfile"
   
   [mounts]
     source = "shared_cache"
     destination = "/data/shared-cache"
   
   [env]
     USE_SHARED_CACHE = "true"
     SHARED_CACHE_DIR = "/data/shared-cache"
   
   [http_service]
     internal_port = 8000
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true
   
   [[vm]]
     cpu_kind = "shared"
     cpus = 1
     memory_mb = 1024
   EOF
   ```

3. **Attach the same volume:**
   ```bash
   # Note: Both apps must use the same volume name and region
   fly volumes create shared_cache --size 1 --region ord
   ```

4. **Deploy mechafil server:**
   ```bash
   fly deploy
   ```

**How mechafil-server accesses the volume on Fly.io:**
- Mounts same volume at `/data/shared-cache`
- Reads cached data using same DiskCache keys
- Auto-starts when HTTP requests arrive
- Auto-stops when idle (scale to zero)
- Volume remains accessible even when app scales to zero

### Step 3: Configure Auto-Scaling

**Enable auto-scaling for mechafil-server:**
```bash
cd /Users/luca/programmi/cel/mechafil/programs/mechafil-server
fly scale count 0 --region ord  # Allow scaling to zero
fly autoscale set min=0 max=3
```

### Step 4: Test Deployment

```bash
# Get app URLs
fly apps list

# Test mechafil server
curl https://mechafil-server.fly.dev/health

# View logs
fly logs -a mechafil-cache-updater
fly logs -a mechafil-server
```

### Deployment Architecture Benefits

1. **Cost Optimization**: 
   - Cache updater runs continuously (minimal cost)
   - Mechafil server scales to zero when not in use
   - Single data fetch serves all server instances

2. **Performance**:
   - No cold starts for data fetching
   - Server instances start quickly (read from cache)
   - Persistent volume survives deployments

3. **Reliability**:
   - Cache updater handles data refresh automatically
   - Server instances are stateless and replaceable
   - Volume provides data persistence

### Monitoring

**Check service status:**
```bash
# Cache updater status
fly status -a mechafil-cache-updater

# Server status
fly status -a mechafil-server

# Volume usage
fly volumes list
```

**View logs:**
```bash
fly logs -a mechafil-cache-updater -f
fly logs -a mechafil-server -f
```

## 🔧 Troubleshooting

### Common Issues

**Cache updater fails to start:**
- Check Spacescope credentials: `fly secrets list -a mechafil-cache-updater`
- Verify volume is mounted: `fly ssh console -a mechafil-cache-updater -C "ls -la /data"`

**Server can't find cache:**
- Verify both apps use same volume name
- Check shared cache directory: `fly ssh console -a mechafil-server -C "ls -la /data/shared-cache"`
- Ensure cache updater has populated data

**Performance issues:**
- Check volume region matches app region
- Monitor volume usage: `fly volumes list`
- Scale server instances: `fly scale count 2 -a mechafil-server`

### Development vs Production

**Development (.env):**
```env
RELOAD_TEST_MODE=true    # 2-minute refresh
RELOAD_TRIGGER=01:00     # Daily at 1:00 UTC
LOG_LEVEL=INFO
```

**Production (fly.io):**
```env
RELOAD_TEST_MODE=false   # Daily refresh only
RELOAD_TRIGGER=01:00     # 1:00 UTC
LOG_LEVEL=WARNING
```

## 📊 Cost Estimation

**Fly.io Costs (approximate):**
- Cache updater (always on): ~$2-5/month
- Shared volume (1GB): ~$0.15/month  
- Mechafil server (auto-scale): $0 when idle, ~$0.01/hour when active

Total: **~$2-6/month** for fully managed, auto-scaling setup.

## 🤝 Contributing

1. Make changes to the appropriate service
2. Test locally with `docker-compose up --build`
3. Deploy to staging environment on Fly.io
4. Run integration tests
5. Deploy to production

## 📝 License

[Add your license information here]