# MechaFil Server Deployment Guide

Comprehensive guide for deploying MechaFil Server to Fly.io with serverless architecture and shared cache volume.

## Table of Contents

- [Overview](#overview)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Fly.io Deployment](#flyio-deployment)
- [GitHub Actions Setup](#github-actions-setup)
- [Configuration](#configuration)
- [Monitoring and Operations](#monitoring-and-operations)
- [Troubleshooting](#troubleshooting)
- [Alternative Deployments](#alternative-deployments)

## Overview

MechaFil Server uses a serverless microservices architecture on Fly.io with:

- **Single unified Docker image** containing both API and cache updater services
- **3GB shared persistent volume** for cache storage
- **Serverless auto-stop/auto-start** behavior (scales to zero when idle)
- **GitHub Actions** for automated daily cache updates
- **Admin endpoint** for manual cache refresh triggers

This deployment pattern minimizes costs by only running when needed while maintaining data persistence across machine restarts.

## Architecture Deep Dive

### Deployment Components

```
┌─────────────────────────────────────────────────────────────┐
│                Fly.io Infrastructure                        │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Fly.io Machine (mechafil-api)                     │    │
│  │  Region: fra (Frankfurt)                           │    │
│  │  Resources: 2 CPU, 2GB RAM                         │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  Docker Container (unified image)            │  │    │
│  │  │                                              │  │    │
│  │  │  ┌────────────────┐  ┌──────────────────┐  │  │    │
│  │  │  │  FastAPI       │  │  Cache Updater   │  │  │    │
│  │  │  │  Web Server    │  │  (Imported)      │  │  │    │
│  │  │  │  Port: 8000    │  │                  │  │  │    │
│  │  │  └───────┬────────┘  └────────┬─────────┘  │  │    │
│  │  │          │                    │            │  │    │
│  │  │          │    Endpoints:      │            │  │    │
│  │  │          │    - /health       │            │  │    │
│  │  │          │    - /simulate     │            │  │    │
│  │  │          │    - /historical   │            │  │    │
│  │  │          │    - /admin/...    │            │  │    │
│  │  │          │           │        │            │  │    │
│  │  │          └───────────┼────────┘            │  │    │
│  │  │                      │                     │  │    │
│  │  │          ┌───────────▼─────────┐           │  │    │
│  │  │          │  DiskCache          │           │  │    │
│  │  │          │  (Python library)   │           │  │    │
│  │  │          └───────────┬─────────┘           │  │    │
│  │  └──────────────────────┼──────────────────────┘  │    │
│  └────────────────────────┼─────────────────────────┘    │
│                           │                              │
│              ┌────────────▼────────────┐                 │
│              │  Fly.io Volume          │                 │
│              │  Name: shared_cache     │                 │
│              │  Size: 3GB              │                 │
│              │  Mount: /data/shared-   │                 │
│              │         cache            │                 │
│              │  Persistent: Yes        │                 │
│              └─────────────────────────┘                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
       ▲                                    ▲
       │                                    │
       │ HTTP Requests                     │ POST
       │ (Auto-starts machine)             │ /admin/update-cache
       │                                    │
┌──────┴──────────┐              ┌─────────┴──────────┐
│  End Users      │              │  GitHub Actions    │
│  Clients        │              │  Daily: 1:00 UTC   │
└─────────────────┘              └────────────────────┘
```

### Data Flow

#### 1. Normal Request Flow (Simulation)

```
User Request → Fly.io Edge → Machine Auto-Start (if stopped)
                              ↓
                        FastAPI loads data from cache
                              ↓
                        mechafil-jax simulation
                              ↓
                        JSON response
                              ↓
                        Machine idle timer starts
                              ↓
                        Auto-stop after ~2-3 min
```

#### 2. Cache Update Flow

```
GitHub Actions (1:00 UTC) → POST /admin/update-cache
                              ↓
                        Machine Auto-Start (if stopped)
                              ↓
                        Endpoint imports cache_updater
                              ↓
                        Fetch data from Spacescope
                              ↓
                        Write to /data/shared-cache
                              ↓
                        Reload data in API service
                              ↓
                        Return success response
                              ↓
                        Machine idle timer starts
                              ↓
                        Auto-stop after ~2-3 min
```

### Key Architecture Decisions

1. **Unified Image**: Both services in one Docker image reduces deployment complexity and ensures consistency
2. **Shared Volume**: Single 3GB volume eliminates data synchronization issues between services
3. **Admin Endpoint**: Cache updater callable via HTTP allows external triggers without separate machines
4. **Serverless Mode**: Auto-stop/auto-start minimizes costs while maintaining responsiveness
5. **No Health Checks**: Removed to enable faster auto-stop (health checks keep machine alive)

## Prerequisites

### Required Tools

```bash
# 1. Fly.io CLI
curl -L https://fly.io/install.sh | sh

# Verify installation
flyctl version

# 2. Docker (for local testing)
docker --version

# 3. Git (for repository management)
git --version
```

### Required Accounts

1. **Fly.io Account**: Sign up at https://fly.io
2. **GitHub Account**: For repository and Actions (if using automated updates)
3. **Spacescope API Access**: Token for fetching Filecoin network data

### Environment Setup

Create `.env` file in project root:

```bash
# Required: Spacescope API authentication
SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE

# Cache configuration
USE_SHARED_CACHE=true
SHARED_CACHE_DIR=/data/shared-cache

# API settings
HOST=0.0.0.0
PORT=8000

# Logging
LOG_LEVEL=INFO

# CORS (restrict in production)
CORS_ORIGINS=*
```

## Initial Setup

### 1. Install Fly.io CLI

```bash
# Linux/Mac
curl -L https://fly.io/install.sh | sh

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.fly/bin:$PATH"

# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex
```

### 2. Authenticate with Fly.io

```bash
flyctl auth login
# Opens browser for authentication
```

### 3. Set Spacescope Token

```bash
# Set as Fly.io secret (recommended for production)
flyctl secrets set SPACESCOPE_TOKEN="Bearer YOUR_TOKEN_HERE" --app mechafil-api

# Or add to fly.toml [env] section (less secure)
```

## Fly.io Deployment

### Step 1: Create Persistent Volume

Create a 3GB volume for cache storage:

```bash
flyctl volumes create shared_cache \
  --region fra \
  --size 3 \
  --app mechafil-api
```

**Important Notes:**
- Volume size is 3GB (sufficient for historical data cache)
- Region must match your app's region (fra = Frankfurt)
- Volume name must match `fly.toml` mount configuration
- Volumes cannot be resized after creation (must create new volume and migrate)

**Verify volume creation:**
```bash
flyctl volumes list --app mechafil-api
```

Expected output:
```
ID                      NAME            SIZE    REGION  ZONE    ENCRYPTED       ATTACHED VM     CREATED AT
vol_xxxxxxxxxxxxx       shared_cache    3GB     fra     49ef    true            -               2024-01-15
```

### Step 2: Configure fly.toml

The provided `fly.toml` is pre-configured. Key sections:

```toml
app = "mechafil-api"
primary_region = "fra"

[build]
  dockerfile = "docker/Dockerfile"

[env]
  USE_SHARED_CACHE = "true"
  SHARED_CACHE_DIR = "/data/shared-cache"
  HOST = "0.0.0.0"
  PORT = "8000"

[mounts]
  source = "shared_cache"
  destination = "/data/shared-cache"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "stop"  # Enable serverless
  auto_start_machines = true   # Auto-start on request
  min_machines_running = 0     # Scale to zero

[[vm]]
  memory = "2gb"
  cpu_kind = "shared"
  cpus = 2
```

**Configuration Notes:**
- `auto_stop_machines = "stop"`: Machine stops after idle period (~2-3 minutes)
- `auto_start_machines = true`: Machine starts automatically on incoming HTTP request
- `min_machines_running = 0`: True serverless - no minimum running machines
- No `[[http_service.checks]]`: Health checks removed to allow faster auto-stop

### Step 3: Initial Deployment

```bash
# From mechafil-server directory
flyctl deploy --app mechafil-api

# Watch deployment progress
flyctl logs --app mechafil-api
```

**Deployment Process:**
1. Builds Docker image using `docker/Dockerfile`
2. Pushes image to Fly.io registry
3. Creates machine from image
4. Mounts shared_cache volume at `/data/shared-cache`
5. Starts container with CMD from Dockerfile: `python -m services.api.main`
6. Machine becomes available at https://mechafil-api.fly.dev

### Step 4: Initial Cache Population

After first deployment, populate the cache:

```bash
# Method 1: Call admin endpoint (machine must start up)
curl -X POST https://mechafil-api.fly.dev/admin/update-cache

# Method 2: SSH into machine and run manually
flyctl ssh console --app mechafil-api
python3 -c "from services.cache_updater.main import run_once; import asyncio; asyncio.run(run_once())"
exit
```

**Verify cache population:**
```bash
flyctl ssh console --app mechafil-api
ls -lah /data/shared-cache
# Should see cache files
exit
```

### Step 5: Verify Deployment

```bash
# Check machine status
flyctl status --app mechafil-api

# Test health endpoint
curl https://mechafil-api.fly.dev/health

# Test simulation endpoint
curl -X POST https://mechafil-api.fly.dev/simulate \
  -H 'Content-Type: application/json' \
  -d '{"forecast_length_days": 365}'

# Test historical data endpoint
curl https://mechafil-api.fly.dev/historical-data
```

**Expected behavior after tests:**
- Machine starts automatically on first request (cold start ~15-20 seconds)
- Subsequent requests are fast (machine already running)
- After ~2-3 minutes of inactivity, machine auto-stops
- Next request triggers auto-start again

## GitHub Actions Setup

### Overview

GitHub Actions automatically triggers cache updates daily at 1:00 UTC by calling the `/admin/update-cache` endpoint.

### Configuration File

The workflow is defined in `.github/workflows/update-cache-daily.yml`:

```yaml
name: Daily Cache Update

on:
  schedule:
    # Run daily at 1:00 AM UTC
    - cron: '0 1 * * *'
  workflow_dispatch:  # Allow manual triggering

jobs:
  update-cache:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Cache Update via Admin Endpoint
        run: |
          # Call the admin endpoint to trigger cache update
          # This will wake up the machine if it's stopped, run the update, then let it idle again
          curl -X POST https://mechafil-api.fly.dev/admin/update-cache \
            -H "Content-Type: application/json" \
            -f -v

      - name: Check Response
        run: |
          echo "Cache update completed successfully!"
```

### Setup Instructions

1. **Enable GitHub Actions** (if not already enabled):
   - Go to repository Settings → Actions → General
   - Set "Actions permissions" to "Allow all actions"

2. **Verify workflow file exists**:
   ```bash
   ls -la .github/workflows/update-cache-daily.yml
   ```

3. **Test manual trigger**:
   - Go to GitHub repository → Actions tab
   - Select "Daily Cache Update" workflow
   - Click "Run workflow" → "Run workflow"
   - Monitor execution

4. **Verify scheduled runs**:
   - Check Actions tab at 1:00 UTC daily
   - Review workflow run history

### Workflow Behavior

**What happens during scheduled run:**
1. GitHub Actions runner executes at 1:00 UTC
2. Sends POST request to `https://mechafil-api.fly.dev/admin/update-cache`
3. Fly.io detects request to stopped machine → auto-starts machine
4. Machine starts, FastAPI loads, endpoint handler executes
5. Endpoint imports `services.cache_updater.main` module
6. Calls `run_once()` function to fetch fresh data from Spacescope
7. Writes updated cache to `/data/shared-cache` volume
8. Reloads data in API service memory
9. Returns success response to GitHub Actions
10. Machine idle timer starts (~2-3 min) → auto-stops

**Timeline:**
- Cold start: ~15-20 seconds
- Cache update: ~30-60 seconds (depends on Spacescope API)
- Data reload: ~5-10 seconds
- Total: ~1-2 minutes
- Auto-stop: ~2-3 minutes after completion

### Advanced Configuration

**Change schedule:**
```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # 2:00 AM UTC
    - cron: '0 14 * * *'  # 2:00 PM UTC (twice daily)
```

**Add retry logic:**
```yaml
- name: Trigger Cache Update with Retry
  run: |
    for i in 1 2 3; do
      if curl -X POST https://mechafil-api.fly.dev/admin/update-cache -f -v; then
        echo "Success on attempt $i"
        break
      else
        echo "Attempt $i failed, retrying..."
        sleep 30
      fi
    done
```

**Add Slack/Discord notifications:**
```yaml
- name: Notify on Failure
  if: failure()
  run: |
    curl -X POST ${{ secrets.SLACK_WEBHOOK_URL }} \
      -H 'Content-Type: application/json' \
      -d '{"text": "Cache update failed! Check https://mechafil-api.fly.dev"}'
```

## Configuration

### Environment Variables Reference

#### Shared Configuration (both services)

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SHARED_CACHE` | `false` | Enable shared cache volume |
| `SHARED_CACHE_DIR` | `/data/shared-cache` | Cache directory path |
| `SPACESCOPE_TOKEN` | - | Spacescope API bearer token |
| `SPACESCOPE_AUTH_FILE` | `.spacescope_auth` | Path to auth JSON file (alternative to token) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |

#### API Service Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | API server host |
| `PORT` | `8000` | API server port |
| `RELOAD` | `false` | Enable hot reload (development only) |

#### Cache Updater Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RELOAD_TRIGGER` | `01:00` | Daily refresh time (HH:MM UTC) |
| `RELOAD_TEST_MODE` | `false` | Enable 2-minute test cycles |

### Fly.io Secrets Management

**Set secrets (recommended for sensitive values):**
```bash
flyctl secrets set SPACESCOPE_TOKEN="Bearer YOUR_TOKEN" --app mechafil-api
```

**List secrets:**
```bash
flyctl secrets list --app mechafil-api
```

**Unset secrets:**
```bash
flyctl secrets unset SPACESCOPE_TOKEN --app mechafil-api
```

**Important:** Secrets are encrypted and not visible in `fly.toml` or logs.

### Volume Management

**List volumes:**
```bash
flyctl volumes list --app mechafil-api
```

**Create snapshot (backup):**
```bash
flyctl volumes snapshots create vol_xxxxxxxxxxxxx --app mechafil-api
```

**List snapshots:**
```bash
flyctl volumes snapshots list vol_xxxxxxxxxxxxx --app mechafil-api
```

**Restore from snapshot:**
```bash
flyctl volumes create shared_cache_restored \
  --snapshot-id vs_xxxxxxxxx \
  --region fra \
  --app mechafil-api
```

**Delete volume (WARNING: irreversible):**
```bash
flyctl volumes delete vol_xxxxxxxxxxxxx --app mechafil-api
```

## Monitoring and Operations

### Viewing Logs

**Real-time logs:**
```bash
flyctl logs --app mechafil-api
```

**Filter logs:**
```bash
# Show only errors
flyctl logs --app mechafil-api | grep ERROR

# Show cache update logs
flyctl logs --app mechafil-api | grep "Cache update"
```

**Historical logs:**
```bash
# Last 1 hour
flyctl logs --app mechafil-api --since 1h

# Specific time range
flyctl logs --app mechafil-api --since "2024-01-15T10:00:00Z"
```

### Machine Management

**Check machine status:**
```bash
flyctl status --app mechafil-api
flyctl machine list --app mechafil-api
```

**Restart machine:**
```bash
flyctl machine restart <machine-id> --app mechafil-api
```

**Stop machine manually:**
```bash
flyctl machine stop <machine-id> --app mechafil-api
```

**Start machine manually:**
```bash
flyctl machine start <machine-id> --app mechafil-api
```

### SSH Access

**Open SSH console:**
```bash
flyctl ssh console --app mechafil-api
```

**Check cache contents:**
```bash
flyctl ssh console --app mechafil-api --command "ls -lah /data/shared-cache"
```

**Check disk usage:**
```bash
flyctl ssh console --app mechafil-api --command "df -h /data/shared-cache"
```

**Inspect cache keys:**
```bash
flyctl ssh console --app mechafil-api --command \
  "python3 -c 'from diskcache import Cache; c = Cache(\"/data/shared-cache\"); print(list(c))'"
```

### Health Monitoring

**Check API health:**
```bash
curl https://mechafil-api.fly.dev/health
```

**Monitor auto-stop behavior:**
```bash
# Terminal 1: Watch machine status
watch -n 5 'flyctl machine list --app mechafil-api'

# Terminal 2: Make request
curl https://mechafil-api.fly.dev/health

# Observe: machine starts, processes request, then stops after ~2-3 min
```

### Performance Monitoring

**Check response times:**
```bash
time curl https://mechafil-api.fly.dev/health
# Cold start: ~15-20 seconds
# Warm: <1 second
```

**Measure simulation performance:**
```bash
time curl -X POST https://mechafil-api.fly.dev/simulate \
  -H 'Content-Type: application/json' \
  -d '{"forecast_length_days": 3650}'
```

## Troubleshooting

### Issue: Machine Not Stopping

**Symptoms:**
- Machine stays "started" even when idle
- Unexpected costs from always-running machine

**Diagnosis:**
```bash
# Check for health checks (should be empty)
flyctl status --app mechafil-api

# Look for CHECKS column - should be empty
# If CHECKS shows "1 total", health checks are preventing auto-stop
```

**Solution:**
```bash
# 1. Verify fly.toml has no health checks
grep -A 10 "http_service" fly.toml
# Should NOT have [[http_service.checks]] section

# 2. If health checks exist, remove them from fly.toml
# 3. Destroy and recreate machine
flyctl machine list --app mechafil-api
flyctl machine destroy <machine-id> --app mechafil-api --force

# 4. Redeploy
flyctl deploy --app mechafil-api

# 5. Verify no health checks
flyctl status --app mechafil-api
# CHECKS column should be empty
```

### Issue: Cache Not Loading

**Symptoms:**
- API returns 503 errors
- Logs show "Failed to load historical data"

**Diagnosis:**
```bash
# Check if cache directory exists and has data
flyctl ssh console --app mechafil-api
ls -lah /data/shared-cache
# Should see cache files with recent timestamps

# Check cache contents
python3 -c "from diskcache import Cache; c = Cache('/data/shared-cache'); print(list(c))"
# Should show cache keys
```

**Solution:**
```bash
# Manually trigger cache update
curl -X POST https://mechafil-api.fly.dev/admin/update-cache -v

# Or SSH and run manually
flyctl ssh console --app mechafil-api
python3 -m services.cache_updater.main --once
exit

# Restart machine to reload data
flyctl machine restart <machine-id> --app mechafil-api
```

### Issue: GitHub Actions Failing

**Symptoms:**
- Workflow shows red X
- Cache not updating daily

**Diagnosis:**
```bash
# Check workflow run history
# Visit: https://github.com/YOUR_ORG/YOUR_REPO/actions

# Test endpoint manually
curl -X POST https://mechafil-api.fly.dev/admin/update-cache -v

# Check machine logs during scheduled run time
flyctl logs --app mechafil-api --since 1h | grep "update-cache"
```

**Common Causes:**

1. **Machine startup timeout:**
   - Cold start can take 15-20 seconds
   - GitHub Actions might timeout
   - **Solution:** Increase timeout in workflow:
     ```yaml
     - name: Trigger Cache Update
       timeout-minutes: 5  # Add timeout
       run: curl -X POST ... -v
     ```

2. **Spacescope API issues:**
   - Token expired or invalid
   - API rate limiting
   - **Solution:** Check logs for Spacescope errors:
     ```bash
     flyctl logs --app mechafil-api | grep -i spacescope
     ```

3. **Network issues:**
   - GitHub Actions runner can't reach Fly.io
   - **Solution:** Add retry logic (see [GitHub Actions Setup](#github-actions-setup))

### Issue: Volume Full

**Symptoms:**
- Errors writing to cache
- Disk space errors in logs

**Diagnosis:**
```bash
# Check disk usage
flyctl ssh console --app mechafil-api --command "df -h /data/shared-cache"

# Check cache size
flyctl ssh console --app mechafil-api --command "du -sh /data/shared-cache"
```

**Solution:**

**Option 1: Clean old cache entries**
```bash
flyctl ssh console --app mechafil-api
python3 -c "
from diskcache import Cache
c = Cache('/data/shared-cache')
c.cull()  # Remove expired entries
c.clear()  # Or clear all (requires repopulation)
"
exit

# Repopulate cache
curl -X POST https://mechafil-api.fly.dev/admin/update-cache
```

**Option 2: Create larger volume**
```bash
# Create new 10GB volume
flyctl volumes create shared_cache_10gb --region fra --size 10 --app mechafil-api

# Update fly.toml
[mounts]
  source = "shared_cache_10gb"
  destination = "/data/shared-cache"

# Redeploy (machine will use new volume, losing old data)
flyctl deploy --app mechafil-api

# Repopulate cache
curl -X POST https://mechafil-api.fly.dev/admin/update-cache
```

### Issue: Slow Cold Starts

**Symptoms:**
- First request after idle takes 15-20 seconds
- Users experience timeouts

**Causes:**
- Normal behavior for serverless architecture
- Machine must boot and load data from volume

**Solutions:**

1. **Keep machine running (disable auto-stop):**
   ```toml
   # In fly.toml
   [http_service]
     auto_stop_machines = false
     min_machines_running = 1
   ```

2. **Add warm-up endpoint to GitHub Actions:**
   ```yaml
   - name: Warm Up Machine
     run: |
       curl https://mechafil-api.fly.dev/health
       # Now machine is warm for actual traffic
   ```

3. **Use Fly.io proxy to keep machine warm:**
   ```bash
   # Keep machine warm during business hours
   # Add to crontab or GitHub Actions
   0 8-18 * * * curl https://mechafil-api.fly.dev/health
   ```

### Issue: Deployment Fails

**Common errors and solutions:**

**Error: "No space left on device"**
```bash
# Solution: Clear Docker build cache locally
docker system prune -a --volumes
flyctl deploy --app mechafil-api
```

**Error: "Volume not found"**
```bash
# Solution: Verify volume exists and name matches fly.toml
flyctl volumes list --app mechafil-api
# Update fly.toml [mounts] source to match volume name
```

**Error: "Region mismatch"**
```bash
# Solution: Ensure volume and app are in same region
flyctl volumes list --app mechafil-api  # Check volume region
flyctl status --app mechafil-api  # Check app region

# If mismatch, create volume in correct region
flyctl volumes create shared_cache --region <app-region> --size 3 --app mechafil-api
```

## Alternative Deployments

### Docker Compose (Development/Single VM)

For local development or single-server deployments:

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - shared_cache:/data/shared-cache
    environment:
      - USE_SHARED_CACHE=true
      - SHARED_CACHE_DIR=/data/shared-cache
      - SPACESCOPE_TOKEN=${SPACESCOPE_TOKEN}
      - HOST=0.0.0.0
      - PORT=8000
    depends_on:
      - cache-updater

  cache-updater:
    build:
      context: .
      dockerfile: docker/Dockerfile
    volumes:
      - shared_cache:/data/shared-cache
    environment:
      - USE_SHARED_CACHE=true
      - SHARED_CACHE_DIR=/data/shared-cache
      - SPACESCOPE_TOKEN=${SPACESCOPE_TOKEN}
    command: python -m services.cache_updater.main
    restart: unless-stopped

volumes:
  shared_cache:
    driver: local
```

**Usage:**
```bash
# Start both services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Update cache manually
docker-compose exec api curl -X POST http://localhost:8000/admin/update-cache
```

### Kubernetes (Production Scale)

For large-scale production deployments:

```yaml
# k8s-deployment.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mechafil-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 3Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mechafil-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mechafil-api
  template:
    metadata:
      labels:
        app: mechafil-api
    spec:
      containers:
      - name: api
        image: your-registry/mechafil-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: USE_SHARED_CACHE
          value: "true"
        - name: SHARED_CACHE_DIR
          value: "/data/shared-cache"
        volumeMounts:
        - name: cache-volume
          mountPath: /data/shared-cache
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: mechafil-cache
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mechafil-cache-updater
spec:
  schedule: "0 1 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: updater
            image: your-registry/mechafil-server:latest
            command:
            - python
            - -m
            - services.cache_updater.main
            - --once
            env:
            - name: USE_SHARED_CACHE
              value: "true"
            - name: SHARED_CACHE_DIR
              value: "/data/shared-cache"
            volumeMounts:
            - name: cache-volume
              mountPath: /data/shared-cache
          restartPolicy: OnFailure
          volumes:
          - name: cache-volume
            persistentVolumeClaim:
              claimName: mechafil-cache
```

### AWS Lambda + EFS

For serverless AWS deployment:

1. **Create EFS volume**
2. **Package application as container image**
3. **Deploy API as Lambda function**
4. **Mount EFS to Lambda**
5. **Use EventBridge to trigger cache updater Lambda daily**

See AWS documentation for detailed Lambda + EFS setup.

## Best Practices

### Security

1. **Secrets Management:**
   - Never commit tokens to git
   - Use Fly.io secrets for sensitive values
   - Rotate tokens regularly

2. **CORS Configuration:**
   ```bash
   # Restrict CORS in production
   flyctl secrets set CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com" --app mechafil-api
   ```

3. **Admin Endpoint Protection:**
   - Consider adding authentication to `/admin/update-cache`
   - Use Fly.io private networking for internal-only access
   - Rate limit admin endpoints

### Cost Optimization

1. **Serverless Configuration:**
   - Keep `auto_stop_machines = "stop"` and `min_machines_running = 0`
   - Remove health checks to allow faster auto-stop
   - Machine only runs when needed

2. **Resource Sizing:**
   - Start with 2GB RAM, 2 CPUs
   - Monitor usage: `flyctl machine list --app mechafil-api`
   - Scale down if underutilized

3. **Volume Management:**
   - 3GB volume sufficient for current data
   - Monitor usage: `df -h /data/shared-cache`
   - Clean old cache entries if needed

### Reliability

1. **Monitoring:**
   - Set up Fly.io metrics monitoring
   - Add healthcheck endpoint monitoring (external service like UptimeRobot)
   - Monitor GitHub Actions workflow runs

2. **Backup Strategy:**
   - Create volume snapshots before major changes
   - Test cache repopulation procedure
   - Document recovery procedures

3. **Testing:**
   - Test deployments in staging environment
   - Verify cache updates after deployment
   - Test cold start behavior

## Support and Resources

### Documentation

- **Fly.io Docs**: https://fly.io/docs/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **JAX Docs**: https://jax.readthedocs.io/

### Community

- **Fly.io Community**: https://community.fly.io/
- **GitHub Issues**: [Your repository issues URL]

### Monitoring Tools

- **Fly.io Dashboard**: https://fly.io/dashboard
- **GitHub Actions**: https://github.com/[org]/[repo]/actions
- **Fly.io Metrics**: `flyctl metrics --app mechafil-api`

## Appendix

### Fly.io CLI Command Reference

```bash
# Deployment
flyctl deploy --app mechafil-api
flyctl deploy --app mechafil-api --no-cache  # Force rebuild

# Machine management
flyctl machine list --app mechafil-api
flyctl machine restart <id> --app mechafil-api
flyctl machine stop <id> --app mechafil-api
flyctl machine start <id> --app mechafil-api
flyctl machine destroy <id> --app mechafil-api --force

# Volume management
flyctl volumes list --app mechafil-api
flyctl volumes snapshots create <vol-id> --app mechafil-api
flyctl volumes snapshots list <vol-id> --app mechafil-api

# Secrets management
flyctl secrets list --app mechafil-api
flyctl secrets set KEY=value --app mechafil-api
flyctl secrets unset KEY --app mechafil-api

# Logs and monitoring
flyctl logs --app mechafil-api
flyctl logs --app mechafil-api -f  # Follow
flyctl status --app mechafil-api
flyctl metrics --app mechafil-api

# SSH access
flyctl ssh console --app mechafil-api
flyctl ssh console --app mechafil-api --command "ls -la"

# Configuration
flyctl config show --app mechafil-api
flyctl config save --app mechafil-api
```

### Environment Variable Complete Reference

```bash
# Shared configuration
USE_SHARED_CACHE=true
SHARED_CACHE_DIR=/data/shared-cache
SPACESCOPE_TOKEN=Bearer YOUR_TOKEN
SPACESCOPE_AUTH_FILE=.spacescope_auth
LOG_LEVEL=INFO
CORS_ORIGINS=*

# API service
HOST=0.0.0.0
PORT=8000
RELOAD=false

# Cache updater service
RELOAD_TRIGGER=01:00
RELOAD_TEST_MODE=false

# Simulation defaults (optional overrides)
WINDOW_DAYS=3650
SECTOR_DURATION_DAYS=360
LOCK_TARGET=0.3
```

### Useful Commands

```bash
# Check if machine is running
flyctl machine list --app mechafil-api | grep -q "started" && echo "Running" || echo "Stopped"

# Get machine ID
MACHINE_ID=$(flyctl machine list --app mechafil-api -j | jq -r '.[0].id')

# Tail logs and filter
flyctl logs --app mechafil-api -f | grep -E "ERROR|Cache update"

# Test auto-stop behavior
curl https://mechafil-api.fly.dev/health
sleep 180  # Wait 3 minutes
flyctl machine list --app mechafil-api  # Should show "stopped"

# Force cache update and watch logs
curl -X POST https://mechafil-api.fly.dev/admin/update-cache & \
flyctl logs --app mechafil-api -f
```
