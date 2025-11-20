# Fly.io Deployment Log

This document provides a step-by-step deployment guide with detailed explanations of what each command does and why.

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Step 1: Install Fly.io CLI](#step-1-install-flyio-cli)
3. [Step 2: Authenticate](#step-2-authenticate)
4. [Step 3: Create Shared Volume](#step-3-create-shared-volume)
5. [Step 4: Deploy Cache Updater](#step-4-deploy-cache-updater)
6. [Step 5: Deploy API Service](#step-5-deploy-api-service)
7. [Step 6: Verify Deployment](#step-6-verify-deployment)
8. [Step 7: Test Endpoints](#step-7-test-endpoints)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

Before starting deployment, verify you have:

- [ ] Spacescope API token (from `.env` or environment)
- [ ] Git repository is clean and pushed (optional but recommended)
- [ ] Docker is installed and working (Fly will use it to build images)
- [ ] You understand the architecture:
  ```
  ┌─────────────────────┐
  │  Internet Request   │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │   Fly.io Proxy      │ ◄── Automatic HTTPS, routing
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐         ┌─────────────────────┐
  │  API Service        │  Read   │  Shared Volume      │
  │  (mechafil-api)     │◄────────┤  (shared_cache)     │
  │  - Auto-start       │         │  - 10GB persistent  │
  │  - Auto-stop        │  Write  │  - DiskCache files  │
  └─────────────────────┘ ◄────── └─────────────────────┘
                                            ▲
                                            │ Write
                                            │
                                   ┌────────┴────────┐
                                   │  Cache Updater  │
                                   │  (mechafil-     │
                                   │   cache-updater)│
                                   │  - Always on    │
                                   │  - Updates daily│
                                   └─────────────────┘
  ```

**Check your Spacescope token:**
```bash
# Method 1: From .env file
cat .env | grep SPACESCOPE_TOKEN

# Method 2: From environment
echo $SPACESCOPE_TOKEN

# You should see something like: SPACESCOPE_TOKEN=Bearer abc123...
```

**Expected token format:** `Bearer YOUR_TOKEN_HERE`

---

## Step 1: Install Fly.io CLI

**What:** Install `flyctl`, the command-line tool for managing Fly.io applications.

**Why:** This is the only tool you need to deploy, manage, and monitor your applications on Fly.io.

**How:**

```bash
# On Linux/macOS
curl -L https://fly.io/install.sh | sh

# On Windows (PowerShell)
pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

**What happens:**
- Downloads the latest `flyctl` binary
- Installs it to `~/.fly/bin/flyctl`
- Adds it to your PATH

**Verify installation:**
```bash
flyctl version
# Expected output: flyctl v0.x.xxx linux/amd64 Commit: ... Build Date: ...
```

**Add to PATH if needed:**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export PATH="$HOME/.fly/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 2: Authenticate

**What:** Login to your Fly.io account (creates one if you don't have it).

**Why:** Fly needs to know who you are to create resources under your account.

**How:**
```bash
flyctl auth login
```

**What happens:**
1. Opens your browser to https://fly.io/app/sign-in
2. You sign in or create an account (GitHub login recommended)
3. Browser redirects back and saves auth token locally
4. Token stored in `~/.fly/config.yml`

**Expected output:**
```
Opening https://fly.io/app/auth/cli/...
Waiting for session...
successfully logged in as your-email@example.com
```

**Verify authentication:**
```bash
flyctl auth whoami
# Expected output: your-email@example.com
```

**Account limits (free tier):**
- Up to 3 shared-cpu-1x VMs with 256MB RAM each
- 3GB persistent volume storage
- 160GB outbound data transfer
- For this project, you'll need to add a credit card (no charge for small usage)

---

## Step 3: Create Shared Volume

**What:** Create a persistent 10GB volume that both services will mount.

**Why:**
- This volume stores the DiskCache files
- Both API and cache updater need access to the same data
- Volume persists even when machines stop (serverless mode)

**Important:** Choose your region carefully - both services MUST be in the same region to share the volume.

**Available regions:**
```bash
flyctl platform regions
```

**Recommended regions:**
- `fra` - Frankfurt, Germany (Europe)
- `iad` - Ashburn, Virginia (US East)
- `lax` - Los Angeles, California (US West)
- `syd` - Sydney, Australia (Asia-Pacific)

**Create volume:**
```bash
flyctl volumes create shared_cache \
  --region fra \
  --size 10
```

**Command breakdown:**
- `volumes create` - Creates a new persistent volume
- `shared_cache` - Name of the volume (must match `fly.toml` files)
- `--region fra` - Physical location (Frankfurt)
- `--size 10` - Size in GB (10GB = plenty for cache data)

**What happens:**
1. Fly creates a persistent disk in the Frankfurt datacenter
2. Volume gets a unique ID (vol_xxxxx)
3. Volume is formatted and ready to mount

**Expected output:**
```
        ID: vol_xxxxxxxxxxxxx
      Name: shared_cache
       App:
    Region: fra
      Zone: xxxx
   Size GB: 10
 Encrypted: true
Created at: 01 Jan 24 12:00 UTC
```

**Verify volume:**
```bash
flyctl volumes list
# Should show your shared_cache volume
```

**Important notes:**
- Volumes are single-zone (if zone fails, volume is unavailable)
- For production, consider creating volume replicas in multiple zones
- Volume can only be mounted by ONE machine at a time per service
- Volumes persist data between deployments

---

## Step 4: Deploy Cache Updater

**What:** Deploy the background service that fetches data from Spacescope daily.

**Why:** This service populates the shared volume with cache data. The API service needs this data to exist before it can start.

**Order matters:** Deploy cache updater FIRST, let it populate cache, then deploy API.

### Step 4.1: Create the App

**What:** Register the app name with Fly.io.

```bash
flyctl apps create mechafil-cache-updater
```

**What happens:**
- Reserves the app name `mechafil-cache-updater.fly.dev`
- Creates app configuration in Fly's system
- App is created but not yet deployed

**Expected output:**
```
New app created: mechafil-cache-updater
```

**Verify app creation:**
```bash
flyctl apps list
# Should show: mechafil-cache-updater
```

### Step 4.2: Set Secrets

**What:** Securely store the Spacescope API token.

**Why:**
- Secrets are encrypted at rest
- Not exposed in config files or logs
- Injected as environment variables at runtime

```bash
# Replace with your actual token
flyctl secrets set SPACESCOPE_TOKEN="Bearer YOUR_TOKEN_HERE" \
  --app mechafil-cache-updater
```

**Command breakdown:**
- `secrets set` - Store encrypted secret
- `SPACESCOPE_TOKEN="..."` - Key-value pair
- `--app mechafil-cache-updater` - Which app to set secret for

**What happens:**
1. Token is encrypted
2. Stored in Fly's secret management system
3. Will be available as `$SPACESCOPE_TOKEN` environment variable in container

**Expected output:**
```
Secrets are staged for the first deployment
```

**Verify secrets (won't show values):**
```bash
flyctl secrets list --app mechafil-cache-updater
# Should show: SPACESCOPE_TOKEN (redacted)
```

**Security notes:**
- Never commit secrets to git
- Never put secrets in `fly.toml` (use `secrets` instead)
- Secrets are encrypted both at rest and in transit

### Step 4.3: Deploy the Service

**What:** Build Docker image and deploy to Fly.io.

```bash
flyctl deploy \
  --config fly-cache-updater.toml \
  --app mechafil-cache-updater
```

**Command breakdown:**
- `deploy` - Build and deploy the application
- `--config fly-cache-updater.toml` - Use this configuration file
- `--app mechafil-cache-updater` - Deploy to this app

**What happens (this takes 3-5 minutes):**

1. **Build Phase:**
   ```
   ==> Building image
   ```
   - Fly reads `docker/cache-updater.Dockerfile`
   - Builds Docker image locally (or on Fly's remote builders)
   - Installs Poetry, dependencies, copies code
   - Size will be ~1-2GB (includes JAX, numpy, etc.)

2. **Push Phase:**
   ```
   ==> Pushing image to fly
   ```
   - Image is pushed to Fly's container registry
   - Compressed and transferred
   - Stored for future deployments

3. **Release Phase:**
   ```
   ==> Creating release
   ```
   - Fly creates a new release version
   - Allocates machine resources
   - Mounts the shared_cache volume to `/data/shared-cache`
   - Injects secrets as environment variables

4. **Deploy Phase:**
   ```
   ==> Monitoring deployment
   ```
   - Starts container
   - Runs health checks
   - Monitors logs for errors

**Expected output:**
```
==> Building image
==> Building image with Docker
...
==> Pushing image to fly
...
==> Creating release
--> v1
...
==> Monitoring deployment
 1 desired, 1 placed, 1 healthy, 0 unhealthy [health checks: 1 total]
--> v1 deployed successfully
```

**Verify deployment:**
```bash
flyctl status --app mechafil-cache-updater
```

**Expected status output:**
```
App
  Name     = mechafil-cache-updater
  Owner    = your-org
  Hostname = mechafil-cache-updater.fly.dev
  Platform = machines

Machines
ID              STATE   REGION  HEALTH  CHECKS  LAST UPDATED
xxxxxxxxxxxxx   started fra             -       2024-01-01T12:00:00Z
```

### Step 4.4: Monitor Initial Cache Population

**What:** Watch the logs to see cache being populated.

**Why:** The first run fetches data from Spacescope and writes to the shared volume. This must complete before deploying the API.

```bash
flyctl logs --app mechafil-cache-updater
```

**What to look for:**
```
[info] Starting cache updater
[info] Shared cache directory: /data/shared-cache
[info] Fetching historical data from Spacescope...
[info] Processing RBP data...
[info] Processing renewal rate data...
[info] Processing FIL+ rate data...
[info] Computing smoothed values...
[info] Writing to cache: offline_data_2022-10-102025-11-182035-11-16
[info] Cache updated successfully
[info] Next update scheduled for: 2025-11-21 01:00:00 UTC
```

**How long it takes:**
- First fetch: ~30-60 seconds (downloading from Spacescope)
- Subsequent updates: ~30 seconds (less data to process)

**If you see errors:**
- `SPACESCOPE_TOKEN not set` - Secret wasn't set correctly
- `Permission denied` - Volume mount issue
- `Connection timeout` - Spacescope API issue (retry)

**Press Ctrl+C to exit logs** (service keeps running)

### Step 4.5: Verify Cache Files

**What:** SSH into the machine and check cache was created.

```bash
flyctl ssh console --app mechafil-cache-updater
```

**Inside the container:**
```bash
# Check cache directory exists
ls -lh /data/shared-cache/

# You should see files like:
# cache.db          - DiskCache database
# shards/           - Data shards directory
# Additional .db files

# Check cache size
du -sh /data/shared-cache/
# Expected: 100MB - 1GB depending on data

# Exit SSH
exit
```

**What this proves:**
- Volume is mounted correctly
- Cache updater can write to volume
- Data is persisted on disk

---

## Step 5: Deploy API Service

**What:** Deploy the FastAPI service that handles HTTP requests.

**Why:** This is your public-facing API that runs simulations and returns results.

**Prerequisites:** Cache updater must be deployed and have populated cache at least once.

### Step 5.1: Create the App

```bash
flyctl apps create mechafil-api
```

**Expected output:**
```
New app created: mechafil-api
```

**Note:** This app will get the URL `https://mechafil-api.fly.dev`

### Step 5.2: Deploy the Service

**What:** Build and deploy the API container.

```bash
flyctl deploy \
  --config fly-api.toml \
  --app mechafil-api
```

**What happens (3-5 minutes):**

1. **Build Phase:**
   - Builds from `docker/api.Dockerfile`
   - Installs Poetry, FastAPI, JAX, mechafil-jax
   - Copies API code

2. **Push Phase:**
   - Pushes image to registry

3. **Deploy Phase:**
   - Creates machine with 2GB RAM, 2 CPUs
   - Mounts shared_cache volume (READ-ONLY access to cache)
   - Starts FastAPI application
   - FastAPI lifespan loads cache into memory
   - Configures HTTP proxy with auto-stop/start

**Expected output:**
```
==> Building image
...
==> Pushing image to fly
...
==> Creating release
--> v1
...
==> Monitoring deployment
 1 desired, 1 placed, 1 healthy, 0 unhealthy [health checks: 1 total, 1 passing]
--> v1 deployed successfully

Visit your newly deployed app at https://mechafil-api.fly.dev
```

**What the auto-stop/start configuration does:**
- **When idle (no requests):** Machine stops after ~5 minutes, costing $0
- **When request arrives:** Fly auto-starts machine in ~1-2 seconds
- **Cold start:** Includes loading cache from disk (~1-2 seconds total)
- **While running:** Handles requests normally, stays running

### Step 5.3: Monitor API Startup

```bash
flyctl logs --app mechafil-api
```

**Look for:**
```
[info] Starting uvicorn
[info] Loading cache from /data/shared-cache
[info] Found cache key: offline_data_2022-10-102025-11-182035-11-16
[info] Loaded historical data: 1132 days
[info] Smoothed RBP: 3.38 PIB/day
[info] Smoothed RR: 0.83
[info] Smoothed FPR: 0.86
[info] Uvicorn running on http://0.0.0.0:8000
[info] Application startup complete
```

**If you see:**
- `No cache data found` - Cache updater hasn't populated cache yet
- `Permission denied` - Volume mount issue
- `Failed to load cache` - Cache corruption (redeploy cache updater)

---

## Step 6: Verify Deployment

**What:** Check both services are running and healthy.

### Check Apps Status

```bash
# List all your apps
flyctl apps list

# Should show:
# NAME                    OWNER   STATUS  LATEST DEPLOY
# mechafil-api            you     deployed  1m ago
# mechafil-cache-updater  you     deployed  5m ago
```

### Check Machines Status

```bash
# API service
flyctl status --app mechafil-api

# Cache updater service
flyctl status --app mechafil-cache-updater
```

**Healthy output looks like:**
```
Machines
ID              STATE   REGION  HEALTH  CHECKS  LAST UPDATED
xxxxxxxxxxxxx   started fra     ✓       1 total 2024-01-01T12:00:00Z
```

**State meanings:**
- `started` - Running normally
- `stopped` - Stopped (API does this when idle - this is OK!)
- `starting` - Booting up
- `destroyed` - Machine was removed (not good)

### Check Volume Status

```bash
flyctl volumes list
```

**Should show:**
```
ID              NAME          SIZE  REGION  ZONE  ATTACHED VM     CREATED AT
vol_xxxxx       shared_cache  10GB  fra     xxxx  yyyyyyyyy       1 hour ago
```

**Important fields:**
- `ATTACHED VM` - Should show machine ID (means volume is mounted)
- `SIZE` - Should be 10GB
- `REGION` - Should match your apps' region

---

## Step 7: Test Endpoints

**What:** Verify the API is working correctly.

### Get API URL

```bash
flyctl info --app mechafil-api
```

**Look for:**
```
Hostname = mechafil-api.fly.dev
```

**Your API is now live at:** `https://mechafil-api.fly.dev`

### Test Health Endpoint

```bash
curl https://mechafil-api.fly.dev/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "jax_backend": "cpu"
}
```

**If first request after idle:**
- May take 2-3 seconds (cold start)
- Subsequent requests will be fast

### Test Historical Data Endpoint

```bash
curl https://mechafil-api.fly.dev/historical-data
```

**Expected response (truncated):**
```json
{
  "date": ["2022-10-10", "2022-10-17", ...],
  "raw_byte_power": [18.5, 18.7, ...],
  "renewal_rate": [0.75, 0.76, ...],
  "filplus_rate": [0.82, 0.83, ...]
}
```

**What this proves:**
- API successfully loaded cache from volume
- Historical data is available
- Downsampling to Mondays works

### Test Simulation Endpoint (Default Parameters)

```bash
curl -X POST https://mechafil-api.fly.dev/simulate \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Expected response:**
```json
{
  "date": ["2025-11-20", "2025-11-27", ...],
  "available_supply": [450123456.78, 451234567.89, ...],
  "network_RBP_EIB": [19.2, 19.5, ...],
  "circ_supply": [580123456.78, 582234567.89, ...],
  ...
}
```

**What this tests:**
- JAX simulation engine works
- Default parameters applied correctly
- Weekly averaging works

### Test Simulation with Custom Parameters

```bash
curl -X POST https://mechafil-api.fly.dev/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "rbp": 4.0,
    "rr": 0.85,
    "fpr": 0.90,
    "forecast_length_days": 365,
    "output": ["available_supply", "network_RBP_EIB"]
  }'
```

**Expected response:**
```json
{
  "date": ["2025-11-20", "2025-11-27", ...],
  "available_supply": [450123456.78, ...],
  "network_RBP_EIB": [19.2, ...]
}
```

**What this tests:**
- Custom parameters work
- Output filtering works
- Simulation with different inputs

### Test in Browser

Open in your browser:
- **API Docs:** https://mechafil-api.fly.dev/docs
- **Health:** https://mechafil-api.fly.dev/health

**You should see:**
- Interactive Swagger UI
- Can test endpoints directly in browser
- Request/response examples

### Performance Testing

```bash
# Test response time
time curl https://mechafil-api.fly.dev/health

# Expected:
# real    0m0.200s  (first request - cold start)
# real    0m0.050s  (subsequent requests - warm)
```

---

## Monitoring and Maintenance

### View Logs (Real-time)

```bash
# API logs
flyctl logs --app mechafil-api

# Cache updater logs
flyctl logs --app mechafil-cache-updater

# Follow logs (like tail -f)
flyctl logs --app mechafil-api --follow
```

### Check Resource Usage

```bash
# API metrics
flyctl metrics --app mechafil-api

# Shows: CPU, Memory, Network usage
```

### SSH into Running Machine

```bash
# API machine
flyctl ssh console --app mechafil-api

# Cache updater machine
flyctl ssh console --app mechafil-cache-updater
```

**Useful commands inside container:**
```bash
# Check cache
ls -lh /data/shared-cache/
du -sh /data/shared-cache/

# Check memory
free -h

# Check running processes
ps aux

# Check environment variables
env | grep CACHE

# Exit
exit
```

### Manual Cache Update

**Force cache update outside schedule:**

```bash
# Method 1: Restart cache updater (triggers immediate update)
flyctl machine restart --app mechafil-cache-updater

# Method 2: SSH and run manually
flyctl ssh console --app mechafil-cache-updater
python -m services.cache_updater.main --once
exit
```

### Scale API (if needed)

```bash
# Scale to always-on (no auto-stop)
flyctl scale count 1 --app mechafil-api

# Scale back to auto-stop (serverless)
flyctl scale count 0 --app mechafil-api

# Scale up resources
flyctl scale memory 4096 --app mechafil-api  # 4GB RAM
flyctl scale cpu 4 --app mechafil-api        # 4 CPUs
```

### Update Environment Variables

```bash
# Change cache update time
flyctl secrets set RELOAD_TRIGGER="03:00" --app mechafil-cache-updater

# Enable test mode (2-minute updates)
flyctl secrets set RELOAD_TEST_MODE="true" --app mechafil-cache-updater

# Restart to apply
flyctl machine restart --app mechafil-cache-updater
```

---

## Troubleshooting

### Problem: API returns "No cache data found"

**Cause:** Cache updater hasn't populated cache yet.

**Solution:**
```bash
# Check cache updater status
flyctl status --app mechafil-cache-updater

# Check logs
flyctl logs --app mechafil-cache-updater

# Manually trigger cache update
flyctl machine restart --app mechafil-cache-updater

# Wait 1-2 minutes, then test API again
curl https://mechafil-api.fly.dev/health
```

### Problem: API slow on first request

**Cause:** This is normal - cold start from idle.

**What happens:**
1. Request arrives at Fly proxy
2. Fly starts the stopped machine (~1 second)
3. FastAPI loads cache from disk (~1 second)
4. Request is processed

**Total cold start: 2-3 seconds**

**If unacceptable:**
```bash
# Keep 1 instance always running
flyctl scale count 1 --app mechafil-api
```

### Problem: Build fails with "out of memory"

**Cause:** Building JAX dependencies requires lots of RAM.

**Solution:**
```bash
# Use Fly's remote builders (more RAM)
flyctl deploy --remote-only --app mechafil-api
```

### Problem: Cache updater crashes on startup

**Check logs:**
```bash
flyctl logs --app mechafil-cache-updater
```

**Common errors:**

1. **"SPACESCOPE_TOKEN not set"**
   ```bash
   flyctl secrets set SPACESCOPE_TOKEN="Bearer YOUR_TOKEN" --app mechafil-cache-updater
   flyctl deploy --app mechafil-cache-updater
   ```

2. **"Permission denied: /data/shared-cache"**
   - Volume not mounted correctly
   - Check `fly-cache-updater.toml` has `[mounts]` section
   - Redeploy

3. **"Connection timeout to Spacescope"**
   - Spacescope API is down
   - Wait and retry
   - Check your token is valid

### Problem: Volume full

**Check volume usage:**
```bash
flyctl ssh console --app mechafil-cache-updater
df -h /data/shared-cache
```

**If near 10GB:**
```bash
# Expand volume (can't shrink)
flyctl volumes extend vol_xxxxx --size 20

# Or clean old cache files
flyctl ssh console --app mechafil-cache-updater
cd /data/shared-cache
ls -lt  # See oldest files
# Manually delete if needed
```

### Problem: High costs

**Check billing:**
```bash
flyctl billing show
```

**Cost optimization:**
```bash
# Ensure API auto-stops (check fly-api.toml)
auto_stop_machines = "stop"
min_machines_running = 0

# Reduce API resources if possible
flyctl scale memory 1024 --app mechafil-api  # 1GB instead of 2GB
flyctl scale cpu 1 --app mechafil-api        # 1 CPU instead of 2

# Run cache updater only when needed (instead of continuous)
# Stop it, use GitHub Actions cron instead
flyctl machine stop --app mechafil-cache-updater
```

### Problem: Want to start over

**Complete reset:**
```bash
# Delete everything
flyctl apps destroy mechafil-api
flyctl apps destroy mechafil-cache-updater
flyctl volumes delete vol_xxxxx

# Start from Step 3 again
```

---

## Cost Estimate

**With this serverless setup:**

| Resource | Usage | Cost/month |
|----------|-------|------------|
| API Machine | ~1 hour/day active | ~$1-2 |
| Cache Updater Machine | Always on | ~$10-15 |
| Volume (10GB) | Persistent | ~$1.50 |
| Bandwidth | Light traffic | ~$0 (free tier) |
| **Total** | | **~$12-18/month** |

**If API gets high traffic (always on):**
- API Machine: ~$20-30/month
- Total: ~$32-47/month

**Free tier includes:**
- 3 shared-cpu-1x VMs (256MB each)
- 3GB storage
- 160GB bandwidth
- You'll need a credit card but usage is low

---

## Next Steps

After successful deployment:

1. **Set up custom domain** (optional)
   ```bash
   flyctl certs create mechafil.yourdomain.com --app mechafil-api
   ```

2. **Set up monitoring alerts**
   - Enable email alerts in Fly.io dashboard
   - Monitor error rates, response times

3. **Set up CI/CD** (optional)
   - GitHub Actions to auto-deploy on push
   - See `.github/workflows/fly-deploy.yml` example

4. **Add authentication** (if needed)
   - Implement API key middleware in FastAPI
   - Store keys in Fly secrets

5. **Backup strategy**
   - Fly volumes have snapshots
   - Consider periodic backups to S3/GCS

---

## Summary

**What you deployed:**
- ✅ Cache updater service (always-on, updates daily at 1 AM UTC)
- ✅ API service (auto-scales to zero, starts on request)
- ✅ Shared volume (10GB persistent storage)
- ✅ HTTPS endpoints (automatic SSL)

**Your endpoints:**
- https://mechafil-api.fly.dev/health
- https://mechafil-api.fly.dev/historical-data
- https://mechafil-api.fly.dev/simulate
- https://mechafil-api.fly.dev/docs

**Architecture benefits:**
- Scale to zero when idle (save money)
- Auto-start on request (serverless)
- Daily data updates (always fresh)
- Persistent cache (fast responses)
- No VPC/networking complexity
- Automatic HTTPS and routing

**Maintenance:**
- Cache updates automatically daily
- API starts/stops automatically
- Monitor with `flyctl logs` and `flyctl status`
- Update code with `flyctl deploy`

---

## Useful Commands Reference

```bash
# Status
flyctl status --app mechafil-api
flyctl status --app mechafil-cache-updater

# Logs
flyctl logs --app mechafil-api --follow
flyctl logs --app mechafil-cache-updater --follow

# SSH
flyctl ssh console --app mechafil-api
flyctl ssh console --app mechafil-cache-updater

# Restart
flyctl machine restart --app mechafil-api
flyctl machine restart --app mechafil-cache-updater

# Deploy updates
flyctl deploy --app mechafil-api
flyctl deploy --app mechafil-cache-updater

# Secrets
flyctl secrets list --app mechafil-cache-updater
flyctl secrets set KEY=VALUE --app mechafil-cache-updater

# Volume
flyctl volumes list
flyctl volumes extend vol_xxxxx --size 20

# Cleanup
flyctl apps destroy mechafil-api
flyctl apps destroy mechafil-cache-updater
flyctl volumes delete vol_xxxxx
```

---

**Deployment completed successfully! 🚀**
