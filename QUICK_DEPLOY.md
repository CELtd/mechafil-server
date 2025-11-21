# Quick Deployment Guide - Single Machine with External Triggers

## What You're Deploying

- **Single Machine**: FastAPI app running API service with built-in cache update capability
- **Admin Endpoint**: `/admin/update-cache` for triggering cache updates externally
- **Shared Volume**: 3GB persistent storage for DiskCache (attached to the single machine)
- **GitHub Actions**: Daily trigger at 1:00 AM UTC that calls the admin endpoint

**Cost**: ~$1-5/month (machine scales to zero when idle, cache updates via HTTP)

**Architecture**: Single machine solves the Fly.io volume limitation (volumes can only be attached to one machine at a time)

---

## Prerequisites

### ⚠️ Critical: Fix .env File First

Your `SPACESCOPE_TOKEN` must **NOT** have quotes:

```bash
# ❌ Wrong (causes API failures)
SPACESCOPE_TOKEN="Bearer ghp_..."

# ✅ Correct
SPACESCOPE_TOKEN=Bearer ghp_...
```

### Verify Setup

- ✅ Fly CLI installed: `flyctl version`
- ✅ Logged in: `flyctl auth whoami`
- ✅ Docker working: `docker --version`

---

## Deployment Steps

### Step 1: Create App and Volume

```bash
# Create single app
flyctl apps create mechafil-api

# Create 3GB volume (enough for ~100-500MB cache)
flyctl volumes create shared_cache --region fra --size 3 --app mechafil-api
```

**Region choice**: `fra` = Frankfurt. Change if needed.

### Step 2: Set Spacescope Token Secret

```bash
# Remove quotes from token value!
flyctl secrets set SPACESCOPE_TOKEN='Bearer YOUR_TOKEN_HERE' --app mechafil-api
```

**Critical**: The token value should NOT have quotes around it.

### Step 3: Deploy the Application

```bash
# Deploy the single machine with both API and cache updater code
flyctl deploy --app mechafil-api

# Monitor the deployment
flyctl logs --app mechafil-api
```

**What happened**:
- Built Docker image with Python, Poetry, JAX, dependencies (~1-2GB)
- Deployed single machine with FastAPI application
- Volume attached to this machine
- Admin endpoint `/admin/update-cache` now available

### Step 4: Initial Cache Population

```bash
# Trigger the first cache update via admin endpoint
curl -X POST https://mechafil-api.fly.dev/admin/update-cache \
  -H "Content-Type: application/json" \
  -v

# This takes ~40 seconds
# Look for: {"status":"success","message":"Cache updated and historical data reloaded"}
```

**What this does**:
- Wakes up the machine if stopped
- Calls cache updater logic (fetches from Spacescope API)
- Writes cache to `/data/shared-cache` volume
- Reloads historical data in the API
- Returns success response

### Step 5: Set Up GitHub Actions Scheduling

**5.1: Ensure Workflow is in Main Branch**

The workflow file `.github/workflows/update-cache-daily.yml` must be in the `main` branch for scheduled runs:

```bash
# If you're on a different branch
git checkout main
git merge <your-branch>  # or cherry-pick the workflow file
git push origin main
```

**5.2: No Secrets Required!**

The new architecture doesn't require any GitHub secrets because:
- The admin endpoint is publicly accessible (you can add auth later if needed)
- GitHub Actions just sends an HTTP POST request

**5.3: Test Workflow Manually**

1. Go to **Actions** tab in GitHub
2. Click **Daily Cache Update**
3. Click **Run workflow** > Select `main` branch > **Run workflow**
4. Watch it complete (~40-50 seconds)

**What the workflow does**:
- ⏰ Runs automatically daily at **1:00 AM UTC**
- 📡 Sends POST request to `/admin/update-cache` endpoint
- 🚀 Machine wakes up if stopped (auto-start)
- 📦 Admin endpoint triggers cache update from Spacescope
- ✅ Returns success/failure status
- 💤 Machine idles and eventually stops (scale to zero)
- 💰 Costs ~$0.01-0.02 per run

### Step 6: Test Your API

```bash
# Health check
curl https://mechafil-api.fly.dev/health

# Historical data
curl https://mechafil-api.fly.dev/historical-data

# Run simulation with defaults
curl -X POST https://mechafil-api.fly.dev/simulate \
  -H "Content-Type: application/json" \
  -d '{}'

# Custom simulation
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

**Cold start timing:**
- Health check: ~15 seconds (machine start + JAX/FastAPI initialization)
- First simulation: ~5-6 seconds (includes lazy loading of cache data)
- Subsequent requests: <1 second (cache already in memory)

---

## Architecture

```
GitHub Actions (1:00 AM UTC daily)
         |
         | POST /admin/update-cache
         v
┌────────────────────────────────────────┐
│   Single Fly.io Machine                │
│                                        │
│   ┌──────────────────────┐             │
│   │  FastAPI Service     │◄────────────┤ User HTTP requests
│   │  (Port 8000)         │             │
│   └──────┬───────────────┘             │
│          │                             │
│   ┌──────▼──────────────────┐          │
│   │  /admin/update-cache    │          │
│   │  (Admin Endpoint)       │          │
│   └──────┬──────────────────┘          │
│          │                             │
│          v Triggers                     │
│   ┌──────────────────────┐             │
│   │  Cache Updater       │─────────────┤ Spacescope API
│   │  (Python Module)     │             │
│   └──────┬───────────────┘             │
│          │                             │
│          v Write/Read                   │
│   ┌──────────────────────┐             │
│   │  Volume              │             │
│   │  /data/shared-cache  │             │
│   │  (3GB persistent)    │             │
│   └──────────────────────┘             │
└────────────────────────────────────────┘
```

**Key Benefits**:
- ✅ **Volume sharing solved**: Single machine = volume attached
- ✅ **External trigger**: GitHub Actions controls when updates happen
- ✅ **True serverless**: Machine scales to zero when idle
- ✅ **No downtime**: API stays available during cache updates
- ✅ **Simple**: No multi-machine orchestration needed

---

## Monitoring

### Check Status

```bash
# App status
flyctl status --app mechafil-api

# Machine list
flyctl machine list --app mechafil-api

# Check if machine is running or stopped
flyctl machine status <MACHINE_ID> --app mechafil-api
```

### View Logs

```bash
# Real-time logs
flyctl logs --app mechafil-api --follow

# Recent logs only
flyctl logs --app mechafil-api
```

### Manual Cache Update

Trigger cache update manually (without waiting for scheduled run):

```bash
# Method 1: Via Admin Endpoint (Recommended)
curl -X POST https://mechafil-api.fly.dev/admin/update-cache \
  -H "Content-Type: application/json" \
  -v

# Method 2: Via GitHub Actions UI
# Go to Actions tab > Daily Cache Update > Run workflow
```

### SSH Into Machine

```bash
# SSH into the machine
flyctl ssh console --app mechafil-api

# Check cache contents
python3 -c "from diskcache import Cache; c = Cache('/data/shared-cache'); print(list(c))"

# Check cache directory size
df -h /data/shared-cache
```

---

## Cost Breakdown

| Component | Usage | Cost/Month |
|-----------|-------|------------|
| **Single Machine** | Idles most of time, scales to zero | ~$0-2 |
| **Cache Updates** | ~40 sec/day triggered by GitHub | ~$0-1 |
| **API Requests** | ~100 req/day with cold starts | ~$0-2 |
| **Volume** | 3GB persistent storage | ~$0.45 |
| **GitHub Actions** | Free tier (2000 min/month) | $0 |
| **Total** | | **~$1-5** |

**If high traffic (machine always on)**: ~$15-20/month

---

## Updating Code

### Update Application (API or Cache Updater)

```bash
# Make code changes to API or cache updater
git commit -am "Update application"

# Deploy (rebuilds image and updates machine)
flyctl deploy --app mechafil-api

# Test the changes
curl https://mechafil-api.fly.dev/health
```

**Note**: Both API and cache updater code are in the same image, so one deployment updates everything.

---

## Troubleshooting

### API Returns "No cache data found"

**Cause**: Cache hasn't been populated yet.

**Fix**:
```bash
# Trigger cache update via admin endpoint
curl -X POST https://mechafil-api.fly.dev/admin/update-cache -v

# Wait ~40 seconds for it to complete
# Then test API again
curl https://mechafil-api.fly.dev/health
```

### Admin Endpoint Timeout or Error

**Cause**: Spacescope API issues or token problems.

**Fix**:
```bash
# Check logs for errors
flyctl logs --app mechafil-api

# Verify secret is set correctly (no quotes!)
flyctl secrets list --app mechafil-api

# Update secret if needed
flyctl secrets set SPACESCOPE_TOKEN='Bearer YOUR_TOKEN' --app mechafil-api

# Restart machine
flyctl machine restart <MACHINE_ID> --app mechafil-api
```

### Machine Won't Start Automatically

**Cause**: `auto_start_machines` might be disabled.

**Fix**:
```bash
# Check fly.toml - ensure it has:
# [http_service]
#   auto_start_machines = true
#   auto_stop_machines = "stop"
#   min_machines_running = 0

# If you changed fly.toml, redeploy:
flyctl deploy --app mechafil-api
```

### GitHub Action Fails

**Cause**: Network timeout or admin endpoint error.

**Fix**:
1. Check workflow logs in GitHub Actions
2. Test admin endpoint manually: `curl -X POST https://mechafil-api.fly.dev/admin/update-cache -v`
3. Check Fly.io logs: `flyctl logs --app mechafil-api`
4. Re-run workflow in GitHub Actions UI

### Build Fails (Out of Memory)

**Fix**:
```bash
# Use Fly's remote builder (more RAM)
flyctl deploy --remote-only --app mechafil-api
```

### Volume Full

**Check usage**:
```bash
flyctl ssh console --app mechafil-api
df -h /data/shared-cache
exit
```

**Expand if needed**:
```bash
flyctl volumes list --app mechafil-api
flyctl volumes extend vol_xxxxx --size 5
```

---

## Security Considerations

### Protecting the Admin Endpoint (Optional)

The `/admin/update-cache` endpoint is currently public. For production, consider adding authentication:

**Option 1: API Key in Header**

Add to `services/api/main.py`:
```python
@app.post("/admin/update-cache")
async def update_cache(api_key: str = Header(None)):
    if api_key != os.getenv("ADMIN_API_KEY"):
        raise HTTPException(status_code=403, detail="Forbidden")
    # ... rest of code
```

Then set secret:
```bash
flyctl secrets set ADMIN_API_KEY='your-secret-key' --app mechafil-api
```

Update GitHub Actions workflow:
```yaml
- name: Trigger Cache Update
  run: |
    curl -X POST https://mechafil-api.fly.dev/admin/update-cache \
      -H "Content-Type: application/json" \
      -H "api-key: ${{ secrets.ADMIN_API_KEY }}" \
      -f -v
```

**Option 2: Keep Public**

For low-value/internal services, keeping it public is acceptable because:
- Cache updates are idempotent (safe to call multiple times)
- Not destructive operation
- Worst case: extra Spacescope API calls and compute time

---

## Important Notes

1. ✅ **Token format critical**: No quotes in `.env` or Fly secrets
2. ✅ **Single machine architecture**: Solves Fly.io volume sharing limitation
3. ✅ **Admin endpoint**: Cache updates triggered via HTTP POST
4. ✅ **Workflow in main**: GitHub Actions only runs from default branch
5. ✅ **Cache updates daily**: 1:00 AM UTC via GitHub Actions
6. ✅ **Machine auto-scales**: Stops after idle period, starts on request
7. ✅ **Cold start**: Health check passes in ~15 seconds (JAX import takes time)
8. ✅ **Lazy loading**: Cache loads on first request (not at startup) for faster boot
9. ✅ **Cache update time**: ~40 seconds to fetch from Spacescope and reload

---

## Summary

**What you deployed**:
- ✅ Single machine running FastAPI with admin endpoint
- ✅ Volume attached to machine for persistent cache
- ✅ GitHub Actions workflow triggering daily cache updates via HTTP
- ✅ Serverless configuration (auto-start, auto-stop, min=0)

**Your endpoints**:
- https://mechafil-api.fly.dev/health
- https://mechafil-api.fly.dev/historical-data
- https://mechafil-api.fly.dev/simulate
- https://mechafil-api.fly.dev/docs
- https://mechafil-api.fly.dev/admin/update-cache (admin endpoint)

**Monthly cost**: ~$1-5

**Architecture advantages**:
- Simple single-machine deployment
- Solves Fly.io volume attachment limitation
- Externally triggered cache updates
- True serverless with auto-scaling
- No complex machine orchestration

**Next steps**: See `ARCHITECTURE.md` for detailed architecture explanation and alternative approaches considered.
