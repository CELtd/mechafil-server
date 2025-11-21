# Quick Deployment Guide - Fly.io with GitHub Actions

## What You're Deploying

- **API Service**: FastAPI app that serves simulations (auto-scales to zero)
- **Cache Updater**: Background job that fetches data from Spacescope (runs daily via GitHub Actions)
- **Shared Volume**: 3GB persistent storage for DiskCache

**Cost**: ~$1-3/month (API scales to zero, cache runs 3-5 min/day)

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
# Create cache updater app
flyctl apps create mechafil-cache-updater

# Create 3GB volume (enough for ~100-500MB cache)
flyctl volumes create shared_cache --region fra --size 3 --app mechafil-cache-updater
```

**Region choice**: `fra` = Frankfurt. Change if needed, but both services must use the same region.

### Step 2: Set Spacescope Token Secret

```bash
# Remove quotes from token value!
flyctl secrets set SPACESCOPE_TOKEN='Bearer YOUR_TOKEN_HERE' --app mechafil-cache-updater
```

**Critical**: The token value should NOT have quotes around it.

### Step 3: Deploy Cache Updater (Initial Population)

```bash
# Deploy to build image and run first cache update
flyctl deploy --config fly-cache-updater.toml --app mechafil-cache-updater

# Monitor logs (takes 2-3 minutes to fetch from Spacescope)
flyctl logs --app mechafil-cache-updater
```

**Look for**: `"✅ Cache update completed successfully!"`

**What happened**:
- Built Docker image with Python, Poetry, JAX, dependencies (~1-2GB)
- Fetched historical data from Spacescope API
- Wrote cache to `/data/shared-cache` volume
- Machine is now running continuously (we'll stop it next)

### Step 4: Configure for One-Shot Runs

```bash
# Get machine ID
flyctl machine list --app mechafil-cache-updater
# Copy the ID (e.g., d8d3dd6c2ed1d8)

# Update machine to exit after cache update
flyctl machine update <MACHINE_ID> \
  --app mechafil-cache-updater \
  --command "python -m services.cache_updater.main --once" \
  --yes

# Stop the machine (GitHub Actions will start it daily)
flyctl machine stop <MACHINE_ID> --app mechafil-cache-updater
```

**What this does**:
- Machine now runs with `--once` flag (exits after updating)
- Stopped machine costs $0
- GitHub Actions will start it daily

### Step 5: Set Up GitHub Actions Scheduling

**5.1: Get Required Information**

```bash
# Get Fly API token
flyctl auth token
# Copy the entire token

# Get machine ID (if you forgot)
flyctl machine list --app mechafil-cache-updater
# Copy the ID
```

**5.2: Add GitHub Secrets**

1. Go to your GitHub repo: **Settings > Secrets and variables > Actions**
2. Click **New repository secret**
3. Add these two secrets:

| Name | Value |
|------|-------|
| `FLY_API_TOKEN` | (paste token from `flyctl auth token`) |
| `CACHE_MACHINE_ID` | (paste machine ID, e.g., `d8d3dd6c2ed1d8`) |

**5.3: Ensure Workflow is in Main Branch**

The workflow file `.github/workflows/update-cache-daily.yml` must be in the `main` branch for scheduled runs:

```bash
# If you're on a different branch
git checkout main
git checkout <your-branch> -- .github/workflows/update-cache-daily.yml
git add .github/workflows/update-cache-daily.yml
git commit -m "Add daily cache update workflow"
git push origin main
```

**5.4: Test Workflow Manually**

1. Go to **Actions** tab in GitHub
2. Click **Daily Cache Update**
3. Click **Run workflow** > Select `main` branch > **Run workflow**
4. Watch it complete (~3-5 minutes)

**What the workflow does**:
- ⏰ Runs automatically daily at **1:00 AM UTC**
- 🚀 Starts the stopped Fly.io machine
- 📦 Machine updates cache from Spacescope
- 🛑 Machine exits and stops automatically
- 💰 Costs ~$0.01 per run

### Step 6: Deploy API Service

```bash
# Create API app
flyctl apps create mechafil-api

# Deploy API
flyctl deploy --config fly-api.toml --app mechafil-api
```

**What this does**:
- Deploys FastAPI application
- Mounts same `shared_cache` volume (read-only)
- Configured to auto-scale to zero when idle
- Auto-starts on incoming requests

**Deployment takes**: 2-3 minutes

### Step 7: Test Your API

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

**First request may take 2-3 seconds** (cold start from idle).

---

## Architecture

```
GitHub Actions (1:00 AM UTC daily)
         │
         │ Start machine
         ▼
┌────────────────────┐
│  Cache Updater     │
│  - Fetches data    │──────► Spacescope API
│  - Updates cache   │
│  - Exits (stopped) │
└─────────┬──────────┘
          │ Write
          ▼
  ┌───────────────┐
  │ Shared Volume │◄────────┐
  │  (3GB cache)  │         │ Read
  └───────┬───────┘         │
          │                 │
          └─────────────────┤
                            │
                   ┌────────┴─────────┐
Internet Request──►│   API Service    │
                   │  - Auto-start    │
                   │  - Auto-stop     │
                   │  - Reads cache   │
                   └──────────────────┘
```

---

## Monitoring

### Check Status

```bash
# App status
flyctl status --app mechafil-api
flyctl status --app mechafil-cache-updater

# Machine list
flyctl machine list --app mechafil-api
flyctl machine list --app mechafil-cache-updater
```

### View Logs

```bash
# Real-time logs
flyctl logs --app mechafil-api --follow
flyctl logs --app mechafil-cache-updater --follow

# Recent logs only
flyctl logs --app mechafil-api
```

### Manual Cache Update

Trigger cache update manually (without waiting for scheduled run):

```bash
# Method 1: Via GitHub Actions
# Go to Actions tab > Daily Cache Update > Run workflow

# Method 2: Via Fly CLI
MACHINE_ID=<your-machine-id>
flyctl machine start $MACHINE_ID --app mechafil-cache-updater
```

### SSH Into Machines

```bash
# API machine
flyctl ssh console --app mechafil-api

# Cache updater machine (must be running)
flyctl machine start <MACHINE_ID> --app mechafil-cache-updater
flyctl ssh console --app mechafil-cache-updater
```

---

## Cost Breakdown

| Component | Usage | Cost/Month |
|-----------|-------|------------|
| **API** | ~100 req/day, scales to zero | ~$0-1 |
| **Cache Updater** | 3-5 min/day (started by GitHub) | ~$0-1 |
| **Volume** | 3GB persistent storage | ~$0.45 |
| **GitHub Actions** | Free tier (2000 min/month) | $0 |
| **Total** | | **~$1-3** |

**If high traffic (API always on)**: ~$15-25/month

---

## Updating Code

### Update API

```bash
# Make code changes
git commit -am "Update API"

# Deploy
flyctl deploy --app mechafil-api
```

### Update Cache Updater

```bash
# Make code changes
git commit -am "Update cache updater"

# Rebuild image
flyctl deploy --config fly-cache-updater.toml --app mechafil-cache-updater

# Get new machine ID (if machine was recreated)
flyctl machine list --app mechafil-cache-updater

# Update GitHub secret CACHE_MACHINE_ID if needed
```

---

## Troubleshooting

### API Returns "No cache data found"

**Cause**: Cache updater hasn't populated cache yet.

**Fix**:
```bash
# Check if cache updater ran successfully
flyctl logs --app mechafil-cache-updater

# Manually trigger cache update
flyctl machine start <MACHINE_ID> --app mechafil-cache-updater

# Wait 3 minutes, then test API again
curl https://mechafil-api.fly.dev/health
```

### Spacescope API Errors (KeyError: 'data')

**Cause**: Token has quotes around it or is invalid.

**Fix**:
```bash
# Check .env file - remove quotes
cat .env | grep SPACESCOPE_TOKEN

# Update secret (no quotes!)
flyctl secrets set SPACESCOPE_TOKEN='Bearer YOUR_TOKEN' --app mechafil-cache-updater

# Restart machine
flyctl machine restart <MACHINE_ID> --app mechafil-cache-updater
```

### GitHub Action Fails

**Cause**: Secrets not set correctly.

**Fix**:
1. Verify secrets exist: GitHub repo > Settings > Secrets and variables > Actions
2. Ensure `FLY_API_TOKEN` and `CACHE_MACHINE_ID` are set
3. Re-run workflow

### Build Fails (Out of Memory)

**Fix**:
```bash
# Use Fly's remote builder (more RAM)
flyctl deploy --remote-only --app mechafil-api
```

### Volume Full

**Check usage**:
```bash
flyctl ssh console --app mechafil-cache-updater
df -h /data/shared-cache
exit
```

**Expand if needed**:
```bash
flyctl volumes list
flyctl volumes extend vol_xxxxx --size 5
```

---

## Important Notes

1. ✅ **Token format critical**: No quotes in `.env` or Fly secrets
2. ✅ **Region must match**: All resources in same region (fra)
3. ✅ **Workflow in main**: GitHub Actions only runs from default branch
4. ✅ **Machine ID in secrets**: Update if machine is recreated
5. ✅ **Cache updates daily**: 1:00 AM UTC via GitHub Actions
6. ✅ **API auto-scales**: Stops after ~5 min idle, starts on request
7. ✅ **Cold start**: First request takes 2-3 seconds

---

## Summary

**What you deployed**:
- ✅ Cache updater (stopped, triggered daily by GitHub Actions)
- ✅ API service (serverless, auto-scales to zero)
- ✅ Shared volume (3GB persistent cache storage)
- ✅ GitHub Actions workflow (daily at 1:00 AM UTC)

**Your endpoints**:
- https://mechafil-api.fly.dev/health
- https://mechafil-api.fly.dev/historical-data
- https://mechafil-api.fly.dev/simulate
- https://mechafil-api.fly.dev/docs

**Monthly cost**: ~$1-3

**Next steps**: See `FLY_DEPLOYMENT_LOG.md` for detailed explanations and advanced topics.
