# MechaFil Server Architecture

## Overview

The MechaFil server uses a **single-machine, externally-triggered** architecture that solves the volume sharing problem while maintaining serverless cost benefits.

## Architecture Diagram

```
GitHub Actions (1:00 AM UTC daily)
         |
         | POST /admin/update-cache
         v
┌────────────────────────────────────────┐
│   Fly.io Machine (mechafil-api)        │
│                                        │
│   ┌──────────────────────┐             │
│   │  FastAPI Service     │◄────────────┤ User requests
│   │  (Port 8000)         │             │
│   └──────┬───────────────┘             │
│          │                             │
│   ┌──────▼──────────────────┐          │
│   │  /admin/update-cache    │          │
│   │  (Admin Endpoint)       │          │
│   └──────┬──────────────────┘          │
│          │                             │
│          v Calls                        │
│   ┌──────────────────────┐             │
│   │  Cache Updater       │             │
│   │  (Python Module)     │─────────────┤ Spacescope API
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

## How It Works

### 1. Single Machine Setup
- **ONE machine** runs the FastAPI application
- Volume is attached to this machine (solves volume sharing problem)
- Machine can scale to zero when idle (serverless)
- Auto-starts on incoming HTTP requests

### 2. Cache Update Flow
1. **GitHub Actions triggers** daily at 1:00 AM UTC
2. **Sends POST request** to `https://mechafil-api.fly.dev/admin/update-cache`
3. **Machine wakes up** if stopped (auto-start on HTTP request)
4. **Admin endpoint** calls `services.cache_updater.main.update_cache()`
5. **Cache updater** fetches data from Spacescope API
6. **Writes to volume** at `/data/shared-cache`
7. **API reloads** historical data from updated cache
8. **Returns success** response to GitHub Actions
9. **Machine idles** and eventually stops (scale to zero)

### 3. API Request Flow
1. **User sends request** to `/simulate` or other endpoints
2. **Machine wakes up** if stopped
3. **API reads** cached data from volume
4. **Runs simulation** using mechafil-jax
5. **Returns results** to user
6. **Machine idles** after ~5 minutes, stops automatically

## Benefits

✅ **Volume sharing solved** - Single machine = single volume attachment
✅ **True serverless** - Scales to zero, starts on demand
✅ **External control** - GitHub Actions triggers cache updates
✅ **Low cost** - Only runs when serving requests (~$1-5/month)
✅ **No downtime** - API stays available during cache updates
✅ **Simple deployment** - One machine, one configuration
✅ **Easy maintenance** - Standard FastAPI patterns

## Cost Breakdown

| Component | Usage | Cost/Month |
|-----------|-------|------------|
| **Machine** | Idle most of time, scales to zero | ~$0-2 |
| **Cache Updates** | ~3-5 min/day triggered by GitHub | ~$0-1 |
| **API Requests** | ~100 req/day with cold starts | ~$0-2 |
| **Volume** | 3GB persistent storage | ~$0.45 |
| **GitHub Actions** | Free tier (2000 min/month) | $0 |
| **Total** | | **~$1-5** |

If API has high traffic and stays running 24/7: ~$15-20/month

## Key Files

### `fly.toml`
- Single machine configuration
- Serverless HTTP service (min_machines_running = 0)
- Volume mount at `/data/shared-cache`

### `services/api/main.py`
- FastAPI application with `/admin/update-cache` endpoint
- Calls cache updater module directly
- Reloads historical data after cache update

### `services/cache_updater/main.py`
- Standalone cache update logic
- Fetches from Spacescope API
- Writes to DiskCache at `/data/shared-cache`

### `.github/workflows/update-cache-daily.yml`
- Runs daily at 1:00 AM UTC
- Sends POST request to admin endpoint
- No Fly CLI needed (just curl)

### `docker/Dockerfile`
- Unified image with both API and cache-updater code
- Poetry dependencies for both services
- Creates `/data/shared-cache` directory

## Deployment Steps

### Initial Setup

```bash
# 1. Create app and volume
flyctl apps create mechafil-api
flyctl volumes create shared_cache --region fra --size 3 --app mechafil-api

# 2. Set Spacescope token secret
flyctl secrets set SPACESCOPE_TOKEN='Bearer YOUR_TOKEN' --app mechafil-api

# 3. Deploy
flyctl deploy --app mechafil-api

# 4. Test admin endpoint (populates cache for first time)
curl -X POST https://mechafil-api.fly.dev/admin/update-cache -v

# 5. Test API
curl https://mechafil-api.fly.dev/health
curl -X POST https://mechafil-api.fly.dev/simulate -H "Content-Type: application/json" -d '{}'
```

### GitHub Actions Setup

1. Go to repo **Settings > Secrets and variables > Actions**
2. No secrets needed (endpoint is public)
3. Workflow runs automatically daily at 1:00 AM UTC

### Updating Code

```bash
# Make changes
git commit -am "Update code"

# Deploy
flyctl deploy --app mechafil-api

# Test
curl https://mechafil-api.fly.dev/health
```

## Security Considerations

### Admin Endpoint Protection (Optional)

The `/admin/update-cache` endpoint is currently public. For production, consider:

**Option 1: API Key Authentication**
```python
@app.post("/admin/update-cache")
async def update_cache(api_key: str = Header(None)):
    if api_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of code
```

**Option 2: IP Allowlist**
Only allow GitHub Actions IPs (requires custom middleware)

**Option 3: Keep Public**
Cache updates are idempotent and not destructive, so keeping public is acceptable for low-value targets

## Monitoring

### Check Status
```bash
flyctl status --app mechafil-api
flyctl machine list --app mechafil-api
```

### View Logs
```bash
# Real-time
flyctl logs --app mechafil-api --follow

# Recent only
flyctl logs --app mechafil-api
```

### Manual Cache Update
```bash
# Via curl
curl -X POST https://mechafil-api.fly.dev/admin/update-cache -v

# Via GitHub Actions
# Go to Actions tab > Daily Cache Update > Run workflow
```

### SSH Into Machine
```bash
flyctl ssh console --app mechafil-api

# Check cache contents
python3 -c "from diskcache import Cache; c = Cache('/data/shared-cache'); print(list(c))"
```

## Troubleshooting

### Cache Update Fails

**Check logs:**
```bash
flyctl logs --app mechafil-api
```

**Common issues:**
- Spacescope token expired or invalid
- Network timeout from Spacescope API
- Volume full (check with `df -h /data/shared-cache`)

**Fix:**
```bash
# Update token
flyctl secrets set SPACESCOPE_TOKEN='Bearer NEW_TOKEN' --app mechafil-api

# Restart machine
flyctl machine restart <MACHINE_ID> --app mechafil-api
```

### API Returns "No cache data found"

**Cause:** Cache hasn't been populated yet

**Fix:**
```bash
# Manually trigger cache update
curl -X POST https://mechafil-api.fly.dev/admin/update-cache -v

# Wait 2-3 minutes, then test
curl https://mechafil-api.fly.dev/health
```

### Machine Won't Stop (Always Running)

**Cause:** High traffic keeping machine active

**Check:**
```bash
flyctl machine list --app mechafil-api
```

**Fix:** This is normal if you have steady traffic. Machine will stop after idle period.

## Comparison with Previous Architecture

### Old Architecture (Two Apps)
❌ Volumes can't be shared between apps
❌ Required two separate deployments
❌ Complex orchestration with machine lifecycle
❌ More expensive (two apps)

### Old Architecture (Two Process Groups)
❌ Volumes can't be shared between machines
❌ Cache-updater machine had no volume access
❌ Data not persisted

### Current Architecture (Single Machine + Admin Endpoint)
✅ Single machine = volume attached
✅ Simple deployment (one app)
✅ Externally triggered via HTTP
✅ True serverless (scales to zero)
✅ Low cost (~$1-5/month)
✅ No downtime during updates

## Alternative Approaches Considered

### 1. Always-On Single Machine with Background Scheduler
- ❌ Higher cost (~$10-15/month)
- ✅ Simpler (no external trigger needed)

### 2. Stop API, Run Cache Update, Restart API
- ❌ API downtime during updates
- ❌ Complex orchestration in GitHub Actions

### 3. Object Storage (S3/R2/Tigris)
- ❌ Additional dependency
- ❌ Code changes needed
- ❌ Network latency
- ✅ True separation of concerns

### 4. Current Solution (Admin Endpoint)
- ✅ **Best balance** of cost, simplicity, and functionality

## Future Enhancements

1. **Authentication** - Add API key or IP allowlist for admin endpoint
2. **Caching Strategy** - Add cache versioning and rollback capability
3. **Monitoring** - Add metrics for cache update success/failure
4. **Alerting** - Send notifications if cache update fails
5. **Rate Limiting** - Protect admin endpoint from abuse
6. **Health Checks** - Add cache age monitoring to health endpoint
