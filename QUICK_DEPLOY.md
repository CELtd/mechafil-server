# Quick Deployment Reference

## What Was Created

Three files for Fly.io deployment:

1. **fly-api.toml** - API service configuration (serverless, scales to zero)
2. **fly-cache-updater.toml** - Cache updater configuration (runs continuously)
3. **FLY_DEPLOYMENT_LOG.md** - Complete deployment guide with explanations

## Prerequisites Verified

- ✅ Spacescope token is configured in `.env`
- ✅ Docker setup exists (`docker/api.Dockerfile`, `docker/cache-updater.Dockerfile`)
- ✅ Git repository is on `external-cache` branch

## Quick Deployment Commands

```bash
# 1. Install Fly CLI (one-time)
curl -L https://fly.io/install.sh | sh
export PATH="$HOME/.fly/bin:$PATH"

# 2. Login to Fly.io (one-time)
flyctl auth login

# 3. Create shared volume (one-time)
flyctl volumes create shared_cache --region fra --size 3

# 4. Deploy cache updater (populates cache)
flyctl apps create mechafil-cache-updater
flyctl secrets set SPACESCOPE_TOKEN="$(grep SPACESCOPE_TOKEN .env | cut -d= -f2-)" --app mechafil-cache-updater
flyctl deploy --config fly-cache-updater.toml --app mechafil-cache-updater

# 5. Wait for cache to populate (check logs)
flyctl logs --app mechafil-cache-updater

# 6. Deploy API (serves requests)
flyctl apps create mechafil-api
flyctl deploy --config fly-api.toml --app mechafil-api

# 7. Test
curl https://mechafil-api.fly.dev/health
```

## After Deployment

**Your API will be available at:**
- https://mechafil-api.fly.dev/health
- https://mechafil-api.fly.dev/historical-data
- https://mechafil-api.fly.dev/simulate
- https://mechafil-api.fly.dev/docs

**Monitoring:**
```bash
# Check status
flyctl status --app mechafil-api
flyctl status --app mechafil-cache-updater

# View logs
flyctl logs --app mechafil-api --follow
flyctl logs --app mechafil-cache-updater --follow

# SSH into machines
flyctl ssh console --app mechafil-api
flyctl ssh console --app mechafil-cache-updater
```

**Updates:**
```bash
# After code changes
flyctl deploy --app mechafil-api
flyctl deploy --app mechafil-cache-updater
```

## Architecture

```
Internet Request
      │
      ▼
Fly.io HTTPS Proxy (automatic)
      │
      ▼
┌──────────────────┐     ┌─────────────────┐
│   API Service    │────►│  Shared Volume  │
│   (serverless)   │     │  (10GB cache)   │
│ - Auto-start     │     │  - DiskCache    │
│ - Auto-stop      │     │  - Persistent   │
└──────────────────┘     └────────▲────────┘
                                  │
                         ┌────────┴────────┐
                         │ Cache Updater   │
                         │ (always-on)     │
                         │ - Daily updates │
                         └─────────────────┘
```

## Cost Estimate

**~$12-18/month** for light traffic:
- API: ~$1-2 (scales to zero)
- Cache Updater: ~$10-15 (always on)
- Volume: ~$1.50 (10GB storage)

## Important Notes

1. **Deploy order matters:** Cache updater FIRST, then API
2. **Region must match:** Both services and volume in same region (fra)
3. **Volume is shared:** Both services mount the same volume
4. **Cache updates daily:** At 1:00 AM UTC (configurable)
5. **API auto-stops:** After ~5 minutes of inactivity
6. **Cold start:** First request takes 2-3 seconds after idle

## Troubleshooting

**API says "no cache found":**
```bash
# Force cache update
flyctl machine restart --app mechafil-cache-updater
# Wait 1-2 minutes, then try API again
```

**Build fails:**
```bash
# Use remote builder (more RAM)
flyctl deploy --remote-only --app mechafil-api
```

**Need help:**
- Read `FLY_DEPLOYMENT_LOG.md` for detailed explanations
- Run `flyctl doctor` to check configuration
- Check logs: `flyctl logs --app APP_NAME`

## Full Documentation

See **FLY_DEPLOYMENT_LOG.md** for:
- Detailed step-by-step instructions
- Explanation of what each command does
- Troubleshooting guide
- Monitoring and maintenance
- Cost optimization tips
