# Cloud Deployment Guide

This guide shows how to deploy mechafil-server with a serverless architecture pattern across three cloud platforms:

- **API Gateway** - Expose endpoints
- **Serverless Compute** - Scale to zero when idle
- **Shared Storage** - Persistent cache volume
- **Scheduled Job** - Daily cache updates

## Architecture Overview

```
┌─────────────────────┐
│   API Gateway       │
│   (HTTP Routing)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐         ┌─────────────────────┐
│  API Service        │  Read   │  Shared Cache       │
│  (Serverless)       │◄────────┤  (Volume/Storage)   │
│  - Scales to zero   │         │  - Persistent       │
│  - Cold start       │  Write  │  - DiskCache files  │
└─────────────────────┘ ◄────── └─────────────────────┘
                                          ▲
                                          │ Write
                                          │
                                 ┌────────┴────────┐
                                 │  Cache Updater  │
                                 │  (Cron Job)     │
                                 │  - Runs daily   │
                                 │  - Fetches data │
                                 └─────────────────┘
```

## Platform Comparison

| Feature | AWS | Google Cloud | Fly.io |
|---------|-----|--------------|--------|
| **API Gateway** | API Gateway | Cloud Run (built-in) | Machines (built-in) |
| **Compute** | Lambda + EFS / Fargate | Cloud Run | Machines |
| **Storage** | EFS / S3 | Filestore / GCS | Volumes |
| **Scheduler** | EventBridge | Cloud Scheduler | Fly Machines API |
| **Cold Start** | ~1-5s (Lambda) | ~1-3s | ~1-2s |
| **Cost (idle)** | Low (EFS charged) | Very Low | Very Low |
| **Complexity** | High | Medium | Low |

---

## 1. AWS Deployment

AWS offers two approaches: Lambda with EFS (true serverless) or ECS Fargate (containerized serverless).

### Option A: Lambda + EFS (Recommended for serverless)

#### Architecture
- **API Gateway** → Lambda (API handler)
- **EFS** (Elastic File System) for shared cache
- **EventBridge** for daily cache updates
- **Lambda Layer** for JAX dependencies (large)

#### Prerequisites
```bash
# Install AWS CLI and SAM CLI
pip install awscli aws-sam-cli
aws configure
```

#### Step 1: Create EFS File System

```bash
# Create EFS
aws efs create-file-system \
  --performance-mode generalPurpose \
  --throughput-mode bursting \
  --encrypted \
  --tags Key=Name,Value=mechafil-cache

# Note the FileSystemId from output
EFS_ID=fs-xxxxxxxxx

# Create mount target in your VPC subnets (replace with your subnet/security group)
aws efs create-mount-target \
  --file-system-id $EFS_ID \
  --subnet-id subnet-xxxxxxxx \
  --security-groups sg-xxxxxxxx

# Create access point for cache directory
aws efs create-access-point \
  --file-system-id $EFS_ID \
  --posix-user Uid=1000,Gid=1000 \
  --root-directory "Path=/mechafil-cache,CreationInfo={OwnerUid=1000,OwnerGid=1000,Permissions=755}" \
  --tags Key=Name,Value=mechafil-cache-access

# Note the AccessPointId
ACCESS_POINT_ID=fsap-xxxxxxxxx
```

#### Step 2: Create Docker Images for Lambda

**lambda/api.Dockerfile:**
```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY shared/ ./shared/
COPY services/api/ ./services/api/

# Install dependencies (without dev dependencies)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Lambda handler
COPY lambda/api_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["api_handler.lambda_handler"]
```

**lambda/cache_updater.Dockerfile:**
```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY shared/ ./shared/
COPY services/cache_updater/ ./services/cache_updater/

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Lambda handler
COPY lambda/cache_updater_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["cache_updater_handler.lambda_handler"]
```

#### Step 3: Create Lambda Handlers

**lambda/api_handler.py:**
```python
import os
import json
from mangum import Mangum
from services.api.main import app

# Set environment for EFS mount
os.environ.setdefault("USE_SHARED_CACHE", "true")
os.environ.setdefault("SHARED_CACHE_DIR", "/mnt/efs/mechafil-cache")

# Mangum adapter converts ASGI (FastAPI) to Lambda events
handler = Mangum(app, lifespan="off")  # lifespan handled by Lambda

def lambda_handler(event, context):
    """Lambda handler for API Gateway events"""
    return handler(event, context)
```

**lambda/cache_updater_handler.py:**
```python
import os
import sys

# Set environment
os.environ.setdefault("USE_SHARED_CACHE", "true")
os.environ.setdefault("SHARED_CACHE_DIR", "/mnt/efs/mechafil-cache")

def lambda_handler(event, context):
    """Lambda handler for scheduled cache updates"""
    # Import here to avoid cold start issues
    from services.cache_updater.data import Data
    from shared.config import settings

    print(f"Starting cache update at {settings.SHARED_CACHE_DIR}")

    try:
        data = Data()
        data.load_historical_data()

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Cache updated successfully',
                'cache_dir': str(settings.SHARED_CACHE_DIR)
            })
        }
    except Exception as e:
        print(f"Error updating cache: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### Step 4: Deploy with SAM

**template.yaml:**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: MechaFil Server Serverless Deployment

Globals:
  Function:
    Timeout: 900  # 15 minutes for cache operations
    MemorySize: 2048
    Environment:
      Variables:
        USE_SHARED_CACHE: "true"
        SHARED_CACHE_DIR: "/mnt/efs/mechafil-cache"
        SPACESCOPE_TOKEN: !Ref SpacescopeToken

Parameters:
  SpacescopeToken:
    Type: String
    NoEcho: true
    Description: Spacescope API token
  FileSystemId:
    Type: String
    Description: EFS File System ID
  AccessPointId:
    Type: String
    Description: EFS Access Point ID
  SubnetIds:
    Type: CommaDelimitedList
    Description: VPC Subnet IDs
  SecurityGroupIds:
    Type: CommaDelimitedList
    Description: Security Group IDs

Resources:
  # API Lambda Function
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      PackageType: Image
      ImageUri: !Sub ${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/mechafil-api:latest
      Events:
        ApiEvent:
          Type: HttpApi
          Properties:
            Path: /{proxy+}
            Method: ANY
      FileSystemConfigs:
        - Arn: !Sub 'arn:aws:elasticfilesystem:${AWS::Region}:${AWS::AccountId}:access-point/${AccessPointId}'
          LocalMountPath: /mnt/efs
      VpcConfig:
        SubnetIds: !Ref SubnetIds
        SecurityGroupIds: !Ref SecurityGroupIds
      Policies:
        - Statement:
            - Effect: Allow
              Action:
                - elasticfilesystem:ClientMount
                - elasticfilesystem:ClientWrite
              Resource: !Sub 'arn:aws:elasticfilesystem:${AWS::Region}:${AWS::AccountId}:file-system/${FileSystemId}'

  # Cache Updater Lambda Function
  CacheUpdaterFunction:
    Type: AWS::Serverless::Function
    Properties:
      PackageType: Image
      ImageUri: !Sub ${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/mechafil-cache-updater:latest
      Events:
        DailySchedule:
          Type: Schedule
          Properties:
            Schedule: cron(0 1 * * ? *)  # Daily at 1:00 AM UTC
            Description: Daily cache update
      FileSystemConfigs:
        - Arn: !Sub 'arn:aws:elasticfilesystem:${AWS::Region}:${AWS::AccountId}:access-point/${AccessPointId}'
          LocalMountPath: /mnt/efs
      VpcConfig:
        SubnetIds: !Ref SubnetIds
        SecurityGroupIds: !Ref SecurityGroupIds
      Policies:
        - Statement:
            - Effect: Allow
              Action:
                - elasticfilesystem:ClientMount
                - elasticfilesystem:ClientWrite
              Resource: !Sub 'arn:aws:elasticfilesystem:${AWS::Region}:${AWS::AccountId}:file-system/${FileSystemId}'

Outputs:
  ApiEndpoint:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${ServerlessHttpApi}.execute-api.${AWS::Region}.amazonaws.com"
```

**Deploy:**
```bash
# Build and push Docker images to ECR
aws ecr create-repository --repository-name mechafil-api
aws ecr create-repository --repository-name mechafil-cache-updater

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push API
docker build -f lambda/api.Dockerfile -t mechafil-api .
docker tag mechafil-api:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/mechafil-api:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/mechafil-api:latest

# Build and push cache updater
docker build -f lambda/cache_updater.Dockerfile -t mechafil-cache-updater .
docker tag mechafil-cache-updater:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/mechafil-cache-updater:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/mechafil-cache-updater:latest

# Deploy with SAM
sam deploy \
  --template-file template.yaml \
  --stack-name mechafil-server \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    SpacescopeToken="Bearer YOUR_TOKEN" \
    FileSystemId=$EFS_ID \
    AccessPointId=$ACCESS_POINT_ID \
    SubnetIds="subnet-xxx,subnet-yyy" \
    SecurityGroupIds="sg-zzz"
```

#### Initial Cache Population

```bash
# Invoke cache updater manually first time
aws lambda invoke \
  --function-name mechafil-server-CacheUpdaterFunction-XXXXX \
  --invocation-type Event \
  /dev/stdout
```

### Option B: ECS Fargate + EFS (Container-based)

If Lambda's limitations are too restrictive, use ECS Fargate:

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name mechafil-cluster

# Use existing Dockerfiles (docker/api.Dockerfile, docker/cache-updater.Dockerfile)
# Push to ECR
# Create ECS task definitions with EFS volume mounts
# Create Fargate services with auto-scaling (min=0 for API)
# Use EventBridge to trigger cache updater task daily
```

**Benefits of Fargate:**
- Full Docker compatibility (no Lambda limitations)
- Easier debugging
- Better for large dependencies

**Drawbacks:**
- Slower cold starts (~30-60s vs Lambda's ~1-5s)
- More complex setup

---

## 2. Google Cloud Deployment

Google Cloud Run is ideal for this use case - it's containerized serverless with built-in endpoints.

### Architecture
- **Cloud Run** (API service with auto-scaling to 0)
- **Cloud Storage** (GCS bucket for cache) or **Filestore** (NFS)
- **Cloud Scheduler** (cron job for cache updates)
- **Cloud Build** (CI/CD)

### Prerequisites
```bash
# Install gcloud CLI
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Step 1: Choose Storage Strategy

#### Option A: Cloud Storage (Recommended - Simpler, Cheaper)

DiskCache can work with a local directory that syncs to GCS:

**Create startup script that downloads cache:**

**cloud-run/api-entrypoint.sh:**
```bash
#!/bin/bash
set -e

# Download cache from GCS to local disk
echo "Downloading cache from GCS..."
mkdir -p /tmp/shared-cache
gsutil -m rsync -r gs://${GCS_CACHE_BUCKET}/shared-cache /tmp/shared-cache

# Start API
export SHARED_CACHE_DIR=/tmp/shared-cache
exec python -m services.api.main
```

**cloud-run/cache-updater-entrypoint.sh:**
```bash
#!/bin/bash
set -e

# Download existing cache
mkdir -p /tmp/shared-cache
gsutil -m rsync -r gs://${GCS_CACHE_BUCKET}/shared-cache /tmp/shared-cache || true

# Update cache
export SHARED_CACHE_DIR=/tmp/shared-cache
python -m services.cache_updater.main --once

# Upload updated cache to GCS
echo "Uploading cache to GCS..."
gsutil -m rsync -r /tmp/shared-cache gs://${GCS_CACHE_BUCKET}/shared-cache

echo "Cache update complete"
```

#### Option B: Filestore (NFS - True Shared Volume)

More expensive but provides true shared filesystem:

```bash
# Create Filestore instance
gcloud filestore instances create mechafil-cache \
  --tier=BASIC_HDD \
  --file-share=name="mechafilcache",capacity=1TB \
  --network=name="default" \
  --zone=us-central1-a
```

### Step 2: Create Docker Images for Cloud Run

**cloud-run/api.Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install gsutil (if using GCS)
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy application
WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY shared/ ./shared/
COPY services/api/ ./services/api/

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction

# Copy entrypoint
COPY cloud-run/api-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Cloud Run expects PORT env var
ENV PORT=8080
ENV HOST=0.0.0.0

ENTRYPOINT ["/entrypoint.sh"]
```

**cloud-run/cache-updater.Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install gsutil
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy application
WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY shared/ ./shared/
COPY services/cache_updater/ ./services/cache_updater/

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction

# Copy entrypoint
COPY cloud-run/cache-updater-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

### Step 3: Build and Push to Container Registry

```bash
# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com

# Set variables
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1

# Create GCS bucket for cache
gsutil mb -l $REGION gs://${PROJECT_ID}-mechafil-cache

# Build and push API image
gcloud builds submit --tag gcr.io/$PROJECT_ID/mechafil-api -f cloud-run/api.Dockerfile .

# Build and push cache updater image
gcloud builds submit --tag gcr.io/$PROJECT_ID/mechafil-cache-updater -f cloud-run/cache-updater.Dockerfile .
```

### Step 4: Deploy Cloud Run Services

```bash
# Deploy API service (scales to 0)
gcloud run deploy mechafil-api \
  --image gcr.io/$PROJECT_ID/mechafil-api \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 10 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "USE_SHARED_CACHE=true,GCS_CACHE_BUCKET=${PROJECT_ID}-mechafil-cache" \
  --set-secrets "SPACESCOPE_TOKEN=spacescope-token:latest"

# Deploy cache updater service (not publicly accessible)
gcloud run deploy mechafil-cache-updater \
  --image gcr.io/$PROJECT_ID/mechafil-cache-updater \
  --platform managed \
  --region $REGION \
  --no-allow-unauthenticated \
  --min-instances 0 \
  --max-instances 1 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 900 \
  --set-env-vars "USE_SHARED_CACHE=true,GCS_CACHE_BUCKET=${PROJECT_ID}-mechafil-cache" \
  --set-secrets "SPACESCOPE_TOKEN=spacescope-token:latest"
```

### Step 5: Create Cloud Scheduler Job

```bash
# Get cache updater service URL
CACHE_UPDATER_URL=$(gcloud run services describe mechafil-cache-updater \
  --region $REGION \
  --format 'value(status.url)')

# Create service account for scheduler
gcloud iam service-accounts create cache-updater-invoker \
  --display-name "Cache Updater Invoker"

# Grant permission to invoke Cloud Run service
gcloud run services add-iam-policy-binding mechafil-cache-updater \
  --region $REGION \
  --member "serviceAccount:cache-updater-invoker@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/run.invoker"

# Create daily scheduled job (1:00 AM UTC)
gcloud scheduler jobs create http mechafil-cache-updater-daily \
  --location $REGION \
  --schedule "0 1 * * *" \
  --uri "${CACHE_UPDATER_URL}" \
  --http-method POST \
  --oidc-service-account-email "cache-updater-invoker@${PROJECT_ID}.iam.gserviceaccount.com" \
  --oidc-token-audience "${CACHE_UPDATER_URL}" \
  --time-zone "UTC" \
  --description "Daily cache update at 1:00 AM UTC"
```

### Step 6: Initial Cache Population

```bash
# Manually trigger first cache update
gcloud scheduler jobs run mechafil-cache-updater-daily --location $REGION

# Or invoke directly
gcloud run services proxy mechafil-cache-updater --region $REGION
```

### API Access

```bash
# Get API URL
API_URL=$(gcloud run services describe mechafil-api \
  --region $REGION \
  --format 'value(status.url)')

echo "API available at: $API_URL"

# Test
curl $API_URL/health
curl -X POST $API_URL/simulate -H "Content-Type: application/json" -d '{}'
```

---

## 3. Fly.io Deployment (Simplest)

Fly.io is already documented in README.md but here's the complete serverless setup:

### Architecture
- **Fly Machines** (containers with built-in auto-start/stop)
- **Fly Volumes** (persistent block storage)
- **Fly Machines API** (for scheduled jobs)

### Prerequisites
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login
```

### Step 1: Create Shared Volume

```bash
# Create volume in your preferred region
flyctl volumes create shared_cache \
  --region fra \
  --size 10
```

### Step 2: Configure and Deploy Cache Updater

**fly-cache-updater.toml:**
```toml
app = "mechafil-cache-updater"
primary_region = "fra"

[build]
  dockerfile = "docker/cache-updater.Dockerfile"

[env]
  USE_SHARED_CACHE = "true"
  SHARED_CACHE_DIR = "/data/shared-cache"
  RELOAD_TRIGGER = "01:00"
  RELOAD_TEST_MODE = "false"

[mounts]
  source = "shared_cache"
  destination = "/data/shared-cache"

# No [http_service] - this is a background job
```

```bash
# Create app
flyctl apps create mechafil-cache-updater

# Set secrets
flyctl secrets set SPACESCOPE_TOKEN="Bearer YOUR_TOKEN" -a mechafil-cache-updater

# Deploy (will run continuously, updating daily)
flyctl deploy --config fly-cache-updater.toml
```

### Step 3: Configure and Deploy API

**fly-api.toml:**
```toml
app = "mechafil-api"
primary_region = "fra"

[build]
  dockerfile = "docker/api.Dockerfile"

[env]
  USE_SHARED_CACHE = "true"
  SHARED_CACHE_DIR = "/data/shared-cache"
  PORT = "8080"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "stop"   # Scale to zero when idle
  auto_start_machines = true    # Auto-start on request
  min_machines_running = 0      # True serverless
  max_machines_running = 10

[mounts]
  source = "shared_cache"
  destination = "/data/shared-cache"

[[vm]]
  memory = "2gb"
  cpu_kind = "shared"
  cpus = 2
```

```bash
# Create app
flyctl apps create mechafil-api

# Deploy
flyctl deploy --config fly-api.toml

# Check status
flyctl status -a mechafil-api
```

### Alternative: Lambda-Style Cache Updates (One-Shot)

Instead of running cache updater continuously, trigger it on-demand:

**GitHub Actions (.github/workflows/update-cache.yml):**
```yaml
name: Update Fly.io Cache

on:
  schedule:
    - cron: '0 1 * * *'  # Daily at 1:00 AM UTC
  workflow_dispatch:  # Manual trigger

jobs:
  update-cache:
    runs-on: ubuntu-latest
    steps:
      - name: Setup Fly.io CLI
        uses: superfly/flyctl-actions/setup-flyctl@master

      - name: Run Cache Update
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
        run: |
          flyctl machine run \
            --app mechafil-cache-updater \
            --region fra \
            --volume shared_cache:/data/shared-cache \
            --env USE_SHARED_CACHE=true \
            --env SHARED_CACHE_DIR=/data/shared-cache \
            --env SPACESCOPE_TOKEN="${{ secrets.SPACESCOPE_TOKEN }}" \
            --entrypoint "" \
            -- python -m services.cache_updater.main --once
```

### API Access

```bash
# Get URL
flyctl status -a mechafil-api

# Test
curl https://mechafil-api.fly.dev/health
curl -X POST https://mechafil-api.fly.dev/simulate \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Cost Comparison (Estimated Monthly)

### Low Traffic (100 requests/day, 3GB cache)

| Platform | Compute | Storage | Scheduler | Total/month |
|----------|---------|---------|-----------|-------------|
| **AWS Lambda + EFS** | $0 (free tier) | ~$3.20 (EFS) | $0 | ~$3-5 |
| **AWS Fargate + EFS** | ~$8-15 | ~$3.20 (EFS) | $0 | ~$11-18 |
| **Google Cloud Run + GCS** | $0 (free tier) | ~$0.10 (GCS) | $0 | ~$0-2 |
| **Fly.io** | $0 (free allowance) | ~$1.50 (volume) | $0 | ~$1-3 |

### Medium Traffic (10K requests/day, 3GB cache)

| Platform | Compute | Storage | Scheduler | Total/month |
|----------|---------|---------|-----------|-------------|
| **AWS Lambda + EFS** | ~$5-10 | ~$3.20 | $0 | ~$8-13 |
| **AWS Fargate + EFS** | ~$20-30 | ~$3.20 | $0 | ~$23-33 |
| **Google Cloud Run + GCS** | ~$10-15 | ~$0.10 | $0 | ~$10-15 |
| **Fly.io** | ~$15-25 | ~$1.50 | $0 | ~$17-27 |

---

## Deployment Decision Matrix

Choose based on your requirements:

| Priority | Recommendation | Reason |
|----------|---------------|---------|
| **Simplest setup** | Fly.io | Native volume support, simple config |
| **Lowest cost (low traffic)** | Google Cloud Run + GCS | Minimal storage costs, generous free tier |
| **AWS ecosystem** | Lambda + EFS | Full AWS integration |
| **Large dependencies** | Cloud Run or Fly.io | No Lambda size limits |
| **Fastest cold start** | Cloud Run or Fly.io | Container-optimized |
| **Enterprise/compliance** | AWS Fargate | Most control, VPC integration |

---

## Monitoring and Logging

### AWS
```bash
# CloudWatch logs
aws logs tail /aws/lambda/mechafil-server-ApiFunction --follow

# EFS metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/EFS \
  --metric-name BurstCreditBalance \
  --dimensions Name=FileSystemId,Value=$EFS_ID
```

### Google Cloud
```bash
# Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=mechafil-api" --limit 50

# View metrics
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_count"'
```

### Fly.io
```bash
# Logs
flyctl logs -a mechafil-api

# Metrics
flyctl status -a mechafil-api
flyctl metrics -a mechafil-api
```

---

## Troubleshooting

### Cold Start Issues
- **Problem**: API slow on first request after idle
- **Solution**: Increase `min_instances` to 1 (costs more) or optimize cache loading

### Cache Not Found
- **Problem**: API fails with "no cache data"
- **Solution**: Run cache updater manually first:
  ```bash
  # AWS
  aws lambda invoke --function-name CacheUpdaterFunction /dev/stdout

  # GCP
  gcloud scheduler jobs run mechafil-cache-updater-daily --location $REGION

  # Fly.io
  flyctl ssh console -a mechafil-cache-updater -C "python -m services.cache_updater.main --once"
  ```

### Storage Permission Issues
- **AWS**: Check IAM policies for `elasticfilesystem:Client*` permissions
- **GCP**: Ensure service account has `storage.objects.create` on GCS bucket
- **Fly.io**: Verify volume is mounted at correct path

### Schedule Not Running
- **AWS**: Check EventBridge rule is enabled and has correct permissions
- **GCP**: Verify Cloud Scheduler job is enabled and service account has `run.invoker` role
- **Fly.io**: Check machine is running (`flyctl machine list`)

---

## Next Steps

1. Choose your platform based on the decision matrix
2. Follow the deployment steps for your chosen platform
3. Run initial cache population
4. Test API endpoints
5. Set up monitoring and alerts
6. Configure CI/CD for automated deployments

For production deployments, also consider:
- Setting up custom domains
- Configuring CDN (CloudFront, Cloud CDN, etc.)
- Implementing authentication/API keys
- Setting up backup strategies for cache data
- Monitoring costs and optimizing resource allocation
