Deployment
==========

This guide covers deploying the MechaFil Server in production environments.

Quick Deployment
----------------

Basic Deployment with Uvicorn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to run the server in production:

.. code-block:: bash

   poetry run uvicorn mechafil_server.main:app \
     --host 0.0.0.0 \
     --port 8000 \
     --workers 4

Or use the provided command:

.. code-block:: bash

   poetry run mechafil-server

Production Server with Gunicorn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For better performance and process management:

.. code-block:: bash

   # Install Gunicorn
   poetry add gunicorn

   # Run with Gunicorn + Uvicorn workers
   gunicorn mechafil_server.main:app \
     --workers 4 \
     --worker-class uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:8000 \
     --timeout 120 \
     --access-logfile - \
     --error-logfile -

Docker Deployment
-----------------

Using Docker
~~~~~~~~~~~~

**Dockerfile**:

.. code-block:: dockerfile

   FROM python:3.11-slim

   # Set working directory
   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       && rm -rf /var/lib/apt/lists/*

   # Install Poetry
   RUN pip install poetry

   # Copy dependency files
   COPY pyproject.toml poetry.lock ./

   # Install dependencies
   RUN poetry config virtualenvs.create false \
       && poetry install --no-dev --no-interaction --no-ansi

   # Copy application code
   COPY mechafil_server ./mechafil_server

   # Create cache directory
   RUN mkdir -p .cache

   # Expose port
   EXPOSE 8000

   # Run the application
   CMD ["poetry", "run", "mechafil-server"]

**Build and run**:

.. code-block:: bash

   # Build image
   docker build -t mechafil-server:latest .

   # Run container
   docker run -d \
     --name mechafil-server \
     -p 8000:8000 \
     -e SPACESCOPE_TOKEN="Bearer YOUR_TOKEN" \
     -e CORS_ORIGINS="https://app.example.com" \
     -v $(pwd)/cache:/app/.cache \
     mechafil-server:latest

Using Docker Compose
~~~~~~~~~~~~~~~~~~~~

**docker-compose.yml**:

.. code-block:: yaml

   version: '3.8'

   services:
     mechafil-server:
       build: .
       container_name: mechafil-server
       ports:
         - "8000:8000"
       environment:
         - HOST=0.0.0.0
         - PORT=8000
         - LOG_LEVEL=INFO
         - SPACESCOPE_TOKEN=${SPACESCOPE_TOKEN}
         - CORS_ORIGINS=${CORS_ORIGINS:-*}
         - RELOAD_TRIGGER=02:00
       volumes:
         - ./cache:/app/.cache
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 40s

**Environment file (.env.production)**:

.. code-block:: bash

   SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE
   CORS_ORIGINS=https://app.example.com,https://dashboard.example.com

**Deploy**:

.. code-block:: bash

   # Load environment variables
   export $(cat .env.production | xargs)

   # Start services
   docker-compose up -d

   # View logs
   docker-compose logs -f

   # Stop services
   docker-compose down

Kubernetes Deployment
---------------------

Basic Deployment
~~~~~~~~~~~~~~~~

**deployment.yaml**:

.. code-block:: yaml

   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: mechafil-server
     labels:
       app: mechafil-server
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: mechafil-server
     template:
       metadata:
         labels:
           app: mechafil-server
       spec:
         containers:
         - name: mechafil-server
           image: mechafil-server:latest
           ports:
           - containerPort: 8000
           env:
           - name: HOST
             value: "0.0.0.0"
           - name: PORT
             value: "8000"
           - name: LOG_LEVEL
             value: "INFO"
           - name: SPACESCOPE_TOKEN
             valueFrom:
               secretKeyRef:
                 name: mechafil-secrets
                 key: spacescope-token
           - name: CORS_ORIGINS
             valueFrom:
               configMapKeyRef:
                 name: mechafil-config
                 key: cors-origins
           resources:
             requests:
               memory: "2Gi"
               cpu: "1000m"
             limits:
               memory: "4Gi"
               cpu: "2000m"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 10
             periodSeconds: 5
           volumeMounts:
           - name: cache
             mountPath: /app/.cache
         volumes:
         - name: cache
           emptyDir: {}

**service.yaml**:

.. code-block:: yaml

   apiVersion: v1
   kind: Service
   metadata:
     name: mechafil-server
   spec:
     selector:
       app: mechafil-server
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: LoadBalancer

**secrets.yaml**:

.. code-block:: yaml

   apiVersion: v1
   kind: Secret
   metadata:
     name: mechafil-secrets
   type: Opaque
   stringData:
     spacescope-token: "Bearer YOUR_TOKEN_HERE"

**configmap.yaml**:

.. code-block:: yaml

   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: mechafil-config
   data:
     cors-origins: "https://app.example.com,https://dashboard.example.com"

**Deploy to Kubernetes**:

.. code-block:: bash

   # Create secret (use base64 encoded value in production)
   kubectl apply -f secrets.yaml

   # Create configmap
   kubectl apply -f configmap.yaml

   # Deploy application
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml

   # Check status
   kubectl get pods -l app=mechafil-server
   kubectl get svc mechafil-server

   # View logs
   kubectl logs -l app=mechafil-server -f

With Persistent Cache
~~~~~~~~~~~~~~~~~~~~~

For persistent cache across pod restarts:

.. code-block:: yaml

   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: mechafil-cache-pvc
   spec:
     accessModes:
     - ReadWriteOnce
     resources:
       requests:
         storage: 10Gi

   ---
   # In deployment.yaml, replace emptyDir with:
   volumes:
   - name: cache
     persistentVolumeClaim:
       claimName: mechafil-cache-pvc

Reverse Proxy Setup
-------------------

Nginx
~~~~~

**nginx.conf**:

.. code-block:: nginx

   upstream mechafil_backend {
       server localhost:8000;
   }

   server {
       listen 80;
       server_name api.example.com;

       # Redirect HTTP to HTTPS
       return 301 https://$server_name$request_uri;
   }

   server {
       listen 443 ssl http2;
       server_name api.example.com;

       # SSL configuration
       ssl_certificate /etc/ssl/certs/api.example.com.crt;
       ssl_certificate_key /etc/ssl/private/api.example.com.key;
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers HIGH:!aNULL:!MD5;

       # Logging
       access_log /var/log/nginx/mechafil-access.log;
       error_log /var/log/nginx/mechafil-error.log;

       # Proxy settings
       location / {
           proxy_pass http://mechafil_backend;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;

           # Timeouts for long-running simulations
           proxy_connect_timeout 120s;
           proxy_send_timeout 120s;
           proxy_read_timeout 120s;
       }

       # Health check endpoint (no auth)
       location /health {
           proxy_pass http://mechafil_backend;
           access_log off;
       }
   }

Caddy
~~~~~

**Caddyfile**:

.. code-block:: text

   api.example.com {
       reverse_proxy localhost:8000 {
           # Timeouts for long-running requests
           transport http {
               dial_timeout 30s
               response_header_timeout 120s
           }
       }

       # Logging
       log {
           output file /var/log/caddy/mechafil.log
       }
   }

Traefik (Docker)
~~~~~~~~~~~~~~~~

**docker-compose.yml with Traefik**:

.. code-block:: yaml

   version: '3.8'

   services:
     traefik:
       image: traefik:v2.9
       command:
         - "--providers.docker=true"
         - "--entrypoints.web.address=:80"
         - "--entrypoints.websecure.address=:443"
         - "--certificatesresolvers.letsencrypt.acme.tlschallenge=true"
         - "--certificatesresolvers.letsencrypt.acme.email=admin@example.com"
         - "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json"
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - "/var/run/docker.sock:/var/run/docker.sock:ro"
         - "./letsencrypt:/letsencrypt"

     mechafil-server:
       build: .
       labels:
         - "traefik.enable=true"
         - "traefik.http.routers.mechafil.rule=Host(`api.example.com`)"
         - "traefik.http.routers.mechafil.entrypoints=websecure"
         - "traefik.http.routers.mechafil.tls.certresolver=letsencrypt"
         - "traefik.http.services.mechafil.loadbalancer.server.port=8000"
       environment:
         - SPACESCOPE_TOKEN=${SPACESCOPE_TOKEN}
       volumes:
         - ./cache:/app/.cache

Cloud Deployment
----------------

AWS (EC2 + ECS)
~~~~~~~~~~~~~~~

**1. Create ECR repository**:

.. code-block:: bash

   aws ecr create-repository --repository-name mechafil-server

**2. Build and push Docker image**:

.. code-block:: bash

   # Authenticate
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

   # Build
   docker build -t mechafil-server .

   # Tag
   docker tag mechafil-server:latest \
     <account-id>.dkr.ecr.us-east-1.amazonaws.com/mechafil-server:latest

   # Push
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/mechafil-server:latest

**3. Create ECS Task Definition** (task-definition.json):

.. code-block:: json

   {
     "family": "mechafil-server",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "2048",
     "memory": "4096",
     "containerDefinitions": [
       {
         "name": "mechafil-server",
         "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/mechafil-server:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "HOST",
             "value": "0.0.0.0"
           },
           {
             "name": "PORT",
             "value": "8000"
           }
         ],
         "secrets": [
           {
             "name": "SPACESCOPE_TOKEN",
             "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:spacescope-token"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/mechafil-server",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }

**4. Deploy**:

.. code-block:: bash

   # Register task definition
   aws ecs register-task-definition --cli-input-json file://task-definition.json

   # Create service
   aws ecs create-service \
     --cluster my-cluster \
     --service-name mechafil-server \
     --task-definition mechafil-server \
     --desired-count 2 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"

Google Cloud (Cloud Run)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build and push to Google Container Registry
   gcloud builds submit --tag gcr.io/PROJECT_ID/mechafil-server

   # Deploy to Cloud Run
   gcloud run deploy mechafil-server \
     --image gcr.io/PROJECT_ID/mechafil-server \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars SPACESCOPE_TOKEN="Bearer YOUR_TOKEN" \
     --memory 4Gi \
     --cpu 2 \
     --timeout 120

Azure (Container Instances)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create resource group
   az group create --name mechafil-rg --location eastus

   # Create container registry
   az acr create --resource-group mechafil-rg --name mechafilacr --sku Basic

   # Build and push
   az acr build --registry mechafilacr --image mechafil-server:latest .

   # Deploy
   az container create \
     --resource-group mechafil-rg \
     --name mechafil-server \
     --image mechafilacr.azurecr.io/mechafil-server:latest \
     --cpu 2 \
     --memory 4 \
     --registry-login-server mechafilacr.azurecr.io \
     --registry-username <username> \
     --registry-password <password> \
     --ports 8000 \
     --environment-variables \
       SPACESCOPE_TOKEN="Bearer YOUR_TOKEN" \
       CORS_ORIGINS="https://app.example.com"

Monitoring and Observability
-----------------------------

Prometheus Metrics
~~~~~~~~~~~~~~~~~~

Add prometheus metrics to monitor the server:

.. code-block:: bash

   # Install prometheus client
   poetry add prometheus-fastapi-instrumentator

Add to ``main.py``:

.. code-block:: python

   from prometheus_fastapi_instrumentator import Instrumentator

   # After creating the FastAPI app
   Instrumentator().instrument(app).expose(app)

Access metrics at ``/metrics``

Health Monitoring
~~~~~~~~~~~~~~~~~

Set up automated health checks:

.. code-block:: bash

   # Using curl in a cron job
   */5 * * * * curl -f http://localhost:8000/health || echo "Server is down!"

   # Using a monitoring service
   # Configure your service to check http://api.example.com/health

Logging
~~~~~~~

Centralized logging with ELK stack or similar:

.. code-block:: python

   # Configure JSON logging for better parsing
   import logging
   from pythonjsonlogger import jsonlogger

   logHandler = logging.StreamHandler()
   formatter = jsonlogger.JsonFormatter()
   logHandler.setFormatter(formatter)
   logging.root.addHandler(logHandler)

Performance Tuning
------------------

Worker Configuration
~~~~~~~~~~~~~~~~~~~~

Recommended worker count:

.. code-block:: python

   # Formula: (2 * CPU cores) + 1
   # For 4 CPU cores:
   workers = 9

Resource Limits
~~~~~~~~~~~~~~~

Set appropriate limits based on usage:

.. code-block:: yaml

   # Kubernetes example
   resources:
     requests:
       memory: "2Gi"
       cpu: "1000m"
     limits:
       memory: "4Gi"
       cpu: "2000m"

JAX Configuration
~~~~~~~~~~~~~~~~~

For GPU acceleration:

.. code-block:: dockerfile

   # Use CUDA base image
   FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

   # Install JAX with CUDA support
   RUN pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Scaling Strategies
------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~~

The server is stateless (except for cache) and can be scaled horizontally:

1. Use a shared cache backend (Redis, Memcached)
2. Deploy multiple instances behind a load balancer
3. Use Kubernetes HPA (Horizontal Pod Autoscaler)

**Kubernetes HPA example**:

.. code-block:: yaml

   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: mechafil-server-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: mechafil-server
     minReplicas: 2
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70

Backup and Disaster Recovery
-----------------------------

Cache Backup
~~~~~~~~~~~~

Backup the cache directory regularly:

.. code-block:: bash

   # Backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d)
   tar -czf mechafil-cache-$DATE.tar.gz .cache/
   aws s3 cp mechafil-cache-$DATE.tar.gz s3://backups/mechafil/

Database-backed Cache
~~~~~~~~~~~~~~~~~~~~~

For production, consider using a database-backed cache:

.. code-block:: python

   # Use SQLite or PostgreSQL instead of DiskCache
   # Example with SQLAlchemy
   from sqlalchemy import create_engine
   from sqlalchemy.orm import sessionmaker

   engine = create_engine('postgresql://user:pass@localhost/mechafil')
   Session = sessionmaker(bind=engine)

Next Steps
----------

* Review :doc:`configuration` for environment setup
* Check :doc:`api/endpoints` for API documentation
* See :doc:`examples/advanced` for integration patterns
