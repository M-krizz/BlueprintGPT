# Standalone Application & Deployment Guide

## Table of Contents

1. [Application Overview](#application-overview)
2. [Deployment Architecture](#deployment-architecture)
3. [Standalone Setup](#standalone-setup)
4. [Production Deployment](#production-deployment)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Application Overview

### What is BlueprintGPT?

```
BlueprintGPT is a standalone AI-powered floor plan generation system
that allows users to:

1. Describe residential floor plans in natural language
2. Generate multiple optimized layout variants
3. Iteratively refine designs through conversation
4. Export professional floor plans (SVG, DWG, PDF)
5. Get compliance validation and explanations
```

### System Components

```
┌─────────────────────────────────────────────────┐
│          Standalone Application                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │  Frontend (React + TypeScript)          │   │
│  │  - Web UI (localhost:3000)              │   │
│  │  - Interactive chat interface           │   │
│  │  - Floor plan visualization (SVG)       │   │
│  │  - Design comparison tools              │   │
│  └─────────────────────────────────────────┘   │
│                    ↓ HTTP/WebSocket             │
│  ┌─────────────────────────────────────────┐   │
│  │  Backend API (FastAPI + Python)         │   │
│  │  - FastAPI server (localhost:8000)      │   │
│  │  - NLP interface (Gemini integration)   │   │
│  │  - Generation backends (4 variants)     │   │
│  │  - Constraint engine                    │   │
│  │  - Session management                   │   │
│  └─────────────────────────────────────────┘   │
│                    ↓                             │
│  ┌─────────────────────────────────────────┐   │
│  │  Data & Models                          │   │
│  │  - Ontology (building codes)            │   │
│  │  - ML checkpoints (LayoutTransformer)   │   │
│  │  - Regulation database                  │   │
│  │  - Session storage (JSON)               │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Features

#### Core Features
- ✅ Multi-backend generation (algorithmic, learned, planner, hybrid)
- ✅ Natural language interface (Gemini-powered)
- ✅ Multi-turn conversations with state management
- ✅ Real-time design ranking and selection
- ✅ Compliance validation with repair loop
- ✅ Professional SVG output with CAD-style rendering

#### Advanced Features
- ✅ Design explanation generation
- ✅ Comparative analysis (show differences)
- ✅ Batch processing (multiple specs)
- ✅ API for external integration
- ✅ Session persistence and recovery
- ✅ Logging and diagnostics

---

## Deployment Architecture

### Development Mode

```
┌──────────────────────────────────────────────────┐
│          Development Machine                     │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ Frontend Dev Server (Vite)                 │ │
│  │ localhost:3000                             │ │
│  │ Hot reload, source maps                    │ │
│  └────────────────────────────────────────────┘ │
│                    ↓                             │
│  ┌────────────────────────────────────────────┐ │
│  │ Backend Dev Server (FastAPI)               │ │
│  │ localhost:8000                             │ │
│  │ Auto-reload, debug mode                    │ │
│  └────────────────────────────────────────────┘ │
│                    ↓                             │
│  ┌────────────────────────────────────────────┐ │
│  │ Local Data                                 │ │
│  │ - SQLite (dev database)                    │ │
│  │ - Models in memory                         │ │
│  │ - Outputs to ./outputs/                    │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Production Mode (Single Server)

```
┌────────────────────────────────────────────────────┐
│          Production Server                        │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ Nginx (Reverse Proxy)                        │ │
│  │ - Port 80 (HTTP) + 443 (HTTPS)              │ │
│  │ - Static file serving                        │ │
│  │ - Load balancing                             │ │
│  └──────────────────────────────────────────────┘ │
│                    ↓                              │
│  ┌──────────────────────────────────────────────┐ │
│  │ FastAPI Server (Gunicorn)                    │ │
│  │ - Internal port 8000                         │ │
│  │ - 4 worker processes                         │ │
│  │ - Graceful shutdown                          │ │
│  └──────────────────────────────────────────────┘ │
│                    ↓                              │
│  ┌──────────────────────────────────────────────┐ │
│  │ Data & Cache                                 │ │
│  │ - PostgreSQL (sessions)                      │ │
│  │ - Redis (response cache)                     │ │
│  │ - GPU memory (loaded models)                 │ │
│  │ - /var/blueprintgpt/ (outputs)              │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Production Mode (Distributed/Kubernetes)

```
┌─────────────────────────────────────────────────────┐
│        Kubernetes Cluster (Production)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Load Balancer (Cloud: AWS ALB, GCP LB)    │   │
│  │  SSL termination                            │   │
│  └─────────────────────────────────────────────┘   │
│                    ↓                               │
│  ┌─────────────────────────────────────────────┐   │
│  │  API Service (Multiple Pods)                │   │
│  │  ├─ Pod 1: FastAPI + worker                │   │
│  │  ├─ Pod 2: FastAPI + worker                │   │
│  │  ├─ Pod 3: FastAPI + worker                │   │
│  │  └─ Pod N: FastAPI + worker                │   │
│  │  HPA: Auto-scale 2-10 replicas            │   │
│  └─────────────────────────────────────────────┘   │
│          ↓              ↓             ↓             │
│  ┌──────────────────────────────────────────────┐  │
│  │  Persistent Layer                            │  │
│  │  ├─ PostgreSQL (managed)                    │  │
│  │  ├─ Redis (managed)                         │  │
│  │  ├─ S3 (artifact storage)                   │  │
│  │  └─ GPU nodes (model inference)             │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Standalone Setup

### Prerequisites

```
System Requirements:
  - OS: Linux (recommended), macOS, or Windows (WSL2)
  - CPU: 4+ cores
  - RAM: 8GB minimum (16GB recommended)
  - GPU: Optional (speeds up ML inference 5-10x)
  - Disk: 20GB free (includes models)
  - Network: Internet connection (for Gemini API calls)

Software Requirements:
  - Python 3.12+
  - Node.js 18+ (for frontend)
  - pip or conda (Python package manager)
  - npm or yarn (Node package manager)
  - Git
```

### Installation Step-by-Step

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/M-krizz/BlueprintGPT.git
cd BlueprintGPT

# Verify structure
ls -la
# → README.md, requirements.txt, api/, frontend/, etc.
```

#### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# Verify activation
python --version  # Should show 3.12.x
```

#### Step 3: Install Python Dependencies

```bash
# Install backend dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify key packages
python -c "import fastapi, shapely, torch; print('✓ Core dependencies OK')"
```

#### Step 4: Set Up Environment Variables

```bash
# Create .env file from template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your editor

# Required variables:
GEMINI_API_KEY=<your-api-key-here>
GEMINI_MODEL=gemini-2.5-flash
BLUEPRINT_BACKEND_MODE=hybrid

# Optional optimizations:
BLUEPRINT_AUTO_CORE_BACKEND=algorithmic
BLUEPRINTGPT_CHECKPOINT=learned/model/checkpoints/improved_v1.pt
```

#### Step 5: Test Backend

```bash
# Start backend server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, test API
curl http://localhost:8000/
# Should return: HTML redirect or welcome message

# Test specific endpoint
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "backend_target": "algorithmic",
    "boundary": {"width": 12, "height": 15},
    "rooms": [
      {"name": "Bedroom", "type": "Bedroom"}
    ]
  }'
# Should return: generation result
```

#### Step 6: Set Up Frontend (Optional for Web UI)

```bash
# Install frontend dependencies
cd frontend
npm install

# Start dev server
npm run dev
# Opens at http://localhost:5173

# Build for production
npm run build
# Creates ./dist/ folder
```

#### Step 7: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/unit/test_polygon_packer.py -v

# Generate coverage report
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

### Quick Start Commands

```bash
# One-liner setup (Linux/macOS)
git clone https://github.com/M-krizz/BlueprintGPT.git && \
cd BlueprintGPT && \
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
cp .env.example .env && \
echo "✓ Setup complete. Edit .env and run: python -m uvicorn api.server:app --reload"

# On Windows PowerShell
git clone https://github.com/M-krizz/BlueprintGPT.git; `
cd BlueprintGPT; `
python -m venv venv; `
venv\Scripts\Activate.ps1; `
pip install -r requirements.txt; `
cp .env.example .env; `
Write-Host "✓ Setup complete. Edit .env and run: python -m uvicorn api.server:app --reload"
```

---

## Production Deployment

### Docker Deployment

#### Step 1: Build Docker Image

```bash
# Build image
docker build -t blueprintgpt:latest .

# Verify image
docker images | grep blueprintgpt
# → Should see image with size ~2.5GB
```

#### Step 2: Run Container

```bash
# Start container with environment variables
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY=<your-key> \
  -e BLUEPRINT_BACKEND_MODE=hybrid \
  -v /var/blueprintgpt:/app/outputs \
  --name blueprintgpt \
  blueprintgpt:latest

# Check logs
docker logs blueprintgpt

# Test endpoint
curl http://localhost:8000/docs
# Should open Swagger UI
```

#### Step 3: Docker Compose (Multi-container)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - BLUEPRINT_BACKEND_MODE=hybrid
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@db:5432/blueprintgpt
    depends_on:
      - redis
      - db
    volumes:
      - ./outputs:/app/outputs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=blueprintgpt
      - POSTGRES_PASSWORD=secure_password
      - POSTGRES_DB=blueprintgpt
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

### Kubernetes Deployment

#### Step 1: Create Deployment Manifest

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: blueprintgpt-api
  labels:
    app: blueprintgpt

spec:
  replicas: 3
  selector:
    matchLabels:
      app: blueprintgpt
      
  template:
    metadata:
      labels:
        app: blueprintgpt
    spec:
      containers:
      - name: api
        image: blueprintgpt:latest
        imagePullPolicy: Always
        
        ports:
        - containerPort: 8000
        
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: blueprintgpt-secrets
              key: gemini-api-key
        - name: BLUEPRINT_BACKEND_MODE
          value: "hybrid"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: blueprintgpt-service
spec:
  type: LoadBalancer
  selector:
    app: blueprintgpt
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: blueprintgpt-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: blueprintgpt-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Step 2: Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace blueprintgpt

# Create secret
kubectl create secret generic blueprintgpt-secrets \
  --from-literal=gemini-api-key=<your-key> \
  -n blueprintgpt

# Apply deployment
kubectl apply -f kubernetes/deployment.yaml -n blueprintgpt

# Verify deployment
kubectl get pods -n blueprintgpt
kubectl get services -n blueprintgpt

# Scale replicas
kubectl scale deployment blueprintgpt-api --replicas=5 -n blueprintgpt

# Monitor
kubectl logs -f deployment/blueprintgpt-api -n blueprintgpt
```

---

## Performance Tuning

### Backend Optimization

#### Model Loading

```python
# Current: Load models on every request (slow)
model = load_model(checkpoint_path)

# Optimized: Load once at startup
class ModelCache:
    _cache = {}
    
    @classmethod
    def get_model(cls, checkpoint_path):
        if checkpoint_path not in cls._cache:
            cls._cache[checkpoint_path] = load_model(checkpoint_path)
        return cls._cache[checkpoint_path]

# Use: model = ModelCache.get_model(path)
# Benefit: 10x faster (100ms → 10ms per request)
```

#### Response Caching

```python
# Cache identical requests
cache = {}

def generate_cached(spec):
    spec_hash = hash_spec(spec)
    if spec_hash in cache:
        return cache[spec_hash]
    
    result = generate(spec)
    cache[spec_hash] = result
    
    # Cleanup old entries (LRU)
    if len(cache) > 100:
        remove_oldest()
    
    return result

# Benefit: Repeated requests 10x faster
```

### Database Optimization

#### Connection Pooling

```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True  # Verify connections
)
```

#### Index Strategies

```sql
-- Index for session queries
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_created_at ON sessions(created_at);

-- Index for design queries
CREATE INDEX idx_designs_session_id ON designs(session_id);
CREATE INDEX idx_designs_backend ON designs(backend_target);

-- Composite index for common queries
CREATE INDEX idx_designs_session_backend ON designs(session_id, backend_target);
```

### Frontend Optimization

#### Code Splitting

```javascript
// Lazy load heavy components
const DesignViewer = React.lazy(() => import('./DesignViewer'));
const Comparison = React.lazy(() => import('./Comparison'));

function App() {
  return (
    <Suspense fallback={<Spinner />}>
      <DesignViewer />
    </Suspense>
  );
}

// Benefit: Initial bundle 30% smaller
```

#### Image Optimization

```javascript
// Compress and optimize SVG
import { compressSvg } from 'svgo';

const optimized = compressSvg(svgString);
// Benefit: 40% size reduction, still scalable
```

### System Configuration

#### Linux Tuning

```bash
# Increase file descriptors
ulimit -n 65536

# Network tuning for FastAPI
sysctl -w net.core.backlog=32768
sysctl -w net.ipv4.tcp_max_syn_backlog=32768

# Memory tuning
echo vm.swappiness=10 >> /etc/sysctl.conf
```

---

## Monitoring & Maintenance

### Health Checks

#### API Endpoints

```python
@app.get("/health")
async def health_check():
    """Quick health check (used by load balancers)"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "uptime": get_uptime()
    }

@app.get("/ready")
async def readiness_check():
    """Detailed readiness check (all components)"""
    checks = {
        "database": check_database(),
        "redis": check_redis(),
        "models_loaded": check_models(),
        "disk_space": check_disk_space(),
        "gemini_api": check_gemini_api()
    }
    
    all_ready = all(checks.values())
    return {
        "ready": all_ready,
        "checks": checks
    }

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    return {
        "requests_total": counter_requests,
        "requests_duration_seconds": histogram_duration,
        "models_loaded": len(ModelCache._cache),
        "cache_size": len(response_cache),
        "memory_mb": psutil.Process().memory_info().rss // 1024 // 1024
    }
```

### Logging

#### Structured Logging

```python
import logging
from pythonjsonlogger import jsonlogger

# Setup JSON logging for ELK stack
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Usage
logger.info("layout_generated", extra={
    "session_id": session_id,
    "backend": backend,
    "score": score,
    "time_ms": duration_ms
})

# Output (machine-readable)
{"message": "layout_generated", "session_id": "...", "backend": "hybrid", ...}
```

### Monitoring Tools

#### Prometheus + Grafana

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'blueprintgpt'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

```
Dashboard Metrics:
  - API response time (p50, p95, p99)
  - Request rate (req/sec)
  - Error rate (%)
  - Model inference time
  - Cache hit rate (%)
  - Database queries/sec
  - Memory usage (MB)
  - GPU utilization (%)
```

### Backup Strategy

#### Session Backup

```bash
# Daily backup to S3
aws s3 sync /var/blueprintgpt/sessions \
  s3://blueprintgpt-backups/sessions/$(date +%Y-%m-%d)/ \
  --delete

# Retention policy: 30 days
aws s3 ls s3://blueprintgpt-backups/sessions/ --recursive | \
  awk '{print $1}' | sort -r | tail -n +31 | \
  xargs -I {} aws s3 rm s3://blueprintgpt-backups/sessions/{}
```

#### Database Backup

```bash
# Daily PostgreSQL dump
pg_dump blueprintgpt | gzip > /backups/db_$(date +%Y%m%d).sql.gz

# Keep 30 days of backups
find /backups -name 'db_*.sql.gz' -mtime +30 -delete

# Upload to S3
aws s3 cp /backups/db_$(date +%Y%m%d).sql.gz \
  s3://blueprintgpt-backups/database/
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "CUDA out of memory"

```
Symptom: Generation fails with CUDA out of memory error

Solution 1 (Immediate):
  - Switch to CPU: BLUEPRINT_BACKEND_MODE=algorithmic
  - Reduce K (number of candidates): k=5 instead of k=10

Solution 2 (Permanent):
  - Quantize model (float32 → int8)
  - Use smaller model variant
  - Upgrade GPU memory

Code:
  model = torch.quantization.quantize_dynamic(
      model, {torch.nn.Linear}, dtype=torch.qint8
  )
```

#### Issue 2: "API responding slowly (>2s)"

```
Diagnosis:
  1. Check model loading time
  2. Check generation backend latency
  3. Check database query time
  
  logs show: backend takes 500ms usually, now 800ms
  → Check GPU temperature (thermal throttling?)
  → Check memory (swap usage?)

Solution:
  - Restart server (clear caches): systemctl restart blueprintgpt
  - Reduce concurrent requests (load balancer)
  - Add more workers (Gunicorn workers)
```

#### Issue 3: "Compliance violations increasing"

```
Symptom: Repair loop not fixing violations

Diagnosis:
  - Room program too complex
  - Constraints conflicting (unsolvable)
  - Regulatory rule changed
  
Solution:
  1. Check violation type (logs)
  2. If regulatory: Update ontology
  3. If complex program: Request spec simplification
  4. Disable specific constraint temporarily (debugging)
```

---

## Maintenance Checklist

```markdown
# Weekly Tasks
- [ ] Check error logs for patterns
- [ ] Monitor response time trends
- [ ] Verify backups completed
- [ ] Check disk space usage
- [ ] Review user feedback

# Monthly Tasks
- [ ] Update dependencies (security patches)
- [ ] Analyze performance metrics
- [ ] Test disaster recovery
- [ ] Review and update documentation
- [ ] Database maintenance (VACUUM ANALYZE)

# Quarterly Tasks
- [ ] Security audit
- [ ] Load testing
- [ ] Capacity planning
- [ ] Model retraining evaluation
- [ ] Feature roadmap review

# Annually Tasks
- [ ] Major version upgrade testing
- [ ] Architecture review
- [ ] Regulatory compliance audit
- [ ] Team training and certification
- [ ] Disaster recovery drill
```

---

**Version**: 1.0  
**Date**: March 27, 2026  
**Status**: Production Ready
