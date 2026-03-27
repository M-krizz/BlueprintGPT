# API Reference & Integration Guide

## Table of Contents

1. [API Overview](#api-overview)
2. [Core Endpoints](#core-endpoints)
3. [Data Models](#data-models)
4. [Error Handling](#error-handling)
5. [Authentication](#authentication)
6. [Integration Examples](#integration-examples)
7. [Advanced Features](#advanced-features)

---

## API Overview

### Base Information

```
API Base URL:         http://localhost:8000
Swagger UI:           http://localhost:8000/docs
ReDoc:                http://localhost:8000/redoc
API Version:          v1.0
Authentication:       API Key (Optional header)
Response Format:      JSON
```

### API Features

```
✅ RESTful architecture with JSON payloads
✅ WebSocket support for real-time feedback
✅ Async/await for high concurrency
✅ Request validation with Pydantic
✅ CORS enabled for cross-domain requests
✅ Gzip compression for large responses
✅ Rate limiting (100 req/min per IP)
```

---

## Core Endpoints

### 1. Generate Layout `/api/generate`

#### Description
Generate a floor plan layout from specifications.

#### Request

```http
POST /api/generate HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "boundary": {
    "width": 12.0,
    "height": 15.0,
    "type": "rectangle"
  },
  "rooms": [
    {
      "name": "Master Bedroom",
      "type": "Bedroom",
      "area_m2": 150,
      "min_width_m": 4.0,
      "aspect_ratio": [0.8, 1.5]
    },
    {
      "name": "Kitchen",
      "type": "Kitchen",
      "area_m2": 100
    }
  ],
  "backend_target": "hybrid",
  "num_variants": 3,
  "constraints": {
    "max_travel_distance": 22.5,
    "required_adjacencies": [
      ["Kitchen", "DiningRoom"]
    ],
    "forbidden_adjacencies": []
  },
  "session_id": null
}
```

#### Response (Success)

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "layout_id": "layout_20240327_001",
  "session_id": "sess_abc123def456",
  "timestamp": "2024-03-27T14:32:10.123Z",
  "backend_used": "hybrid",
  "variants": [
    {
      "variant_id": "var_001",
      "design_score": 0.832,
      "ranking": 1,
      "rooms": [
        {
          "name": "Master Bedroom",
          "type": "Bedroom",
          "bounds": {
            "x": 0.0,
            "y": 0.0,
            "width": 5.0,
            "height": 6.0
          },
          "area_m2": 150.0
        }
      ],
      "doors": [
        {
          "id": "door_001",
          "from_room": "Master Bedroom",
          "to_room": "Corridor",
          "position": [5.0, 3.0]
        }
      ],
      "svg_url": "/artifacts/layout_20240327_001/variant_001.svg",
      "compliance_status": "PASS",
      "metrics": {
        "area_utilization": 0.935,
        "design_diversity": 0.87,
        "connectivity_score": 0.95
      }
    }
  ],
  "compliance": {
    "status": "PASS",
    "violations": [],
    "regulations_checked": 12
  },
  "metadata": {
    "generation_time_ms": 385,
    "algorithm_iterations": 47,
    "repair_attempts": 2
  }
}
```

#### Response (Error)

```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/json

{
  "error": "validation_error",
  "message": "Invalid request data",
  "details": [
    {
      "field": "rooms[0].area_m2",
      "error": "Area exceeds available plot area"
    }
  ]
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `boundary` | Object | Yes | Plot dimensions and shape |
| `boundary.width` | Float | Yes | Width in meters (min: 5, max: 50) |
| `boundary.height` | Float | Yes | Height in meters (min: 5, max: 50) |
| `boundary.type` | String | No | Shape: "rectangle" (default), "L_shape" |
| `rooms` | Array | Yes | List of rooms to layout |
| `rooms[i].name` | String | Yes | Room identifier |
| `rooms[i].type` | String | Yes | Room type from enum |
| `rooms[i].area_m2` | Float | Yes | Target area in m² |
| `backend_target` | String | No | Generation backend: "algorithmic", "learned", "planner", "hybrid" (default) |
| `num_variants` | Integer | No | Number of variants to generate (1-10, default: 3) |
| `session_id` | String | No | Existing session ID for context |

#### Room Types

```
Bedroom, MasterBedroom, Kitchen, Bathroom, LivingRoom, 
DiningRoom, DrawingRoom, Corridor, Garage, Store, Hall, 
StudyRoom, UtilityRoom
```

#### Backend Comparison

| Backend | Speed | Quality | Compliance | Use Case |
|---------|-------|---------|-----------|----------|
| **algorithmic** | 150-250ms | 0.78 | 96% | Quick iterations |
| **learned** | 300-500ms | 0.83 | 85% | Aesthetic designs |
| **planner** | 250-400ms | 0.74 | 91% | Structured layouts |
| **hybrid** | 350-550ms | 0.83 | 98% | Best results |

---

### 2. Chat Interface `/api/chat`

#### Description
Natural language interface with multi-turn conversation support.

#### Request

```http
POST /api/chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "message": "I want a 3BHK apartment with an open kitchen-living area. Budget around 1500 sq ft.",
  "session_id": null,
  "mode": "design",
  "preferences": {
    "style": "modern",
    "lighting": "natural",
    "ventilation": "cross"
  }
}
```

#### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "session_id": "sess_xyz789abc",
  "message_id": "msg_001",
  "intent": "DESIGN",
  "confidence": 0.94,
  "extracted_spec": {
    "rooms": [
      {
        "name": "Master Bedroom",
        "type": "Bedroom",
        "area_m2": 120
      },
      {
        "name": "Bedroom 2",
        "type": "Bedroom",
        "area_m2": 100
      },
      {
        "name": "Kitchen-Living",
        "type": "LivingRoom",
        "area_m2": 280
      }
    ],
    "boundary": {
      "width": 16,
      "height": 14,
      "area_sqft": 1500
    }
  },
  "layout": {
    "layout_id": "layout_chat_20240327_001",
    "svg_url": "/artifacts/layout_chat_20240327_001/design.svg",
    "design_score": 0.815,
    "compliance_status": "PASS"
  },
  "assistant_response": "I've created a modern 3BHK layout with an open kitchen-living area as requested. The design includes cross-ventilation for natural airflow. Total area is 1450 sq ft, fitting your budget.",
  "next_actions": [
    "generate_alternatives",
    "refine_design",
    "export_layout"
  ]
}
```

#### Intent Types

```python
DESIGN       # New design request
CORRECTION   # Modify existing design
QUESTION     # Ask about current design
CONVERSATION # General chat
FEEDBACK     # User feedback
```

#### Intent Classification Examples

```
User Message                          → Intent
─────────────────────────────────────────────────
"3BHK, 1200 sqft"                      → DESIGN
"Make the bedroom bigger"              → CORRECTION
"Does this pass building codes?"       → QUESTION
"That looks great!"                    → FEEDBACK
"Tell me about this layout"            → CONVERSATION
```

---

### 3. Correction/Refinement `/api/correct`

#### Description
Refine existing design based on feedback.

#### Request

```http
POST /api/correct HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "layout_id": "layout_20240327_001",
  "session_id": "sess_abc123",
  "corrections": [
    {
      "type": "area_adjustment",
      "room": "Master Bedroom",
      "adjustment": "+20%"
    },
    {
      "type": "adjacency_change",
      "room1": "Kitchen",
      "room2": "DiningRoom",
      "action": "ensure_adjacent"
    },
    {
      "type": "constraint_add",
      "constraint": "bedroom_corner_windows"
    }
  ],
  "preservation": {
    "preserve_layout_structure": true,
    "preserve_adjacencies": ["Kitchen-Dining"]
  }
}
```

#### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "layout_id": "layout_20240327_002",
  "original_layout_id": "layout_20240327_001",
  "correction_summary": {
    "changes_applied": 3,
    "changes_failed": 0,
    "score_delta": 0.023
  },
  "new_design": {
    "svg_url": "/artifacts/layout_20240327_002/design.svg",
    "design_score": 0.855,
    "changes": [
      {
        "room": "Master Bedroom",
        "old_area": 150.0,
        "new_area": 180.0
      }
    ]
  }
}
```

---

### 4. Explanation `/api/explain`

#### Description
Get AI-generated explanation of layout design.

#### Request

```http
POST /api/explain HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "layout_id": "layout_20240327_001",
  "detail_level": "comprehensive",
  "include_metrics": true,
  "include_recommendations": true
}
```

#### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "layout_id": "layout_20240327_001",
  "explanation": {
    "overview": "This is a well-balanced 3BHK layout with excellent space utilization at 93.5%. The design follows the principle of functional zoning with sleeping areas separated from active zones.",
    
    "design_highlights": [
      "Master bedroom positioned for morning sunlight",
      "Kitchen-Living area creates social flow",
      "Cross-ventilation throughout the unit",
      "All bedrooms have windows/ventilation",
      "Efficient corridor layout minimizes wasted space"
    ],
    
    "compliance_analysis": {
      "status": "PASS",
      "regulations_met": 12,
      "violations": [],
      "details": {
        "fire_safety": "✓ Emergency exit provided",
        "accessibility": "✓ Door widths meet standards",
        "ventilation": "✓ All rooms have windows",
        "travel_distance": "✓ Max 22.5m achieved: 18.3m"
      }
    },
    
    "metrics_breakdown": {
      "design_score": {
        "value": 0.815,
        "components": {
          "geometry": 0.82,
          "adjacency": 0.88,
          "functionality": 0.79
        }
      },
      "area_utilization": "93.5%",
      "connectivity": "4/4 rooms connected"
    },
    
    "recommendations": [
      "Consider adding a window to the corridor for natural light",
      "Kitchen island could improve work triangle efficiency"
    ]
  }
}
```

---

### 5. Session Management `/api/sessions/{session_id}`

#### Get Session

```http
GET /api/sessions/sess_abc123 HTTP/1.1
Host: localhost:8000

---

HTTP/1.1 200 OK
Content-Type: application/json

{
  "session_id": "sess_abc123",
  "created_at": "2024-03-27T10:00:00Z",
  "last_updated": "2024-03-27T14:32:10Z",
  "status": "active",
  "message_count": 5,
  "designs_generated": 12,
  
  "messages": [
    {
      "message_id": "msg_001",
      "timestamp": "2024-03-27T10:00:00Z",
      "role": "user",
      "content": "3BHK apartment"
    },
    {
      "message_id": "msg_002",
      "timestamp": "2024-03-27T10:00:05Z",
      "role": "assistant",
      "content": "Generating 3BHK layout..."
    }
  ],
  
  "designs": [
    {
      "layout_id": "layout_001",
      "timestamp": "2024-03-27T10:00:10Z",
      "design_score": 0.815,
      "backend_used": "hybrid",
      "status": "active"
    }
  ]
}
```

#### Delete Session

```http
DELETE /api/sessions/sess_abc123 HTTP/1.1
Host: localhost:8000

---

HTTP/1.1 200 OK
Content-Type: application/json

{
  "message": "Session deleted successfully",
  "session_id": "sess_abc123"
}
```

---

### 6. Export `/api/export`

#### Export as SVG

```http
POST /api/export HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "layout_id": "layout_20240327_001",
  "format": "svg",
  "style": "technical",
  "include_dimensions": true,
  "include_labels": true
}

---

HTTP/1.1 200 OK
Content-Type: image/svg+xml

<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 200 250" xmlns="http://www.w3.org/2000/svg">
  <!-- Floor plan SVG content -->
</svg>
```

#### Export as DWG (Requires plugin)

```http
POST /api/export HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "layout_id": "layout_20240327_001",
  "format": "dwg",
  "scale": "1:100",
  "include_dimensions": true
}

---

HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="layout.dwg"

[Binary DWG file content]
```

#### Export as PDF

```http
POST /api/export HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "layout_id": "layout_20240327_001",
  "format": "pdf",
  "include_compliance_report": true,
  "include_explanation": true
}

---

HTTP/1.1 200 OK
Content-Type: application/pdf
Content-Disposition: attachment; filename="layout.pdf"

[PDF file content]
```

---

## Data Models

### Room Model

```python
class Room(BaseModel):
    name: str                              # e.g., "Master Bedroom"
    type: RoomType                         # Enum: Bedroom, Kitchen, etc.
    area_m2: float                         # Target area in square meters
    min_width_m: Optional[float] = None    # Minimum width constraint
    min_height_m: Optional[float] = None   # Minimum height constraint
    aspect_ratio: Optional[Tuple[float, float]] = None  # Min/max ratio
    adjacencies: Optional[List[str]] = None  # Required neighbors
    forbidden_adjacencies: Optional[List[str]] = None
    has_window: Optional[bool] = True
    natural_light_required: Optional[bool] = True
    ventilation_type: Optional[str] = "natural"  # natural, mechanical, both
```

### Boundary Model

```python
class Boundary(BaseModel):
    width: float                           # Width in meters
    height: float                          # Height in meters
    type: Literal["rectangle", "L_shape"] = "rectangle"
    polygon_coords: Optional[List[Tuple[float, float]]] = None
    area_sqft: Optional[float] = None
    plot_orientation: Optional[str] = "N"  # N, E, S, W
```

### Design Score Components

```python
class DesignMetrics(BaseModel):
    design_score: float                    # Overall (0-1)
    geometry_score: float                  # Room proportions
    adjacency_score: float                 # Logical placement
    functionality_score: float             # Usability
    area_utilization: float                # % of plot used
    connectivity: int                      # Rooms connected
    travel_efficiency: float               # Avg travel distance
```

### Compliance Report

```python
class ComplianceReport(BaseModel):
    status: Literal["PASS", "PASS_WITH_REPAIRS", "FAIL"]
    violations: List[str]
    regulations_checked: int
    emergency_exit_present: bool
    max_travel_distance_met: bool
    accessibility_compliant: bool
    fire_safety_compliant: bool
    natural_light_compliant: bool
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {},
  "request_id": "req_12345"
}
```

### Common Error Codes

| Code | Status | Description | Example |
|------|--------|-------------|---------|
| `validation_error` | 422 | Invalid request data | Area exceeds plot |
| `resource_not_found` | 404 | Session/layout not found | Session ID invalid |
| `unauthorized` | 401 | Authentication required | Missing API key |
| `rate_limited` | 429 | Too many requests | 100 req/min exceeded |
| `server_error` | 500 | Internal server error | Model loading failed |
| `timeout` | 504 | Request took too long | Generation >10s |

### Retry Logic

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Usage
response = requests_retry_session().post(
    "http://localhost:8000/api/generate",
    json=spec
)
```

---

## Authentication

### Optional API Key

```http
GET /api/sessions/sess_abc123 HTTP/1.1
Host: localhost:8000
Authorization: Bearer YOUR_API_KEY

# Or via query parameter
GET /api/sessions/sess_abc123?api_key=YOUR_API_KEY HTTP/1.1
```

### Rate Limiting

```
Default: 100 requests per minute per IP

Response headers:
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1711530900
```

---

## Integration Examples

### Python Integration

```python
import requests
import json

class BlueprintGPTClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.session_id = None
    
    def generate(self, rooms, boundary, backend="hybrid"):
        """Generate layout"""
        payload = {
            "rooms": rooms,
            "boundary": boundary,
            "backend_target": backend,
            "session_id": self.session_id
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.session_id = data["session_id"]
            return data
        else:
            raise Exception(f"Error: {response.json()}")
    
    def chat(self, message):
        """Natural language interface"""
        payload = {
            "message": message,
            "session_id": self.session_id
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        
        return response.json()
    
    def explain(self, layout_id):
        """Get layout explanation"""
        payload = {
            "layout_id": layout_id,
            "detail_level": "comprehensive"
        }
        
        response = requests.post(
            f"{self.base_url}/api/explain",
            json=payload
        )
        
        return response.json()

# Usage
client = BlueprintGPTClient()

# Generate layout
result = client.generate(
    rooms=[
        {"name": "Bedroom", "type": "Bedroom", "area_m2": 150},
        {"name": "Kitchen", "type": "Kitchen", "area_m2": 100}
    ],
    boundary={"width": 12, "height": 15}
)

# Refine via chat
response = client.chat("Make the kitchen bigger")

# Get explanation
explanation = client.explain(result["layout_id"])
```

### JavaScript Integration

```javascript
class BlueprintGPTClient {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
    this.sessionId = null;
  }

  async generate(rooms, boundary, backend = "hybrid") {
    const payload = {
      rooms,
      boundary,
      backend_target: backend,
      session_id: this.sessionId
    };

    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (response.ok) {
      const data = await response.json();
      this.sessionId = data.session_id;
      return data;
    } else {
      throw new Error(await response.text());
    }
  }

  async chat(message) {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: this.sessionId
      })
    });

    return await response.json();
  }
}

// Usage
const client = new BlueprintGPTClient();

const result = await client.generate(
  [
    { name: "Bedroom", type: "Bedroom", area_m2: 150 },
    { name: "Kitchen", type: "Kitchen", area_m2: 100 }
  ],
  { width: 12, height: 15 }
);

console.log("Layout generated:", result.layout_id);
```

### cURL Examples

```bash
# Generate layout
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "rooms": [
      {"name": "Bedroom", "type": "Bedroom", "area_m2": 150},
      {"name": "Kitchen", "type": "Kitchen", "area_m2": 100}
    ],
    "boundary": {"width": 12, "height": 15},
    "backend_target": "hybrid"
  }'

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "3BHK apartment"}'

# Get explanation
curl -X POST http://localhost:8000/api/explain \
  -H "Content-Type: application/json" \
  -d '{
    "layout_id": "layout_20240327_001",
    "detail_level": "comprehensive"
  }'

# Export as SVG
curl -X POST http://localhost:8000/api/export \
  -H "Content-Type: application/json" \
  -d '{"layout_id": "layout_20240327_001", "format": "svg"}' \
  -o layout.svg
```

---

## Advanced Features

### WebSocket Support (Real-time)

```javascript
// Real-time generation feedback
const ws = new WebSocket("ws://localhost:8000/ws/generate");

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "start_generation",
    rooms: [...],
    boundary: {...}
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === "progress") {
    console.log(`Generation: ${message.progress}%`);
  } else if (message.type === "complete") {
    console.log("Layout ready:", message.layout);
  }
};
```

### Batch Processing

```python
# Generate multiple layouts in one request
payload = {
  "batch": [
    {
      "rooms": [...],
      "boundary": {...},
      "name": "Option 1"
    },
    {
      "rooms": [...],
      "boundary": {...},
      "name": "Option 2"
    }
  ]
}

response = requests.post(
  "http://localhost:8000/api/batch_generate",
  json=payload
)

layouts = response.json()["results"]
```

### Webhooks

```python
# Register webhook for completion notification
payload = {
  "layout_id": "layout_20240327_001",
  "webhook_url": "https://myapp.com/blueprintgpt/callback",
  "events": ["generation_complete", "compliance_checked"]
}

requests.post(
  "http://localhost:8000/api/webhooks/register",
  json=payload
)

# Callback format (POST to your URL):
# {
#   "event": "generation_complete",
#   "layout_id": "...",
#   "timestamp": "...",
#   "data": {...}
# }
```

---

**Version**: 1.0  
**Date**: March 27, 2026  
**Status**: Production Ready
