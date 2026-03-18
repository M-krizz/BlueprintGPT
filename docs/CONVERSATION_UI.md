# ChatGPT-style Conversation UI

This document describes the ChatGPT-like conversation interface for BlueprintGPT that enables natural language interactions with AI-powered explanations and corrections.

## Overview

The conversation system provides:
1. **Multi-turn conversations** - Accumulate requirements across multiple messages
2. **AI-powered NL understanding** - Gemini API converts natural language to structured specs
3. **Multiple design generation** - Generate and rank multiple floor plan options
4. **AI explanations** - Each design includes an explanation of its ranking and trade-offs
5. **Natural language corrections** - Request changes using plain language ("make the kitchen larger")

## Architecture

```
User Input (NL)
     │
     ▼
┌─────────────────┐
│ Gemini Adapter  │ ── Extracts structured spec from natural language
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Conversation    │ ── Manages session state, spec accumulation
│ Session Manager │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Backend Runner  │ ── Generates multiple floor plan designs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Explainer       │ ── Generates ranking explanations via Gemini
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Correction      │ ── Parses correction requests, applies changes
│ Handler         │
└─────────────────┘
```

## API Endpoints

### Core Conversation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/conversation/message` | POST | Send a message, receive AI response + designs |
| `/conversation/correct` | POST | Request corrections to a specific design |
| `/conversation/session/new` | POST | Create a new conversation session |
| `/conversation/session/{id}` | GET | Get session state |
| `/conversation/session/{id}` | DELETE | Delete a session |
| `/status` | GET | Check system status including Gemini availability |

### Request/Response Examples

#### Send Message
```json
POST /conversation/message
{
  "session_id": "abc123",
  "message": "I need a 3 bedroom house with kitchen near dining",
  "boundary": {"width": 15, "height": 10},
  "num_designs": 3
}

Response:
{
  "session_id": "abc123",
  "response": "I'll design a 3 bedroom house with the kitchen adjacent to the dining area...",
  "state": "generated",
  "designs": [
    {
      "index": 0,
      "rank": 1,
      "score": 0.87,
      "svg_url": "/outputs/design_1.svg",
      "violations": []
    },
    // ... more designs
  ],
  "explanations": [
    {
      "rank": 1,
      "summary": "This design maximizes adjacency satisfaction...",
      "strengths": ["Kitchen next to dining", "Good traffic flow"],
      "weaknesses": ["Slightly cramped bedrooms"],
      "ranking_reason": "Highest overall score with full compliance"
    }
  ]
}
```

#### Request Correction
```json
POST /conversation/correct
{
  "session_id": "abc123",
  "design_index": 0,
  "correction": "make the master bedroom larger and move kitchen away from entrance",
  "regenerate": true
}

Response:
{
  "success": true,
  "message": "Applied 2 changes: resize_room, move_room",
  "changes_applied": [
    {"type": "resize_room", "room": "Bedroom_1", "size_change": "larger"},
    {"type": "move_room", "room": "Kitchen", "direction": "away_from_entrance"}
  ],
  "needs_regeneration": true,
  "new_designs": [...]
}
```

## Frontend Features

### Design Gallery
Multiple designs are displayed in a responsive grid with:
- Rank badge (gold/silver/bronze for top 3)
- Design score
- SVG preview
- AI-generated explanation
- Strengths and weaknesses summary
- View and Modify buttons

### Correction UI
Click "Modify" on any design to open the correction panel:
- Quick suggestion chips for common corrections
- Free-form text input for any correction
- Submit button to apply changes and regenerate

### Session Management
- Session ID displayed in top nav
- New Session button to start fresh
- Export/Import for saving conversations

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | "" | Google Gemini API key |
| `GEMINI_MODEL` | "gemini-1.5-flash" | Gemini model to use |
| `GEMINI_ENABLED` | "true" | Enable/disable Gemini features |

### Running the Server

```bash
# Set Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Start the server
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

## Supported Corrections

The system understands these correction types:

| Type | Examples |
|------|----------|
| Add room | "add a bathroom", "add another bedroom" |
| Remove room | "remove the store room", "take out the garage" |
| Resize room | "make kitchen larger", "reduce bedroom size" |
| Move room | "move bedroom to the left", "shift kitchen away from entrance" |
| Swap rooms | "swap kitchen and dining", "exchange bedroom positions" |
| Change adjacency | "put kitchen next to living room", "separate bathroom from bedroom" |

## Fallback Behavior

If Gemini is unavailable:
- NL parsing falls back to regex-based extraction
- Explanations use template-based summaries
- Corrections use keyword matching
- All core functionality remains operational

## Testing

Run the conversation tests:
```bash
python -m pytest tests/unit/test_conversation.py -v
```

34 tests cover:
- Message and session management
- Spec accumulation across turns
- Correction parsing and application
- Geometry transformations
- Validation warnings
