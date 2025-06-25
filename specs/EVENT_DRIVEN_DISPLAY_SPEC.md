# JSON Event-Driven Display Architecture Specification

## Current State Analysis

The codebase already has significant JSON event infrastructure:

### Existing JSON Event System
1. **StreamingJSONHandler** (`streaming_json.py`): Complete Claude Code-compatible JSON event system
   - SystemInitMessage, AssistantMessage, UserMessage classes
   - Methods: send_assistant_text(), send_assistant_tool_use(), send_tool_result()
   - Already integrated into main agent via streaming_json_callback

2. **Current Integration Points**:
   - `agent.py` lines 982-984: callback registration system
   - `base_agent.py` line 1762: streaming_json_callback check to skip normal display
   - Response handlers already detect streaming JSON mode

3. **Current Display Pipeline**:
   - `chat_interface.py`: handle_streaming_display() processes async generators
   - `response_handler.py`: handle_streaming_response_generic() coordinates tool execution
   - Input detection during streaming (ESC key monitoring)

## Proposed Refactoring: Event-Driven Display Architecture

### Core Concept
Transform the current character-by-character streaming into discrete JSON events that drive all display updates. This would create an "event bus" architecture where:

1. **All LLM responses generate structured events** (already exists)
2. **Display system consumes events** (needs implementation)  
3. **Input detection runs independently** (partially exists)

### Implementation Plan

#### Phase 1: Event Bus Infrastructure (Low Difficulty)
- Create `EventBus` class to manage event flow
- Events: `TextEvent`, `ToolCallEvent`, `ToolResultEvent`, `StatusEvent`, `ErrorEvent`
- Centralized event dispatcher that all components can subscribe to

#### Phase 2: Display Manager (Medium Difficulty)
- Create `DisplayManager` class that subscribes to events
- Handles all terminal output formatting and rendering
- Separates display logic from response processing
- Never yields control during display operations

#### Phase 3: Input Detection Separation (Medium Difficulty)  
- Make input detection fully independent of display
- Continuous background monitoring for ESC/Ctrl+C
- Events can be interrupted but display pipeline cannot be

#### Phase 4: Integration and Migration (High Difficulty)
- Modify all response handlers to emit events instead of direct printing
- Update streaming response processing to use event bus
- Migrate chat interface to event-driven model
- Maintain backward compatibility

### Benefits
1. **Uninterruptible Display**: Display operations never yield control
2. **Cleaner Architecture**: Separation of concerns between input, processing, and display
3. **Better Testing**: Event-driven system easier to test and mock
4. **Extensibility**: Easy to add new event types and display modes
5. **Consistency**: All output goes through same formatting pipeline

### Difficulty Assessment: **Medium to High**

**Medium Difficulty Aspects:**
- JSON event infrastructure already exists
- Callback system already in place
- Display separation is conceptually straightforward

**High Difficulty Aspects:**
- Need to refactor existing streaming response handling
- Complex integration with tool execution flow
- Ensuring backward compatibility
- Managing state transitions during interruption
- Coordinating between event bus and existing async generators

### Estimated Effort: 2-3 days
- Day 1: Event bus and display manager infrastructure
- Day 2: Response handler migration and integration
- Day 3: Testing, debugging, and polish

### Risks
1. **Breaking existing integrations** (subagents, MCP tools)
2. **Performance impact** from additional event layer
3. **Complex debugging** due to increased indirection
4. **Incomplete migration** leaving mixed display approaches

## Technical Implementation Details

### Event Bus Architecture

```python
class EventBus:
    """Central event dispatcher for display and processing coordination."""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_queue = asyncio.Queue()
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to specific event types."""
        
    async def emit(self, event: Event):
        """Emit event to all subscribers."""
        
    async def process_events(self):
        """Main event processing loop."""
```

### Event Types

```python
@dataclass
class TextEvent(Event):
    """Text content from LLM response."""
    content: str
    chunk_id: Optional[str] = None
    
@dataclass 
class ToolCallEvent(Event):
    """Tool execution request."""
    tool_name: str
    arguments: Dict[str, Any]
    tool_id: str
    
@dataclass
class ToolResultEvent(Event):
    """Tool execution result."""
    tool_id: str
    result: str
    is_error: bool = False
    
@dataclass
class StatusEvent(Event):
    """Status updates (subagents, interruptions, etc)."""
    status: str
    details: Optional[str] = None
```

### Display Manager

```python
class DisplayManager:
    """Handles all terminal output in response to events."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.formatter = ResponseFormatter()
        
        # Subscribe to relevant events
        event_bus.subscribe("text", self.handle_text_event)
        event_bus.subscribe("tool_call", self.handle_tool_call_event)
        event_bus.subscribe("tool_result", self.handle_tool_result_event)
        
    async def handle_text_event(self, event: TextEvent):
        """Display text content immediately."""
        formatted = self.formatter.format_markdown(event.content)
        print(formatted, end="", flush=True)
        
    async def handle_tool_call_event(self, event: ToolCallEvent):
        """Display tool execution start."""
        print(f"\n🔧 Executing {event.tool_name}...", flush=True)
        
    async def handle_tool_result_event(self, event: ToolResultEvent):
        """Display tool execution result."""
        if event.is_error:
            print(f"❌ Error: {event.result}", flush=True)
        else:
            print(f"✅ Result: {event.result[:100]}...", flush=True)
```

### Integration Points

1. **Response Handlers**: Emit events instead of direct printing
2. **Tool Execution**: Emit ToolCallEvent before execution, ToolResultEvent after
3. **Streaming**: Convert streaming chunks to TextEvent sequence
4. **Input Detection**: Independent background task that can interrupt event processing

### Migration Strategy

1. **Phase 1**: Implement event bus and display manager alongside existing system
2. **Phase 2**: Add event emission to response handlers while maintaining current display
3. **Phase 3**: Switch display manager to active mode, disable legacy display
4. **Phase 4**: Remove legacy display code after validation

## Conclusion

The refactoring is feasible and would significantly improve the architecture's robustness and maintainability. The existing JSON event infrastructure provides a solid foundation, making this primarily an integration and refactoring effort rather than building from scratch.

Key success factors:
1. Incremental migration to avoid breaking changes
2. Comprehensive testing at each phase
3. Maintaining backward compatibility during transition
4. Clear separation of concerns between event generation, processing, and display