# Redundant Interrupt Handling Patterns Report

## Overview
After implementing the new global interrupt system (`global_interrupt.py` and `interrupt_aware_streaming.py`), several redundant interrupt handling patterns can be removed from the codebase.

## 1. Signal Handling Code

### File: `cli_agent/core/input_handler.py`
**Lines: 123-125, 180-182**
```python
# Manually trigger the global interrupt manager since basic input intercepted the signal
import signal
import os
os.kill(os.getpid(), signal.SIGINT)  # Re-send SIGINT to trigger our global handler
```
**Description**: Manual re-sending of SIGINT signals when KeyboardInterrupt is caught.
**Replacement**: No longer needed as the global interrupt manager already handles signal registration centrally.

## 2. Manual Interrupt Checking Patterns

### File: `cli_agent/core/base_agent.py`
**Lines: 465-477**
```python
# Check for interruption before starting any tool execution
from cli_agent.core.global_interrupt import get_global_interrupt_manager

global_interrupt_manager = get_global_interrupt_manager()

if global_interrupt_manager.is_interrupted():
    all_tool_output.append("🛑 Tool execution interrupted by global interrupt")
    return all_tool_output

# Also check for local input handler interruption
if input_handler and hasattr(input_handler, 'interrupted'):
    # Check for both existing interrupted state and new interrupts
    if input_handler.interrupted or input_handler.check_for_interrupt():
        all_tool_output.append("🛑 Tool execution interrupted by user")
        return all_tool_output
```
**Description**: Dual checking of both global interrupt manager and local input handler interrupts.
**Replacement**: Can be simplified to only use the global interrupt manager's `check_interrupt()` function.

### File: `cli_agent/core/input_handler.py`
**Lines: 201-272 (check_for_interrupt method)**
```python
def check_for_interrupt(self) -> bool:
    """Check if an interrupt signal is pending without blocking.
    ...
```
**Description**: Manual interrupt checking by reading terminal input for ESC/Ctrl+C.
**Replacement**: Can be replaced with calls to the global interrupt manager's `is_interrupted()`.

## 3. Streaming Interrupt Handling

### File: `cli_agent/core/base_llm_provider.py`
**Lines: 104-109**
```python
# Check for interrupts during streaming processing
from cli_agent.core.global_interrupt import get_global_interrupt_manager
interrupt_manager = get_global_interrupt_manager()
if interrupt_manager.is_interrupted():
    logger.info("Streaming response interrupted by user")
    raise KeyboardInterrupt("Streaming response interrupted")
```
**Description**: Manual interrupt checking during streaming response processing.
**Replacement**: Already uses the global interrupt manager, but could be simplified to use `check_interrupt()` instead of manual checking and raising.

### File: All Provider Files (anthropic_provider.py, deepseek_provider.py, etc.)
**Pattern**: Manual wrapping of streaming responses with `make_streaming_interruptible()`
```python
# Wrap response with interrupt checking
interruptible_response = self.make_streaming_interruptible(response, "Provider streaming")
async for chunk in interruptible_response:
```
**Description**: All providers manually wrap their streaming responses.
**Replacement**: This pattern is actually good and uses the new interrupt-aware streaming utilities. No change needed.

## 4. Subprocess Interrupt Handling

### File: `cli_agent/core/builtin_tool_executor.py`
**Lines: 31-34**
```python
from cli_agent.core.interrupt_aware_streaming import InterruptAwareSubprocess

# Use interrupt-aware subprocess execution
result = await InterruptAwareSubprocess.run_with_interrupt_checking(
```
**Description**: Already using the new interrupt-aware subprocess handler.
**Replacement**: No change needed - this is the correct pattern.

## 5. AsyncIO Interrupt Patterns

### File: `cli_agent/core/base_llm_provider.py`
**Lines: 55-58, 125-133**
```python
# Make API request with streaming and aggressive interrupt monitoring
from cli_agent.core.interrupt_aware_streaming import run_with_interrupt_monitoring

response = await run_with_interrupt_monitoring(
```
**Description**: Already using the new interrupt monitoring utility.
**Replacement**: No change needed - this is the correct pattern.

### File: `agent.py`
**Lines: 323-335**
```python
except KeyboardInterrupt:
    # Check if this is the second interrupt (user wants to exit)
    from cli_agent.core.global_interrupt import get_global_interrupt_manager
    interrupt_manager = get_global_interrupt_manager()
    
    if interrupt_manager._interrupt_count >= 2:
        # Second interrupt - user wants to exit, re-raise to exit the CLI
        click.echo("\n👋 Exiting...")
        raise
    else:
        # First interrupt - just return to exit this chat session
        click.echo("\n👋 Chat interrupted by user")
        return
```
**Description**: Manual handling of interrupt counts to determine exit behavior.
**Replacement**: This is actually using the global interrupt manager correctly. However, accessing `_interrupt_count` directly is not ideal - should use a public method.

## 6. ESC Key Detection Patterns

### File: `cli_agent/core/input_handler.py`
**Lines: 244-245**
```python
# Check for ESC (27) or Ctrl+C (3)
if ord(char) == 27 or ord(char) == 3:  # ESC or Ctrl+C
```
**Description**: Manual detection of ESC and Ctrl+C key codes.
**Replacement**: Could be integrated with the global interrupt manager's callback system instead of manual checking.

## Summary of Recommended Changes

1. **Remove manual signal re-sending** in `input_handler.py` - the global interrupt manager handles this.

2. **Simplify dual interrupt checking** in `base_agent.py` - use only the global interrupt manager.

3. **Replace manual ESC/Ctrl+C detection** in `input_handler.py` with global interrupt manager integration.

4. **Add public method** to global interrupt manager for checking interrupt count instead of accessing `_interrupt_count` directly.

5. **Keep the interrupt-aware streaming wrappers** - these are correctly using the new system.

6. **Keep the subprocess interrupt handling** - already using the new `InterruptAwareSubprocess`.

The main redundancy is in the input handler's manual interrupt detection and the dual checking of both local and global interrupt states. These can be consolidated to use only the global interrupt system.