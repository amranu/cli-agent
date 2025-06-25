#!/usr/bin/env python3
"""Simple test of the event-driven response display."""

import asyncio
import sys
from config import load_config
from agent import create_host

async def test_simple():
    config = load_config()
    host = create_host(config)
    
    # Initialize event system
    await host.event_bus.start_processing()
    
    print("🧪 Testing single response with events...")
    
    # Generate a simple response
    messages = [{'role': 'user', 'content': 'Say "Hello World"'}]
    response = await host.generate_response(messages, stream=True)
    
    # The response should be an async generator
    # When we iterate it, _process_streaming_chunks_with_events should:
    # 1. Collect all chunks silently
    # 2. Emit one formatted TextEvent at the end
    
    if hasattr(response, "__aiter__"):
        print("📝 Processing streaming response...")
        final_content = ""
        async for chunk in response:
            final_content += str(chunk)
        print(f"✅ Collected: {repr(final_content)}")
    else:
        print(f"✅ Non-streaming: {repr(str(response))}")
    
    await asyncio.sleep(0.1)  # Let events process
    await host.event_bus.stop_processing()
    print("🎉 Test complete!")

if __name__ == "__main__":
    asyncio.run(test_simple())