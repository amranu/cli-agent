#!/usr/bin/env python3
"""Async version of the CLI that avoids Click's sync/async mixing issues."""

import asyncio
import sys
from simple_prompt import SimplePromptManager
from config import load_config


async def interactive_chat():
    """Simple async interactive chat without Click complications."""
    try:
        # Load configuration
        config = load_config()
        
        # Check if Gemini backend should be used
        if config.deepseek_model == "gemini":
            if not config.gemini_api_key:
                print("Error: GEMINI_API_KEY not set. Run 'python agent.py init' first and update .env file.")
                return
            
            # Import and create Gemini host
            from mcp_gemini_host import MCPGeminiHost
            host = MCPGeminiHost(config)
        else:
            if not config.deepseek_api_key:
                print("Error: DEEPSEEK_API_KEY not set. Run 'python agent.py init' first and update .env file.")
                return
            
            # Create Deepseek host
            from mcp_deepseek_host import MCPDeepseekHost
            host = MCPDeepseekHost(config)
        
        # Connect to all configured MCP servers
        for server_name, server_config in config.mcp_servers.items():
            print(f"Starting MCP server: {server_name}")
            success = await host.start_mcp_server(server_name, server_config)
            if not success:
                print(f"Failed to start server: {server_name}")
            else:
                print(f"✅ Connected to MCP server: {server_name}")
        
        # Create prompt manager
        prompt_manager = SimplePromptManager()
        
        # Print welcome
        if hasattr(host, 'deepseek_config'):
            model_info = host.deepseek_config.model
        elif hasattr(host, 'gemini_config'):
            model_info = host.gemini_config.model
        else:
            model_info = "Unknown"
        
        print(f"\n🤖 MCP Agent - Interactive Chat")
        print(f"Model: {model_info}")
        print(f"Available tools: {len(host.available_tools)}")
        print("Commands: 'quit' to exit, 'tools' to list tools, ESC/Ctrl+C to interrupt")
        print("=" * 60)
        
        messages = []
        
        while True:
            try:
                # Get user input
                user_input = prompt_manager.get_multiline_input("You: ")
                
                if user_input is None:  # Interrupted
                    print("\n🛑 Interrupted")
                    continue
                
                if user_input.lower().strip() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower().strip() == 'tools':
                    print("\nAvailable tools:")
                    for tool_name, tool_info in host.available_tools.items():
                        print(f"  • {tool_name}: {tool_info['description']}")
                    print()
                    continue
                
                # Handle slash commands
                if user_input.strip().startswith('/'):
                    try:
                        if hasattr(host, 'slash_commands'):
                            slash_response = await host.slash_commands.handle_slash_command(user_input.strip())
                            if slash_response:
                                print(f"\n{slash_response}\n")
                        continue
                    except Exception as e:
                        print(f"\nError handling slash command: {e}\n")
                        continue
                
                if not user_input.strip():
                    continue
                
                # Add user message
                messages.append({"role": "user", "content": user_input})
                
                print("\n🤖 Assistant:")
                try:
                    # Get response
                    if hasattr(host, 'chat_completion'):
                        response = await host.chat_completion(messages, stream=False)
                    else:
                        response = await host.generate_response(messages)
                    
                    print(response)
                    messages.append({"role": "assistant", "content": str(response)})
                    
                except Exception as e:
                    print(f"Error: {e}")
                
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    finally:
        if 'host' in locals():
            try:
                await host.shutdown()
            except:
                pass


async def main():
    """Main async entry point."""
    if len(sys.argv) > 1:
        print("This is the async chat version. For other commands, use: python agent.py [command]")
        return
    
    await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())