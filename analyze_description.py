#!/usr/bin/env python3
"""Analyze the MCP tool description structure."""

import asyncio
from cli_agent.mcp.model_server import create_model_server

async def analyze_description():
    server = create_model_server()
    tools = await server.get_tools()
    
    if 'chat' in tools:
        chat_tool = tools['chat']
        desc = chat_tool.description
        
        print(f'Full description length: {len(desc)}')
        print()
        
        # Find where 'Available models:' starts
        models_start = desc.find('Available models: ')
        if models_start != -1:
            models_part = desc[models_start + len('Available models: '):]
            models_list = [m.strip() for m in models_part.split(',')]
            
            print(f'Total models: {len(models_list)}')
            print()
            
            # Show first and last 10 models
            print('First 10 models:')
            for i, model in enumerate(models_list[:10]):
                print(f'  {i+1:2d}. {model}')
            
            print()
            print('Last 10 models:')
            for i, model in enumerate(models_list[-10:]):
                print(f'  {len(models_list)-9+i:3d}. {model}')
            
            print()
            
            # Check if only last models are deepseek/google
            last_50 = models_list[-50:]
            providers_in_last_50 = set()
            for model in last_50:
                if ':' in model:
                    provider = model.split(':')[0]
                    providers_in_last_50.add(provider)
            
            print(f'Providers in last 50 models: {sorted(providers_in_last_50)}')
            
            # Show where each provider starts and ends
            print('\nProvider positions in model list:')
            current_provider = None
            provider_ranges = {}
            
            for i, model in enumerate(models_list):
                if ':' in model:
                    provider = model.split(':')[0]
                    if provider != current_provider:
                        if current_provider:
                            provider_ranges[current_provider]['end'] = i - 1
                        provider_ranges[provider] = {'start': i}
                        current_provider = provider
            
            # Close the last provider
            if current_provider:
                provider_ranges[current_provider]['end'] = len(models_list) - 1
            
            for provider, range_info in provider_ranges.items():
                start = range_info['start']
                end = range_info['end']
                count = end - start + 1
                print(f'  {provider}: positions {start+1}-{end+1} ({count} models)')

if __name__ == "__main__":
    asyncio.run(analyze_description())