#!/usr/bin/env python3
"""
Simple test script to demonstrate the hooks system functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_sample_hook_config():
    """Create a sample hook configuration for testing."""
    hook_config = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "echo '[PRE-TOOL] About to execute: {{tool_name}} with args: {{tool_args}}'"
                        }
                    ]
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "bash_execute",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "echo '[POST-TOOL] Bash command completed in {{execution_time}}s'"
                        }
                    ]
                }
            ],
            "Notification": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "echo '[NOTIFICATION] {{message}}'"
                        }
                    ]
                }
            ],
            "Stop": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "echo '[STOP] Agent finished responding. Conversation length: {{conversation_length}}'"
                        }
                    ]
                }
            ]
        }
    }
    
    return hook_config

def test_hook_config_loading():
    """Test hook configuration loading and validation."""
    print("Testing hook configuration loading...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig, HookType
        
        # Create sample config data
        config_data = create_sample_hook_config()
        
        # Test parsing
        config = HookConfig._parse_config_data(config_data, "test_source")
        
        print(f"✓ Loaded configuration with {len(config.hooks)} hook types")
        
        # Test validation
        errors = config.validate()
        if errors:
            print(f"⚠ Validation errors: {errors}")
        else:
            print("✓ Configuration validation passed")
            
        # Test hook matching
        bash_hooks = config.get_matching_hooks(HookType.PRE_TOOL_USE, "bash_execute")
        print(f"✓ Found {len(bash_hooks)} pre-tool hooks for bash_execute")
        
        # Test pattern matching
        all_hooks = config.get_matching_hooks(HookType.POST_TOOL_USE, "bash_execute")
        print(f"✓ Found {len(all_hooks)} post-tool hooks for bash_execute")
        
        return True
        
    except Exception as e:
        print(f"✗ Hook configuration test failed: {e}")
        return False

def test_hook_execution():
    """Test hook execution with sample commands."""
    print("\nTesting hook execution...")
    
    try:
        from cli_agent.core.hooks.hook_executor import HookExecutor
        from cli_agent.core.hooks.hook_config import HookDefinition
        
        executor = HookExecutor()
        
        # Test simple command
        hook = HookDefinition(
            type="command",
            command="echo 'Hello from hook: {{tool_name}}'"
        )
        
        context = {
            "tool_name": "test_tool",
            "tool_args": {"test": "value"},
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        import asyncio
        result = asyncio.run(executor.execute_hook(hook, context))
        
        print(f"✓ Hook executed successfully: exit_code={result.exit_code}")
        print(f"✓ Hook output: {result.stdout.strip()}")
        
        if result.success:
            print("✓ Hook execution test passed")
            return True
        else:
            print(f"✗ Hook execution failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Hook execution test failed: {e}")
        return False

def test_template_substitution():
    """Test template variable substitution."""
    print("\nTesting template variable substitution...")
    
    try:
        from cli_agent.core.hooks.hook_executor import HookExecutor
        
        executor = HookExecutor()
        
        command = "echo 'Tool: {{tool_name}}, Args: {{tool_args}}, Time: {{timestamp}}'"
        context = {
            "tool_name": "bash_execute",
            "tool_args": {"command": "ls -la"},
            "timestamp": "2024-01-01T12:00:00Z",
            "result": "Command output here"
        }
        
        substituted = executor.substitute_template_variables(command, context)
        print(f"✓ Template substitution result: {substituted}")
        
        # Verify all variables were substituted
        if "{{" not in substituted and "}}" not in substituted:
            print("✓ Template substitution test passed")
            return True
        else:
            print("✗ Some template variables not substituted")
            return False
            
    except Exception as e:
        print(f"✗ Template substitution test failed: {e}")
        return False

def create_test_config_file():
    """Create a test configuration file for integration testing."""
    # Create .config/agent directory in current directory
    config_dir = Path.cwd() / ".config" / "agent"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "settings.json"
    config_data = create_sample_hook_config()
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✓ Created test configuration file: {config_file}")
    return config_file

def test_config_file_loading():
    """Test loading configuration from files."""
    print("\nTesting configuration file loading...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig
        
        # Create test config file
        config_file = create_test_config_file()
        
        # Test loading from file
        config = HookConfig.load_from_file(config_file)
        
        if config.has_hooks():
            print(f"✓ Loaded configuration from file with {len(config.hooks)} hook types")
            return True
        else:
            print("✗ No hooks found in loaded configuration")
            return False
            
    except Exception as e:
        print(f"✗ Configuration file loading test failed: {e}")
        return False
    finally:
        # Cleanup
        try:
            config_file.unlink()
            config_file.parent.rmdir()  # Remove agent dir
            config_file.parent.parent.rmdir()  # Remove .config dir
        except:
            pass

def main():
    """Run all hook system tests."""
    print("CLI-Agent Hooks System Test Suite")
    print("=" * 40)
    
    tests = [
        test_hook_config_loading,
        test_template_substitution,
        test_hook_execution,
        test_config_file_loading,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Hooks system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())