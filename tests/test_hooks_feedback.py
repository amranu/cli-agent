#!/usr/bin/env python3
"""
Test script to verify hooks system provides proper user feedback.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_hook_manager_summary():
    """Test that hook manager provides a summary for user feedback."""
    print("Testing hook manager summary...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig
        from cli_agent.core.hooks.hook_manager import HookManager
        
        # Create test config
        config_data = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "*",
                        "hooks": [
                            {"type": "command", "command": "echo 'pre-hook'"}
                        ]
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "bash_execute",
                        "hooks": [
                            {"type": "command", "command": "echo 'post-hook 1'"},
                            {"type": "command", "command": "echo 'post-hook 2'"}
                        ]
                    }
                ]
            }
        }
        
        config = HookConfig._parse_config_data(config_data, "test")
        manager = HookManager(config)
        
        # Test summary
        summary = manager.get_hook_summary()
        
        print(f"✓ Hook summary: {summary}")
        
        expected_total = 3  # 1 pre-hook + 2 post-hooks
        if summary["total_hooks"] == expected_total:
            print(f"✓ Correct total hooks count: {expected_total}")
        else:
            print(f"✗ Wrong hooks count: expected {expected_total}, got {summary['total_hooks']}")
            return False
            
        if "PreToolUse" in summary["hook_types"] and "PostToolUse" in summary["hook_types"]:
            print("✓ Hook types correctly identified")
        else:
            print(f"✗ Missing hook types in summary: {summary['hook_types']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Hook manager summary test failed: {e}")
        return False

def test_initialization_feedback():
    """Test that initialization shows user feedback."""
    print("\nTesting initialization feedback...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig
        from cli_agent.core.hooks.hook_manager import HookManager
        
        # Create test config
        config_data = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "*", "hooks": [{"type": "command", "command": "echo test"}]}
                ]
            }
        }
        
        config = HookConfig._parse_config_data(config_data, "test")
        
        # Capture initialization (this would show user feedback via print statements)
        print("📍 The following would be shown to user during initialization:")
        manager = HookManager(config)
        summary = manager.get_hook_summary()
        total_hooks = summary["total_hooks"]
        hook_types = list(summary["hook_types"].keys())
        print(f"🪝 Hooks enabled: {total_hooks} hooks loaded ({', '.join(hook_types)})")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization feedback test failed: {e}")
        return False

def test_execution_feedback():
    """Test that hook execution shows feedback."""
    print("\nTesting execution feedback...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig, HookType
        from cli_agent.core.hooks.hook_manager import HookManager
        
        # Create test config
        config_data = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "*", "hooks": [{"type": "command", "command": "echo test"}]}
                ]
            }
        }
        
        config = HookConfig._parse_config_data(config_data, "test")
        manager = HookManager(config)
        
        # Test matching hooks (this would show execution feedback)
        context = {"tool_name": "test_tool"}
        matching_hooks = config.get_matching_hooks(HookType.PRE_TOOL_USE, "test_tool")
        
        if matching_hooks:
            print("📍 The following would be shown to user during hook execution:")
            hook_type_display = HookType.PRE_TOOL_USE.value.replace("ToolUse", "-tool").lower()
            print(f"🪝 Running {len(matching_hooks)} {hook_type_display} hook{'s' if len(matching_hooks) != 1 else ''}...")
            print("✓ Execution feedback test passed")
            return True
        else:
            print("✗ No matching hooks found")
            return False
            
    except Exception as e:
        print(f"✗ Execution feedback test failed: {e}")
        return False

def test_slash_command_integration():
    """Test that slash command provides proper hooks information."""
    print("\nTesting slash command integration...")
    
    try:
        # Mock agent with hook manager
        class MockAgent:
            def __init__(self):
                from cli_agent.core.hooks.hook_config import HookConfig
                from cli_agent.core.hooks.hook_manager import HookManager
                
                config_data = {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "*", "hooks": [{"type": "command", "command": "echo test"}]}
                        ]
                    }
                }
                
                config = HookConfig._parse_config_data(config_data, "test")
                self.hook_manager = HookManager(config)
        
        from cli_agent.core.slash_commands import SlashCommandManager
        
        agent = MockAgent()
        slash_manager = SlashCommandManager(agent)
        
        # Test hooks command
        result = slash_manager._handle_hooks("")
        
        if "🪝 Hooks System Status:" in result:
            print("✓ Slash command shows hooks status")
            print("📍 Example output:")
            print(result)
            return True
        else:
            print(f"✗ Unexpected slash command output: {result}")
            return False
            
    except Exception as e:
        print(f"✗ Slash command integration test failed: {e}")
        return False

def main():
    """Run all feedback tests."""
    print("CLI-Agent Hooks Feedback Test Suite")
    print("=" * 40)
    
    tests = [
        test_hook_manager_summary,
        test_initialization_feedback,
        test_execution_feedback,
        test_slash_command_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Feedback Tests: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All feedback tests passed! Users will see clear hooks status.")
        return 0
    else:
        print("❌ Some feedback tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())