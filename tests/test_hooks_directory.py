#!/usr/bin/env python3
"""
Test script to verify hooks directory loading functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_individual_hook_loading():
    """Test loading individual hook files from a directory."""
    print("Testing individual hook file loading...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig
        
        # Create temporary directory with hook files
        with tempfile.TemporaryDirectory() as temp_dir:
            hooks_dir = Path(temp_dir) / "hooks"
            hooks_dir.mkdir()
            
            # Create individual hook files
            pre_hook = hooks_dir / "pre-security.json"
            with open(pre_hook, 'w') as f:
                json.dump({
                    "matcher": "bash_execute",
                    "command": "echo 'Security check: {{tool_args}}'",
                    "timeout": 10
                }, f)
            
            post_hook = hooks_dir / "post-format.json" 
            with open(post_hook, 'w') as f:
                json.dump({
                    "matcher": "write_file|replace_in_file", 
                    "command": "black '{{tool_args.file_path}}'",
                    "timeout": 15
                }, f)
            
            notify_hook = hooks_dir / "notify-desktop.json"
            with open(notify_hook, 'w') as f:
                json.dump({
                    "matcher": "*",
                    "command": "notify-send 'CLI-Agent' '{{message}}'"
                }, f)
            
            stop_hook = hooks_dir / "stop-log.json"
            with open(stop_hook, 'w') as f:
                json.dump({
                    "matcher": "*", 
                    "command": "echo 'Session ended: {{timestamp}}' >> ~/.session.log"
                }, f)
            
            # Load configuration from directory
            config = HookConfig.load_from_directory(hooks_dir)
            
            if not config.has_hooks():
                print("✗ No hooks loaded from directory")
                return False
                
            # Verify all hook types were detected
            expected_types = {"PreToolUse", "PostToolUse", "Notification", "Stop"}
            actual_types = set(ht.value for ht in config.hooks.keys())
            
            if not expected_types.issubset(actual_types):
                print(f"✗ Missing hook types. Expected: {expected_types}, Got: {actual_types}")
                return False
                
            print(f"✓ Loaded hooks for types: {actual_types}")
            
            # Test hook matching
            from cli_agent.core.hooks.hook_config import HookType
            
            bash_hooks = config.get_matching_hooks(HookType.PRE_TOOL_USE, "bash_execute")
            if len(bash_hooks) != 1:
                print(f"✗ Expected 1 bash pre-hook, got {len(bash_hooks)}")
                return False
                
            write_hooks = config.get_matching_hooks(HookType.POST_TOOL_USE, "write_file")
            if len(write_hooks) != 1:
                print(f"✗ Expected 1 write post-hook, got {len(write_hooks)}")
                return False
                
            print("✓ Hook matching works correctly")
            print("✓ Individual hook file loading test passed")
            return True
            
    except Exception as e:
        print(f"✗ Individual hook loading test failed: {e}")
        return False

def test_mixed_config_sources():
    """Test loading from both individual files and traditional config."""
    print("\nTesting mixed configuration sources...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create traditional config file
            settings_file = temp_path / "settings.json"
            with open(settings_file, 'w') as f:
                json.dump({
                    "hooks": {
                        "PreToolUse": [
                            {
                                "matcher": "*",
                                "hooks": [
                                    {"type": "command", "command": "echo 'Traditional pre-hook'"}
                                ]
                            }
                        ]
                    }
                }, f)
            
            # Create hooks directory
            hooks_dir = temp_path / "hooks"
            hooks_dir.mkdir()
            
            post_hook = hooks_dir / "post-format.json"
            with open(post_hook, 'w') as f:
                json.dump({
                    "matcher": "write_file",
                    "command": "echo 'Directory post-hook'"
                }, f)
            
            # Load both sources
            file_config = HookConfig.load_from_file(settings_file)
            dir_config = HookConfig.load_from_directory(hooks_dir)
            
            merged_config = HookConfig._merge_configs(file_config, dir_config)
            
            # Verify both sources were loaded
            from cli_agent.core.hooks.hook_config import HookType
            
            pre_hooks = merged_config.get_matching_hooks(HookType.PRE_TOOL_USE, "test_tool")
            post_hooks = merged_config.get_matching_hooks(HookType.POST_TOOL_USE, "write_file")
            
            if len(pre_hooks) != 1:
                print(f"✗ Expected 1 pre-hook from traditional config, got {len(pre_hooks)}")
                return False
                
            if len(post_hooks) != 1:
                print(f"✗ Expected 1 post-hook from directory, got {len(post_hooks)}")
                return False
                
            print("✓ Both traditional config and directory hooks loaded")
            print("✓ Mixed configuration sources test passed")
            return True
            
    except Exception as e:
        print(f"✗ Mixed config sources test failed: {e}")
        return False

def test_filename_type_detection():
    """Test automatic hook type detection from filenames."""
    print("\nTesting filename-based type detection...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig, HookType
        
        test_cases = [
            ("pre-security.json", HookType.PRE_TOOL_USE),
            ("before-validation.json", HookType.PRE_TOOL_USE),
            ("post-format.json", HookType.POST_TOOL_USE),
            ("after-cleanup.json", HookType.POST_TOOL_USE),
            ("notify-desktop.json", HookType.NOTIFICATION),
            ("notification-slack.json", HookType.NOTIFICATION),
            ("stop-session.json", HookType.STOP),
            ("end-cleanup.json", HookType.STOP),
            ("finish-log.json", HookType.STOP),
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            hooks_dir = Path(temp_dir) / "hooks"
            hooks_dir.mkdir()
            
            for filename, expected_type in test_cases:
                hook_file = hooks_dir / filename
                with open(hook_file, 'w') as f:
                    json.dump({
                        "matcher": "*",
                        "command": f"echo 'Hook from {filename}'"
                    }, f)
            
            config = HookConfig.load_from_directory(hooks_dir)
            
            # Verify all expected types were detected
            detected_types = set(config.hooks.keys())
            expected_types = set(expected_type for _, expected_type in test_cases)
            
            if detected_types != expected_types:
                print(f"✗ Type detection failed. Expected: {expected_types}, Got: {detected_types}")
                return False
                
            print(f"✓ Correctly detected hook types: {[t.value for t in detected_types]}")
            print("✓ Filename-based type detection test passed")
            return True
            
    except Exception as e:
        print(f"✗ Filename type detection test failed: {e}")
        return False

def test_real_directory_loading():
    """Test loading from the actual example hooks directory."""
    print("\nTesting real directory loading...")
    
    try:
        from cli_agent.core.hooks.hook_config import HookConfig
        
        # Test loading from the actual example directory we created
        hooks_dir = Path("/home/andrew/Github/cli-agent/.config/agent/hooks")
        
        if not hooks_dir.exists():
            print("✗ Example hooks directory not found")
            return False
            
        config = HookConfig.load_from_directory(hooks_dir)
        
        if not config.has_hooks():
            print("✗ No hooks loaded from example directory")
            return False
            
        summary = {
            "total_hooks": 0,
            "hook_types": {}
        }
        
        for hook_type, matchers in config.hooks.items():
            hook_count = sum(len(matcher.hooks) for matcher in matchers)
            summary["hook_types"][hook_type.value] = {
                "matchers": len(matchers),
                "hooks": hook_count
            }
            summary["total_hooks"] += hook_count
            
        print(f"✓ Loaded {summary['total_hooks']} hooks from example directory")
        print(f"✓ Hook types: {list(summary['hook_types'].keys())}")
        print("✓ Real directory loading test passed")
        return True
        
    except Exception as e:
        print(f"✗ Real directory loading test failed: {e}")
        return False

def main():
    """Run all directory loading tests."""
    print("CLI-Agent Hooks Directory Loading Test Suite")
    print("=" * 50)
    
    tests = [
        test_individual_hook_loading,
        test_mixed_config_sources,
        test_filename_type_detection,
        test_real_directory_loading,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Directory Tests: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All directory loading tests passed!")
        return 0
    else:
        print("❌ Some directory loading tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())