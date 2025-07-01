"""
Hook configuration loading and validation for CLI-Agent.

This module handles loading hook configurations from various sources
and validating their structure.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of hooks supported by the system."""
    
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    NOTIFICATION = "Notification"
    STOP = "Stop"


@dataclass
class HookDefinition:
    """Definition of a single hook to execute."""
    
    type: str  # "command" (other types may be added later)
    command: str
    name: Optional[str] = None  # Hook name (from filename)
    timeout: Optional[int] = 30
    working_directory: Optional[str] = None
    env: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Validate hook definition after initialization."""
        if self.type not in ["command"]:
            raise ValueError(f"Unsupported hook type: {self.type}")
        
        if not self.command or not self.command.strip():
            raise ValueError("Hook command cannot be empty")
            
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Hook timeout must be positive")


@dataclass 
class HookMatcher:
    """Matcher configuration for determining which hooks to execute."""
    
    matcher: str  # Pattern to match against tool names
    hooks: List[HookDefinition] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate matcher after initialization."""
        if not self.matcher:
            raise ValueError("Hook matcher pattern cannot be empty")
            
        # Convert matcher pattern to regex for matching
        self._regex = self._compile_matcher_pattern(self.matcher)
    
    def _compile_matcher_pattern(self, pattern: str) -> re.Pattern:
        """Compile matcher pattern to regex."""
        if pattern == "*":
            # Match everything
            return re.compile(r".*")
        elif "|" in pattern:
            # Pipe-separated alternatives 
            alternatives = [re.escape(alt.strip()) for alt in pattern.split("|")]
            return re.compile(f"^({'|'.join(alternatives)})$")
        elif pattern.endswith("*"):
            # Prefix matching (e.g., "builtin:*")
            prefix = re.escape(pattern[:-1])
            return re.compile(f"^{prefix}.*$")
        else:
            # Exact matching
            return re.compile(f"^{re.escape(pattern)}$")
    
    def matches(self, tool_name: str) -> bool:
        """Check if this matcher matches the given tool name."""
        return bool(self._regex.match(tool_name))


@dataclass
class HookConfig:
    """Complete hook configuration loaded from files."""
    
    hooks: Dict[HookType, List[HookMatcher]] = field(default_factory=dict)
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> "HookConfig":
        """Load hook configuration from a single JSON file."""
        if not config_path.exists():
            logger.debug(f"Hook config file not found: {config_path}")
            return cls()
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return cls._parse_config_data(data, str(config_path))
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in hook config {config_path}: {e}")
            return cls()
        except Exception as e:
            logger.error(f"Error loading hook config from {config_path}: {e}")
            return cls()
    
    @classmethod
    def load_from_directory(cls, hooks_dir: Path) -> "HookConfig":
        """Load hook configurations from a directory of individual hook files."""
        if not hooks_dir.exists() or not hooks_dir.is_dir():
            logger.debug(f"Hook directory not found: {hooks_dir}")
            return cls()
            
        config = cls()
        hooks_loaded = 0
        
        # Look for hook files in the directory
        for hook_file in hooks_dir.iterdir():
            if hook_file.is_file() and hook_file.suffix in ['.json', '.yaml', '.yml']:
                try:
                    file_config = cls._load_individual_hook_file(hook_file)
                    if file_config.has_hooks():
                        config = cls._merge_configs(config, file_config)
                        hooks_loaded += 1
                        logger.debug(f"Loaded hook from: {hook_file}")
                except Exception as e:
                    logger.warning(f"Error loading hook file {hook_file}: {e}")
                    continue
        
        if hooks_loaded > 0:
            logger.debug(f"Loaded {hooks_loaded} hook files from {hooks_dir}")
            
        return config
    
    @classmethod
    def _load_individual_hook_file(cls, hook_file: Path) -> "HookConfig":
        """Load a single hook file (JSON or YAML)."""
        try:
            with open(hook_file, 'r', encoding='utf-8') as f:
                if hook_file.suffix == '.json':
                    data = json.load(f)
                elif hook_file.suffix in ['.yaml', '.yml']:
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                    except ImportError:
                        logger.warning(f"YAML support not available, skipping {hook_file}")
                        return cls()
                else:
                    logger.warning(f"Unsupported file format: {hook_file}")
                    return cls()
                    
            # Handle both full config format and simplified individual hook format
            if "hooks" in data:
                # Full format: {"hooks": {"PreToolUse": [...]}}
                return cls._parse_config_data(data, str(hook_file))
            else:
                # Simplified format: auto-detect hook type from filename or content
                return cls._parse_individual_hook(data, hook_file)
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in hook file {hook_file}: {e}")
            return cls()
        except Exception as e:
            logger.error(f"Error loading hook file {hook_file}: {e}")
            return cls()
    
    @classmethod
    def _parse_individual_hook(cls, data: Dict[str, Any], hook_file: Path) -> "HookConfig":
        """Parse an individual hook file with simplified format."""
        config = cls()
        
        # Try to determine hook type from filename
        filename_lower = hook_file.stem.lower()
        hook_type = None
        
        if any(x in filename_lower for x in ['pre', 'before']):
            hook_type = HookType.PRE_TOOL_USE
        elif any(x in filename_lower for x in ['post', 'after']):
            hook_type = HookType.POST_TOOL_USE
        elif any(x in filename_lower for x in ['notification', 'notify']):
            hook_type = HookType.NOTIFICATION
        elif any(x in filename_lower for x in ['stop', 'end', 'finish']):
            hook_type = HookType.STOP
        elif 'type' in data:
            # Explicit type in the file
            try:
                hook_type = HookType(data['type'])
            except ValueError:
                logger.warning(f"Unknown hook type '{data['type']}' in {hook_file}")
                return config
        else:
            logger.warning(f"Could not determine hook type for {hook_file}")
            return config
        
        # Parse the hook definition
        try:
            # Handle different formats
            if 'matcher' in data and 'hooks' in data:
                # Format: {"matcher": "...", "hooks": [...]}
                matcher = cls._parse_matcher(data, str(hook_file))
                if matcher:
                    config.hooks[hook_type] = [matcher]
            elif 'command' in data:
                # Simple format: {"command": "...", "matcher": "...", ...}
                matcher_pattern = data.get('matcher', '*')
                
                # Extract hook name from filename (remove type prefix)
                hook_name = hook_file.stem
                for prefix in ['pre-', 'post-', 'notify-', 'notification-', 'stop-', 'before-', 'after-', 'end-', 'finish-']:
                    if hook_name.lower().startswith(prefix):
                        hook_name = hook_name[len(prefix):]
                        break
                
                # Create hook definition from simple format
                hook_def = HookDefinition(
                    type="command",
                    command=data['command'],
                    name=hook_name,
                    timeout=data.get('timeout'),
                    working_directory=data.get('working_directory'),
                    env=data.get('env')
                )
                
                matcher = HookMatcher(matcher=matcher_pattern, hooks=[hook_def])
                config.hooks[hook_type] = [matcher]
            else:
                logger.warning(f"Invalid hook format in {hook_file}")
                
        except Exception as e:
            logger.warning(f"Error parsing individual hook {hook_file}: {e}")
            
        return config
    
    @classmethod
    def load_from_multiple_sources(cls) -> "HookConfig":
        """Load and merge hook configurations from all standard sources."""
        # Traditional single file sources
        config_sources = [
            Path.home() / ".config" / "agent" / "settings.json", 
            Path.home() / ".config" / "agent" / "settings.local.json",
        ]
        
        # Hook directory sources  
        hook_directories = [
            Path.home() / ".config" / "agent" / "hooks",
        ]
        
        merged_config = cls()
        sources_loaded = 0
        
        # Load from traditional config files
        for config_path in config_sources:
            if config_path.exists():
                logger.debug(f"Loading hook config from: {config_path}")
                file_config = cls.load_from_file(config_path)
                merged_config = cls._merge_configs(merged_config, file_config)
                sources_loaded += 1
            else:
                logger.debug(f"Hook config file not found: {config_path}")
        
        # Load from hook directories
        for hooks_dir in hook_directories:
            if hooks_dir.exists() and hooks_dir.is_dir():
                logger.debug(f"Loading hooks from directory: {hooks_dir}")
                dir_config = cls.load_from_directory(hooks_dir)
                if dir_config.has_hooks():
                    merged_config = cls._merge_configs(merged_config, dir_config)
                    sources_loaded += 1
                
        if merged_config.has_hooks():
            logger.info(f"Loaded hooks configuration from {sources_loaded} sources")
        else:
            logger.debug("No hook configurations found")
            
        return merged_config
    
    @classmethod
    def _parse_config_data(cls, data: Dict[str, Any], source: str) -> "HookConfig":
        """Parse hook configuration data from JSON."""
        config = cls()
        
        hooks_data = data.get("hooks", {})
        if not isinstance(hooks_data, dict):
            logger.warning(f"Invalid hooks section in {source}: expected object")
            return config
            
        for hook_type_str, matchers_data in hooks_data.items():
            try:
                hook_type = HookType(hook_type_str)
            except ValueError:
                logger.warning(f"Unknown hook type '{hook_type_str}' in {source}")
                continue
                
            if not isinstance(matchers_data, list):
                logger.warning(f"Invalid matchers for {hook_type_str} in {source}: expected array")
                continue
                
            matchers = []
            for matcher_data in matchers_data:
                try:
                    matcher = cls._parse_matcher(matcher_data, source)
                    if matcher:
                        matchers.append(matcher)
                except Exception as e:
                    logger.warning(f"Error parsing matcher in {source}: {e}")
                    continue
                    
            if matchers:
                config.hooks[hook_type] = matchers
                
        return config
    
    @classmethod
    def _parse_matcher(cls, matcher_data: Dict[str, Any], source: str) -> Optional[HookMatcher]:
        """Parse a single matcher configuration."""
        if not isinstance(matcher_data, dict):
            raise ValueError("Matcher must be an object")
            
        pattern = matcher_data.get("matcher")
        if not pattern or not isinstance(pattern, str):
            raise ValueError("Matcher must have a 'matcher' string field")
            
        hooks_data = matcher_data.get("hooks", [])
        if not isinstance(hooks_data, list):
            raise ValueError("Matcher 'hooks' must be an array")
            
        hooks = []
        for hook_data in hooks_data:
            try:
                hook = cls._parse_hook_definition(hook_data, source)
                if hook:
                    hooks.append(hook)
            except Exception as e:
                logger.warning(f"Error parsing hook definition in {source}: {e}")
                continue
                
        if not hooks:
            logger.debug(f"No valid hooks found for matcher '{pattern}' in {source}")
            return None
            
        return HookMatcher(matcher=pattern, hooks=hooks)
    
    @classmethod
    def _parse_hook_definition(cls, hook_data: Dict[str, Any], source: str) -> Optional[HookDefinition]:
        """Parse a single hook definition."""
        if not isinstance(hook_data, dict):
            raise ValueError("Hook definition must be an object")
            
        hook_type = hook_data.get("type")
        if not hook_type or not isinstance(hook_type, str):
            raise ValueError("Hook definition must have a 'type' string field")
            
        command = hook_data.get("command")
        if not command or not isinstance(command, str):
            raise ValueError("Hook definition must have a 'command' string field")
            
        timeout = hook_data.get("timeout")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            raise ValueError("Hook timeout must be a positive integer")
            
        working_directory = hook_data.get("working_directory")
        if working_directory is not None and not isinstance(working_directory, str):
            raise ValueError("Hook working_directory must be a string")
            
        env = hook_data.get("env")
        if env is not None and not isinstance(env, dict):
            raise ValueError("Hook env must be an object")
            
        return HookDefinition(
            type=hook_type,
            command=command,
            timeout=timeout,
            working_directory=working_directory,
            env=env
        )
    
    @classmethod
    def _merge_configs(cls, base: "HookConfig", overlay: "HookConfig") -> "HookConfig":
        """Merge two hook configurations, with overlay taking precedence."""
        merged = cls()
        
        # Start with base config
        for hook_type, matchers in base.hooks.items():
            merged.hooks[hook_type] = list(matchers)
            
        # Add overlay config
        for hook_type, matchers in overlay.hooks.items():
            if hook_type in merged.hooks:
                merged.hooks[hook_type].extend(matchers)
            else:
                merged.hooks[hook_type] = list(matchers)
                
        return merged
    
    def has_hooks(self) -> bool:
        """Check if this configuration has any hooks defined."""
        return bool(self.hooks)
    
    def get_hooks_for_type(self, hook_type: HookType) -> List[HookMatcher]:
        """Get all matchers for a specific hook type."""
        return self.hooks.get(hook_type, [])
    
    def get_matching_hooks(self, hook_type: HookType, tool_name: str) -> List[HookDefinition]:
        """Get all hook definitions that match the given hook type and tool name."""
        matching_hooks = []
        
        for matcher in self.get_hooks_for_type(hook_type):
            if matcher.matches(tool_name):
                matching_hooks.extend(matcher.hooks)
                
        return matching_hooks
    
    def validate(self) -> List[str]:
        """Validate the configuration and return list of error messages."""
        errors = []
        
        for hook_type, matchers in self.hooks.items():
            for i, matcher in enumerate(matchers):
                try:
                    # Test matcher pattern compilation
                    test_pattern = matcher._regex
                    test_pattern.match("test")
                except Exception as e:
                    errors.append(f"Invalid matcher pattern in {hook_type.value}[{i}]: {e}")
                    
                for j, hook in enumerate(matcher.hooks):
                    if not hook.command.strip():
                        errors.append(f"Empty command in {hook_type.value}[{i}].hooks[{j}]")
                        
        return errors