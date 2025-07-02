# System Prompt Override

The CLI Agent supports customizing the system prompt through override files, allowing you to tailor the agent's behavior for different models or use cases.

## Overview

The system prompt override feature provides a flexible way to customize how the agent behaves by replacing the default system prompt with your own custom instructions. This is useful for:

- Adapting the agent's style for specific projects
- Creating specialized agent roles (e.g., code reviewer, documentation writer)
- Fine-tuning behavior for different language models
- Implementing organization-specific guidelines

## Priority Order

The system checks for override files in the following priority order:

1. **Model-specific override** (highest priority): `~/.config/agent/prompts/{provider}_{model}.txt`
2. **General override** (fallback): `~/.config/agent/prompts/system_prompt.txt`
3. **Default behavior**: Built-in system prompt with all dynamic elements

## File Locations

### General Override
- **Path**: `~/.config/agent/prompts/system_prompt.txt`
- **Usage**: Applied to all models unless a model-specific override exists
- **Purpose**: Set a consistent baseline prompt across all models

### Model-Specific Override
- **Path**: `~/.config/agent/prompts/{provider}_{model}.txt`
- **Usage**: Applied only to the specific provider-model combination
- **Purpose**: Customize behavior for specific models

#### Model-Specific File Naming Examples:
```
# Claude via Anthropic provider
anthropic_claude-3.5-sonnet.txt

# GPT-4 via OpenAI provider  
openai_gpt-4-turbo.txt
openai_gpt-4o.txt

# Claude via OpenRouter provider
openrouter_claude-3.5-sonnet.txt

# DeepSeek models via DeepSeek provider
deepseek_deepseek-chat.txt
deepseek_deepseek-reasoner.txt

# Gemini models via Google provider
google_gemini-2.5-flash.txt
google_gemini-2.5-pro.txt
```

## Dynamic Placeholders

Override files support dynamic placeholders that are replaced with runtime information:

### `{{TOOLS}}`
Replaced with the current list of available tools and their descriptions.

**Example:**
```
**Available Tools:**
{{TOOLS}}
```

**Becomes:**
```
**Available Tools:**
- **bash_execute**: Executes a given bash command in a persistent shell session
- **read_file**: Reads a file from the local filesystem
- **write_file**: Writes a file to the local filesystem
- **glob**: Retrieve files using glob pattern matching
- **grep**: Search file contents using regular expressions
...
```

### `{{LLM_INSTRUCTIONS}}`
Replaced with model-specific instructions optimized for the particular language model.

**Example:**
```
You are a helpful assistant.

{{LLM_INSTRUCTIONS}}

Focus on code quality.
```

## Setup and Usage

### 1. Create Override Directory
```bash
mkdir -p ~/.config/agent/prompts
```

### 2. Create General Override (Optional)
```bash
cat > ~/.config/agent/prompts/system_prompt.txt << 'EOF'
You are a helpful AI assistant focused on software development.

**Available Tools:**
{{TOOLS}}

{{LLM_INSTRUCTIONS}}

Guidelines:
- Write clean, maintainable code
- Provide clear explanations
- Follow best practices
EOF
```

### 3. Create Model-Specific Overrides (Optional)
```bash
# For Claude models - more conversational style
cat > ~/.config/agent/prompts/anthropic_claude-3.5-sonnet.txt << 'EOF'
You are Claude, a thoughtful AI assistant specializing in software development.

I have access to these tools:
{{TOOLS}}

{{LLM_INSTRUCTIONS}}

I focus on:
- Clean, readable code
- Thorough explanations
- Security best practices
- Performance considerations
EOF

# For GPT models - more direct style
cat > ~/.config/agent/prompts/openai_gpt-4o.txt << 'EOF'
You are GPT-4, an efficient coding assistant.

Available tools:
{{TOOLS}}

{{LLM_INSTRUCTIONS}}

Approach:
- Direct, actionable responses
- Optimized solutions
- Best practices enforcement
EOF
```

## Complete Examples

### Example 1: Documentation-Focused Assistant
```bash
cat > ~/.config/agent/prompts/system_prompt.txt << 'EOF'
You are a documentation specialist AI assistant.

Your primary role is to help create, improve, and maintain high-quality technical documentation.

{{TOOLS}}

{{LLM_INSTRUCTIONS}}

Focus areas:
- Clear, concise writing
- Proper structure and formatting
- User-friendly explanations
- Code examples with documentation
- Consistent style and terminology

Always prioritize clarity and usability in your responses.
EOF
```

### Example 2: Security-Focused Code Reviewer
```bash
cat > ~/.config/agent/prompts/system_prompt.txt << 'EOF'
You are a security-focused code review assistant.

Your mission is to identify security vulnerabilities, suggest secure coding practices, and help maintain secure codebases.

{{TOOLS}}

{{LLM_INSTRUCTIONS}}

Security priorities:
- Input validation and sanitization
- Authentication and authorization
- Secure data handling
- Dependency security
- Common vulnerability patterns (OWASP Top 10)
- Secure coding standards

Always consider security implications in your recommendations.
EOF
```

### Example 3: Model-Specific Customization
```bash
# For reasoning models like DeepSeek
cat > ~/.config/agent/prompts/deepseek_deepseek-reasoner.txt << 'EOF'
You are DeepSeek, an AI assistant with advanced reasoning capabilities.

Use your reasoning abilities to thoroughly analyze problems before providing solutions.

{{TOOLS}}

{{LLM_INSTRUCTIONS}}

Approach:
1. Analyze the problem systematically
2. Consider multiple solution approaches
3. Reason through trade-offs and implications
4. Provide well-justified recommendations
5. Explain your reasoning process

Take time to think through complex problems step by step.
EOF
```

## Troubleshooting

### Verify Override is Being Used
Check the agent logs for confirmation messages:
```
INFO - Using system prompt override: /home/user/.config/agent/prompts/system_prompt.txt
```

### Common Issues

#### Override Not Applied
- Verify file exists in the correct location
- Check file permissions (should be readable)
- Ensure correct filename format for model-specific overrides
- Check for typos in provider/model names

#### Dynamic Placeholders Not Working
- Ensure exact placeholder syntax: `{{TOOLS}}` and `{{LLM_INSTRUCTIONS}}`
- Placeholders are case-sensitive
- No spaces inside the braces

#### Finding Provider/Model Names
Use the agent with debug logging to see the exact provider and model names:
```bash
# The agent displays this information on startup
agent chat --model your-model
# Look for: "Using provider-model: provider:model"
```

## Advanced Usage

### Conditional Content
You can create overrides that work differently based on whether placeholders are replaced:

```
You are a helpful assistant.

{{TOOLS}}

{{LLM_INSTRUCTIONS}}

# If tools aren't available, the {{TOOLS}} placeholder will remain
# You can structure your prompt to handle this gracefully
```

### Project-Specific Overrides
While the system doesn't automatically support project-specific overrides, you can:

1. Create different override files for different projects
2. Symlink the appropriate override file to the standard location
3. Use a script to switch between different override configurations

### Integration with Agent.md
Override prompts work alongside the Agent.md project context system. The Agent.md content is injected into the first user message, while override prompts replace the system prompt.

## Best Practices

1. **Start with General Override**: Create a general override first, then add model-specific ones as needed

2. **Use Dynamic Placeholders**: Always include `{{TOOLS}}` to ensure the agent knows what tools are available

3. **Keep LLM Instructions**: Include `{{LLM_INSTRUCTIONS}}` to preserve model-specific optimizations

4. **Test Thoroughly**: Test your overrides with different models to ensure they work as expected

5. **Version Control**: Keep your override files in version control if they're project-specific

6. **Document Changes**: Document what customizations you've made and why

7. **Backup Defaults**: Test your changes and keep backups of working configurations

## Security Considerations

- Override files can completely change agent behavior
- Ensure override files are secure and not writable by unauthorized users
- Review override prompts to ensure they maintain security guidelines
- Be cautious with prompts that might change security-focused behavior

## Limitations

- Override files completely replace the default system prompt (when no placeholders are used)
- Agent type differentiation (main agent vs subagent) may be lost without proper prompt design
- Some dynamic behaviors may need to be manually recreated in override prompts
- Changes to override files require restarting the agent to take effect