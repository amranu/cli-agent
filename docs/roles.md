# Agent Roles System

The CLI Agent supports a flexible role system that allows customizing agent behavior and available tools through YAML configuration files.

## Configuration Location

Role files are stored in `~/.config/agent/roles/`:
- `{role_name}.yaml` - Role definition with instructions and behavior
- `{role_name}_tools` - Optional tool list for the role (one tool pattern per line)

## Role YAML Format

```yaml
name: "Role Name"
description: "Brief description of the role"
agent_role: "You are a [role description]. You are responsible for..."
instructions: |
  **Specific Instructions:**
  - Detailed behavior guidelines
  - Tool usage patterns
  - Task-specific guidance
  
  **Dynamic Placeholders:**
  - {{TOOLS}} - Available tools list
  - {{MODEL_NAME}} - Current model name
  - {{AGENT_TYPE}} - "main" or "subagent"
  - {{LLM_INSTRUCTIONS}} - Model-specific instructions
```

## Tool File Format

Tool files control which tools are available to each role:

```bash
# Tool patterns for role (one per line)
# Comments start with #

# Exact tool names
builtin:read_file
builtin:write_file

# Wildcard patterns
builtin:*file*
*search*

# Partial matches
grep
bash

# MCP tools
mcp__*
```

## Usage

### CLI Usage
```bash
# Use role in interactive chat
python agent.py chat --role security-expert

# Use role for one-off tasks  
python agent.py ask --role security-expert "Analyze this code for vulnerabilities"
```

### Subagent Usage
```bash
# Spawn subagent with specific role
/task role=security-expert description="Security audit" prompt="Check for vulnerabilities in the codebase"
```

## Pattern Matching

Tool patterns are matched in the following order:
1. `fnmatch` against tool_key (e.g., `"builtin:read_file"`)
2. `fnmatch` against tool_name (e.g., `"read_file"`)
3. Substring match in tool_key
4. Substring match in tool_name

If no tool file exists for a role, all available tools are accessible.

## Dynamic Placeholders

Role YAML files support dynamic content insertion:

- **`{{TOOLS}}`** - Inserts formatted list of available tools (filtered by role if tool file exists)
- **`{{MODEL_NAME}}`** - Current runtime model name
- **`{{AGENT_TYPE}}`** - "main" for main agents, "subagent" for subagents
- **`{{LLM_INSTRUCTIONS}}`** - Model-specific instructions from the LLM provider

## Example Role Configuration

### Security Expert Role (`security-expert.yaml`)
```yaml
name: "Security Expert"
description: "Specialized in security analysis and defensive coding practices"
agent_role: "You are a cybersecurity expert running on {{MODEL_NAME}} focused exclusively on defensive security analysis."
instructions: |
  **Security Focus:**
  - Prioritize security best practices and defensive measures only
  - Look for common vulnerabilities (injection, XSS, CSRF, etc.)
  - Analyze code for security anti-patterns and weaknesses
  
  **Available Security Tools:**
  {{TOOLS}}
  
  **Agent Type:** {{AGENT_TYPE}} agent focused on security analysis
  
  {{LLM_INSTRUCTIONS}}
```

### Security Expert Tools (`security-expert_tools`)
```bash
# Security Expert Role - Available Tools
builtin:read_file
builtin:list_directory
builtin:grep
builtin:glob
builtin:bash_execute
builtin:webfetch
builtin:websearch
builtin:task
builtin:todo_read
builtin:todo_write
builtin:emit_result
```

### Researcher Role (`researcher.yaml`)
```yaml
name: "Researcher"
description: "Comprehensive research and analysis with web access and detailed reporting"
agent_role: "You are a thorough research specialist running on {{MODEL_NAME}}. You are responsible for conducting comprehensive research on any given topic and producing detailed, well-sourced reports."
instructions: |
  **Step-by-Step Research Process:**
  1. **Initial Discovery**: Use `websearch` with broad search terms to identify relevant sources
  2. **Source Evaluation**: Review search results to identify the most authoritative sources
  3. **Deep Reading**: Use `webfetch` to read each promising source in full detail
  4. **Iterative Searching**: Based on findings, use `websearch` with more specific terms
  5. **Cross-Verification**: Use `webfetch` on additional sources to verify key claims
  6. **Gap Analysis**: Identify information gaps and search for additional sources
  
  **Tool Usage Guidelines:**
  - **websearch**: Use for discovering sources with multiple search queries
  - **webfetch**: Use to read full content of specific URLs found through websearch
  - **todo_write**: Track research progress, source URLs, and key findings
  - **write_file**: Save detailed notes and organize research findings
  
  **Research Workflow Example:**
  1. Use `websearch` with "AI development tools market 2024" 
  2. Review results and identify 5-10 authoritative source URLs
  3. Use `webfetch` on each URL to read full articles/reports
  4. Use `websearch` with more specific terms based on initial findings
  5. Continue `webfetch` cycle until comprehensive coverage achieved
  6. Compile comprehensive report with all findings and sources
  
  **Available Research Tools:**
  {{TOOLS}}
  
  **Report Requirements:**
  - Provide verbose, comprehensive reports that thoroughly cover the subject
  - Structure information logically with clear sections and subsections
  - Include specific details, data points, statistics, and examples
  - Cite sources and provide context for where information was found
  
  {{LLM_INSTRUCTIONS}}
```

### Researcher Tools (`researcher_tools`)
```bash
# Web research capabilities
builtin:webfetch
builtin:websearch

# File operations for saving research
builtin:read_file
builtin:write_file
builtin:list_directory

# Task management and delegation
builtin:todo_read
builtin:todo_write
builtin:task

# Result reporting
builtin:emit_result
```

## Role Priority

The system follows this priority order:
1. **User roles** in `~/.config/agent/roles/` (highest priority)
2. **Default roles** shipped with the project in `cli_agent/core/default_roles/`
3. If no role file exists, all tools are available

Tool files follow the same priority order.

## Default Roles

The system ships with several built-in roles:

- **`developer`** - General software development with full tool access
- **`researcher`** - Comprehensive web research with detailed reporting
- **`analyst`** - Code and system analysis with read-focused tools  
- **`security-expert`** - Security analysis and defensive practices
- **`subagent`** - Default role for subagents with focused tool set
- **`example-dynamic`** - Demonstrates all placeholder features

## Customizing Roles

To customize a default role:
1. Copy the role files from `cli_agent/core/default_roles/` to `~/.config/agent/roles/`
2. Modify the YAML and tool files as needed
3. Your customized version will take precedence over the default

Example:
```bash
# Copy security expert role for customization
cp cli_agent/core/default_roles/security-expert.yaml ~/.config/agent/roles/
cp cli_agent/core/default_roles/security-expert_tools ~/.config/agent/roles/

# Edit the files in ~/.config/agent/roles/ as needed
```