# Tab Completion for /switch Command

The `/switch` command now supports intelligent tab completion for provider:model combinations.

## Usage Examples

### 1. Show all available combinations
```bash
/switch <TAB>
```
This will show all available provider:model combinations like:
- `anthropic:claude-3-5-sonnet-20241022`
- `openai:gpt-4-turbo-preview`
- `deepseek:deepseek-chat`
- `google:gemini-2.5-flash`

### 2. Complete provider names
```bash
/switch anth<TAB>
```
This will complete to:
- `anthropic:` 
- `anthropic:claude-3-5-sonnet-20241022`
- `anthropic:claude-3-5-haiku-20241022`
- `anthropic:claude-3-opus-20240229`

### 3. Complete model names within a provider
```bash
/switch anthropic:cl<TAB>
```
This will complete to Claude models:
- `anthropic:claude-3-5-sonnet-20241022`
- `anthropic:claude-3-5-haiku-20241022`
- `anthropic:claude-3-opus-20240229`

### 4. Complete OpenAI models
```bash
/switch openai:gp<TAB>
```
This will complete to GPT models:
- `openai:gpt-4-turbo-preview`
- `openai:gpt-4`
- `openai:gpt-3.5-turbo`

### 5. Complete DeepSeek
```bash
/switch dee<TAB>
```
This will complete to:
- `deepseek:`
- `deepseek:deepseek-chat`
- `deepseek:deepseek-reasoner`

## How It Works

The tab completion system:

1. **Detects context**: Recognizes when you're typing a `/switch` command
2. **Fetches available models**: Uses the config system to get current provider-model combinations
3. **Intelligent filtering**: Shows relevant completions based on what you've typed
4. **Supports partial matching**: Works for both provider and model parts of the combination

## Fallback Behavior

If the dynamic model fetching fails, the system falls back to a predefined list of common models for each provider, ensuring tab completion always works.

## Interactive Demo

To try this out:
1. Start the CLI agent
2. Type `/switch ` and press TAB to see all options
3. Type `/switch anth` and press TAB to see Anthropic options
4. Type `/switch anthropic:` and press TAB to see Claude models