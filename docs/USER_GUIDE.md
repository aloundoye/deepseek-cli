# DeepSeek CLI User Guide

A complete guide to using DeepSeek CLI — from first install to advanced features.

---

## Table of Contents

1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Chat Modes](#chat-modes)
4. [TUI Keyboard Shortcuts](#tui-keyboard-shortcuts)
5. [Slash Commands](#slash-commands)
6. [Tools & What the Agent Can Do](#tools--what-the-agent-can-do)
7. [How the Agent Thinks](#how-the-agent-thinks)
8. [Permissions & Safety](#permissions--safety)
9. [Hooks](#hooks)
10. [Skills](#skills)
11. [Subagents](#subagents)
12. [MCP Servers](#mcp-servers)
13. [Local ML](#local-ml)
14. [Session Management](#session-management)
15. [Code Index](#code-index)
16. [Autopilot](#autopilot)
17. [Code Review](#code-review)
18. [Configuration](#configuration)
19. [Troubleshooting](#troubleshooting)

---

## Installation

### Prebuilt binary (recommended)

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/aloutndoye/deepseek-cli/main/scripts/install.sh | bash -s -- --version latest
```

Windows (PowerShell):

```powershell
$script = Join-Path $env:TEMP "deepseek-install.ps1"
Invoke-WebRequest https://raw.githubusercontent.com/aloutndoye/deepseek-cli/main/scripts/install.ps1 -OutFile $script
& $script -Version latest
```

### Build from source

```bash
git clone https://github.com/aloutndoye/deepseek-cli.git
cd deepseek-cli
cargo build --release --bin deepseek
# Binary at ./target/release/deepseek
```

For local ML support (optional, adds ~10MB + model downloads):

```bash
cargo build --release --bin deepseek --features local-ml
```

### Verify installation

```bash
deepseek --version
deepseek --help
```

---

## Getting Started

### Set your API key

```bash
export DEEPSEEK_API_KEY="sk-..."
```

Or persist it per-project:

```bash
mkdir -p .deepseek
echo '{"llm": {"api_key": "sk-..."}}' > .deepseek/settings.local.json
```

Add `.deepseek/settings.local.json` to your `.gitignore`.

### Start chatting

```bash
deepseek chat
```

This opens the interactive TUI. Type your request and press Enter.

### One-shot commands

```bash
deepseek ask "What does this project do?"
deepseek ask "Find all TODO comments in the codebase"
deepseek plan "Refactor the authentication module"
```

### First conversation tips

- The agent automatically gathers context about your project (file tree, git status, package manifests) on the first turn — you don't need to explain your project
- For coding tasks, just describe what you want: `"Add a retry mechanism to the HTTP client"`
- For questions, ask naturally: `"How does the auth middleware work?"`
- The agent will read files, search code, and run commands as needed
- You'll be asked for approval before any write operation (edits, shell commands)

---

## Chat Modes

| Mode | Command | What it does |
|------|---------|-------------|
| **Code** (default) | `deepseek chat` | Full agent with all tools — reads, writes, runs commands |
| **Ask** | `deepseek ask "..."` | Read-only — answers questions using code search, never modifies files |
| **Context** | `/context` in chat | Read-only with focus on project structure and dependencies |

Switch modes mid-conversation with slash commands: `/code`, `/ask`, `/context`.

---

## TUI Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Enter** | Send message |
| **\\ + Enter** | Multiline input (newline without sending) |
| **Ctrl+C** | Cancel current generation |
| **Ctrl+L** | Clear screen |
| **Ctrl+R** | Reverse history search |
| **Ctrl+B** | Background current task |
| **Esc Esc** | Open rewind menu (undo to checkpoint) |
| **Alt+P** | Switch model (chat ↔ reasoner) |
| **Alt+T** | Toggle thinking mode |
| **Shift+Tab** | Cycle permission mode (ask → auto → locked) |
| **Tab** | Accept autocomplete / ghost text suggestion |
| **Alt+Right** | Accept one word of ghost text |
| **Esc** | Dismiss autocomplete / ghost text |
| **Ctrl+V** | Paste (includes image paste support) |

### Input shortcuts

| Input | Action |
|-------|--------|
| `@path` | File autocomplete — type `@src/` to browse files |
| `!command` | Direct shell execution — `!cargo test` runs without LLM |

---

## Slash Commands

Type these in the chat input:

### Mode & Profile

| Command | Description |
|---------|-------------|
| `/ask` | Switch to ask (read-only) mode |
| `/code` | Switch to code (full agent) mode |
| `/architect` | Switch to architect mode |
| `/chat-mode` | Show/change current mode |
| `/model` | Interactive model selector |
| `/settings` | Open settings |

### Context & Workspace

| Command | Description |
|---------|-------------|
| `/add <path>` | Add file to context |
| `/drop <path>` | Remove file from context |
| `/read-only <path>` | Add file as read-only reference |
| `/map` | Show repo map (ranked file listing) |
| `/map-refresh` | Rebuild repo map |
| `/context` | Show context window usage (colored bar) |
| `/compact [focus]` | Compact conversation history (optional focus topic to preserve) |

### Git & Code

| Command | Description |
|---------|-------------|
| `/stage` | Stage changes |
| `/unstage` | Unstage changes |
| `/diff` | Show current diff |
| `/commit` | Commit staged changes (interactive message) |
| `/undo` | Undo last commit |
| `/git <args>` | Git passthrough |
| `/run <cmd>` | Run a command |
| `/test` | Run project tests |
| `/lint` | Run project linter |

### Session

| Command | Description |
|---------|-------------|
| `/load` | Load a saved session |
| `/save` | Save current session |
| `/paste` | Paste from clipboard |
| `/copy` | Copy last response |
| `/stats` | Show session statistics |
| `/memory` | Open memory file in editor |
| `/rename <name>` | Rename current session |
| `/theme` | Cycle color theme |

### Integrations

| Command | Description |
|---------|-------------|
| `/web <query>` | Web search |
| `/voice` | Voice input |
| `/mcp` | Interactive MCP server menu |

---

## Tools & What the Agent Can Do

The agent has access to 30+ tools. You don't call these directly — the LLM decides which tools to use based on your request.

### File Operations

| Tool | What it does |
|------|-------------|
| `fs_read` | Read file contents (supports images and PDFs) |
| `fs_write` | Create or overwrite a file |
| `fs_edit` | Targeted find-and-replace edit within a file |
| `fs_glob` | Find files by pattern (e.g., `**/*.rs`) |
| `fs_grep` | Search file contents by regex (ripgrep-powered) |
| `fs_list` | List directory contents |

### Shell & System

| Tool | What it does |
|------|-------------|
| `bash_run` | Execute a shell command (with policy approval) |
| `web_search` | Search the web |
| `web_fetch` | Fetch and extract content from a URL |

### Agent-Level

| Tool | What it does |
|------|-------------|
| `user_question` | Ask you a clarifying question |
| `spawn_task` | Launch a background subagent |
| `skill` | Invoke a registered skill |
| `extended_thinking` | Switch to deep reasoning mode for complex problems |

All write operations (edits, shell commands) require your approval unless you've set auto-approve in permissions.

---

## How the Agent Thinks

DeepSeek CLI wraps every conversation in intelligence layers. You don't need to configure these — they work automatically.

### Project Awareness

On your first message, the agent gathers context about your project:
- Directory tree and file structure
- Git status, branch, and recent commits
- Package manifests (Cargo.toml, package.json, go.mod, etc.)
- README excerpt
- Most architecturally important files (by dependency analysis)

This means the agent understands your project before it starts working.

### Task Complexity

Every prompt is automatically classified:

- **Simple** (e.g., "fix the typo in README") → Fast response, minimal thinking
- **Medium** (e.g., "add input validation to the form") → Reads files first, tests after
- **Complex** (e.g., "refactor the database layer to use connection pooling") → Full planning protocol: explore codebase, create a plan, implement file by file, test after each change

### Automatic Model Routing

- Simple and medium tasks use `deepseek-chat` (fast, efficient)
- Complex tasks that hit errors automatically escalate to `deepseek-reasoner` (deeper thinking, up to 64K output)
- After 3 consecutive successes, it de-escalates back to `deepseek-chat`

### Error Recovery

- If the agent hits a compile error or test failure, it gets guidance to re-read the error and reconsider its approach
- If it hits the same error 3+ times, it gets stronger guidance to try a completely different strategy
- This resets on success

### Long Conversations

When the conversation gets too long for the context window, older messages are compacted into a summary that preserves:
- Which files were modified and read
- What errors were encountered
- Key decisions that were made
- How many tool calls were made

---

## Permissions & Safety

### Permission Modes

| Mode | Behavior |
|------|----------|
| `ask` (default) | Asks for approval before every write operation |
| `auto` | Auto-approves commands matching the allowlist |
| `plan` | Shows what it would do without executing |
| `locked` | Blocks all write operations |

Cycle modes with **Shift+Tab** in the TUI, or:

```bash
deepseek --permission-mode auto chat
deepseek permissions set --approve-bash auto --approve-edits ask
```

### Command Allowlist

Commands matching the allowlist run without approval in `auto` mode:

```json
{
  "policy": {
    "allowlist": ["cargo test", "cargo build", "npm test", "git status", "git diff"]
  }
}
```

### Blocked Paths

Files matching these patterns are never read or written:

```json
{
  "policy": {
    "block_paths": [".env*", ".ssh/*", "*credentials*", "*.pem", "*.key"]
  }
}
```

### Inspect Policy

```bash
deepseek permissions show          # Show current policy
deepseek permissions dry-run bash_run  # Test what would happen
```

### Team-Managed Policies

Enterprise teams can lock settings via `.deepseek/managed-settings.json`:

```json
{
  "permission_mode": "ask",
  "max_turns": 30,
  "allowed_tools": ["fs_read", "fs_grep", "fs_glob"]
}
```

Locked settings cannot be overridden by individual users.

---

## Hooks

Hooks let you run shell commands at specific points in the agent lifecycle.

### Configuration

Add to `.deepseek/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      { "command": "echo 'Tool: $TOOL_NAME used on $FILE_PATH'" }
    ],
    "PreToolUse": [
      { "command": "my-approval-script.sh $TOOL_NAME", "once": true }
    ],
    "SessionEnd": [
      { "command": "git stash" }
    ]
  }
}
```

### Available Events

| Event | When it fires | Use case |
|-------|--------------|----------|
| `SessionStart` | Session begins | Set up environment |
| `UserPromptSubmit` | You send a message | Logging, analytics |
| `PreToolUse` | Before a tool runs | Custom approval logic |
| `PostToolUse` | After a tool succeeds | Notifications, logging |
| `PostToolUseFailure` | After a tool fails | Error tracking |
| `PermissionRequest` | Tool needs approval | Auto-approve via script |
| `PreCompact` | Before context compaction | Log what's being compacted |
| `SubagentStart` | Subagent spawns | Track background work |
| `SubagentStop` | Subagent finishes | Collect results |
| `Notification` | Agent sends notification | Custom alerts |
| `ConfigChange` | Settings change | Audit trail |
| `Stop` | Agent stops | Cleanup |
| `SessionEnd` | Session ends | Save state, cleanup |
| `TaskCompleted` | Background task done | Trigger follow-up |

### Options

- `"once": true` — Fire only once per session
- `"disabled": true` — Skip this hook without removing it
- `PermissionRequest` hooks can return a `PermissionDecision` to auto-approve or deny

---

## Skills

Skills are reusable prompt templates stored as markdown files.

### Location

Skills are discovered from:
1. `.deepseek/skills/` (project skills)
2. `~/.deepseek/skills/` (user skills)
3. Built-in skills

### Creating a Skill

Create `.deepseek/skills/review-security.md`:

```markdown
---
name: review-security
description: Security-focused code review
allowed-tools:
  - fs_read
  - fs_grep
  - fs_glob
---

Review the current codebase for security vulnerabilities:
1. Check for SQL injection, XSS, and command injection
2. Verify input validation on all public endpoints
3. Check for hardcoded secrets
4. Report findings with file paths and line numbers
```

### Using Skills

```
> /review-security
```

The LLM can also invoke skills automatically during a conversation.

### Skill Options

| Frontmatter | Description |
|-------------|-------------|
| `context: fork` | Run in an isolated subagent (separate context window) |
| `allowed-tools: [...]` | Only these tools are available |
| `disallowed-tools: [...]` | These tools are blocked |
| `disable-model-invocation: true` | Only callable via slash command, not by the LLM |

---

## Subagents

Subagents are background tasks that run independently with their own context window.

### How They Work

- The agent can spawn subagents via the `spawn_task` tool for parallel work
- Press **Ctrl+B** to background the current task
- Each subagent has independent context management and auto-compaction

### Worktree Isolation

Subagents can run in isolated git worktrees — they get a full copy of the repository and their changes don't affect your working directory until you explicitly merge them.

### Custom Agents

Define custom agents in `.deepseek/agents/`:

```markdown
---
name: security-reviewer
role: explore
read-only: true
---

You are a security-focused code reviewer. Analyze the codebase for:
- Authentication and authorization issues
- Input validation gaps
- Secret management problems
```

### Managing Subagents

```bash
deepseek agents list               # List available agents
deepseek agents show <name>        # Show agent definition
deepseek agents create <name>      # Create a new agent definition
```

---

## MCP Servers

DeepSeek CLI supports the Model Context Protocol (MCP) for connecting to external tool servers.

### Configuration

Add MCP servers to `.deepseek/settings.json`:

```json
{
  "mcp": {
    "servers": {
      "my-server": {
        "command": "npx",
        "args": ["-y", "@my-org/mcp-server"],
        "env": {
          "API_KEY": "${MY_API_KEY}"
        }
      }
    }
  }
}
```

### Supported Transports

- **stdio**: Server communicates over stdin/stdout (most common)
- **http**: Server runs as an HTTP endpoint

### Using MCP

```
> /mcp                             # Interactive server menu
```

MCP server prompts are available as slash commands. Environment variables in server config support `${VAR}` expansion.

---

## Local ML

An optional layer that runs ML models locally on your machine for code retrieval, privacy scanning, and inline completions. See `docs/LOCAL_ML_GUIDE.md` for the full guide.

### Quick Enable

```json
{
  "local_ml": {
    "enabled": true
  }
}
```

This gives you:
- **Hybrid retrieval**: Automatically searches your codebase and injects relevant code before the LLM responds
- **Privacy scanning**: Detects secrets in tool outputs and redacts them before they reach the API
- **Ghost text**: Inline code completions in the TUI (requires `--features local-ml` build for real completions)

### Privacy Commands

```bash
deepseek privacy scan              # Find sensitive files in your project
deepseek privacy policy            # Show current privacy rules
deepseek privacy redact-preview    # Preview what would be redacted
```

### Full ML (with models)

Build with `--features local-ml` to enable Candle-powered backends. Models (~300MB–1.5GB) download from HuggingFace on first use and cache locally in `~/.cache/deepseek/`.

---

## Session Management

Every conversation is automatically persisted.

```bash
deepseek run                       # Resume last session
deepseek run <session-id>          # Resume specific session
deepseek status                    # Show current session status
deepseek usage                     # Show token usage and cost
```

### Checkpoints & Rewind

- Checkpoints are created before destructive operations (file edits, shell commands)
- Press **Esc Esc** in the TUI to open the rewind picker
- Select a checkpoint to undo all changes since that point

### Replay

```bash
deepseek replay list               # List recorded sessions
deepseek replay run --session-id <id>  # Replay a session deterministically
```

---

## Code Index

DeepSeek CLI can build a local code index for fast search across your project.

```bash
deepseek index build               # Build the index
deepseek index update              # Update with recent changes
deepseek index status              # Show index stats
deepseek index watch               # Auto-rebuild on file changes
deepseek index query "search term" # Search the index directly
```

The index uses Tantivy (full-text search) and optionally the hybrid vector index (when local ML is enabled).

---

## Autopilot

Run the agent in an unattended loop with a time budget:

```bash
deepseek autopilot "Fix all failing tests" --hours 2
deepseek autopilot status --follow    # Watch progress
```

Autopilot will:
- Execute the task in a loop
- Create checkpoints before each change
- Stop after the time budget or max consecutive failures
- Never auto-commit (you decide what to keep)

Configuration:

```json
{
  "autopilot": {
    "default_max_consecutive_failures": 10,
    "heartbeat_interval_seconds": 5,
    "persist_checkpoints": true
  }
}
```

---

## Code Review

```bash
deepseek review --staged            # Review staged changes
deepseek review --diff              # Review working tree changes
deepseek review --pr 123            # Review a pull request
deepseek review --pr 123 --publish  # Post findings as PR comments
deepseek review --focus security    # Focus on specific concern
```

Review output follows a structured findings schema with severity levels. Use `--max-comments 20` to limit PR comment volume.

---

## Configuration

Configuration merges in this order (later wins):

1. `.deepseek/config.toml` (legacy)
2. `~/.deepseek/settings.json` (user — applies to all projects)
3. `.deepseek/settings.json` (project — shared with team)
4. `.deepseek/settings.local.json` (local overrides — gitignored)

### View & Edit

```bash
deepseek config show               # Show merged config (API keys redacted)
deepseek config edit               # Open config in editor
deepseek --setting-sources          # Show which files contributed each setting
```

### Common Settings

```json
{
  "llm": {
    "base_model": "deepseek-chat",
    "max_think_model": "deepseek-reasoner",
    "context_window_tokens": 128000,
    "temperature": 0.2,
    "api_key_env": "DEEPSEEK_API_KEY"
  },
  "agent_loop": {
    "tool_loop_max_turns": 50,
    "context_bootstrap_enabled": true
  },
  "policy": {
    "approve_edits": "ask",
    "approve_bash": "ask",
    "allowlist": ["cargo *", "npm test", "git status", "git diff"]
  },
  "ui": {
    "enable_tui": true,
    "reduced_motion": false
  }
}
```

See `docs/CONFIG_REFERENCE.md` for the full reference.

---

## Troubleshooting

### API Key Issues

```
Error: missing API key
```

Set `DEEPSEEK_API_KEY` in your environment or add it to `.deepseek/settings.local.json`. In interactive mode, you'll be prompted if the key is missing.

### Rate Limits / Timeouts

Increase retry settings:

```json
{
  "llm": {
    "timeout_seconds": 120,
    "max_retries": 5,
    "retry_base_ms": 1000
  }
}
```

### Agent Uses Too Many Turns

Lower the max turns:

```json
{
  "agent_loop": {
    "tool_loop_max_turns": 20
  }
}
```

### Agent Edits Wrong Files

Switch to `ask` permission mode to approve each action:

```bash
deepseek --permission-mode ask chat
```

Or block specific paths:

```json
{
  "policy": {
    "block_paths": ["src/critical-module/*", "*.prod.config"]
  }
}
```

### Context Window Full

The agent auto-compacts when context hits ~86%. You can also manually compact:

```
> /compact auth flow
```

The optional focus parameter tells the compactor to preserve context about that topic.

### TUI Not Showing

TUI requires an interactive terminal. It won't activate with:
- `--json` flag
- Piped input/output
- Non-TTY environments

Force it off: `"ui": {"enable_tui": false}`

### Local ML Issues

See `docs/LOCAL_ML_GUIDE.md` for detailed troubleshooting.

Quick fixes:
- Ghost text empty → need `--features local-ml` build for real completions
- Retrieval irrelevant → `deepseek index --hybrid clean` to rebuild
- Privacy false positives → adjust `local_ml.privacy.path_globs` in config
- Model download fails → check `~/.cache/deepseek/` permissions, falls back to mocks automatically
