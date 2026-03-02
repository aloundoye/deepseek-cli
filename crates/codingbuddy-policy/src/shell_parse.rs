//! Tree-sitter based bash command analysis for security enforcement.
//!
//! Replaces regex-based pattern matching with proper AST analysis to detect
//! dangerous shell constructs: command chaining, subshells, redirections,
//! process substitution, here-strings, and background execution.

use std::cell::RefCell;
use tree_sitter::{Parser, Tree};

thread_local! {
    /// Reuse a single tree-sitter parser per thread to avoid repeated
    /// allocation and language-table initialization on every call.
    static BASH_PARSER: RefCell<Parser> = RefCell::new({
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_bash::LANGUAGE.into())
            .expect("tree-sitter-bash language should load");
        parser
    });
}

/// Parse a shell command string into a tree-sitter AST.
/// Returns `None` if parsing fails.
fn parse_bash(cmd: &str) -> Option<Tree> {
    BASH_PARSER.with(|p| p.borrow_mut().parse(cmd, None))
}

/// Forbidden AST node types that indicate dangerous shell constructs.
/// These are checked recursively throughout the parse tree.
const FORBIDDEN_NODE_KINDS: &[&str] = &[
    "command_substitution", // $(cmd) or `cmd`
    "process_substitution", // <(cmd) or >(cmd)
    "herestring_redirect",  // <<< 'string'
];

/// Detect dangerous shell constructs using tree-sitter AST analysis.
///
/// Returns `true` if the command contains any of:
/// - Command chaining (`;`, `&&`, `||`)
/// - Command substitution (`$(...)`, backticks)
/// - Process substitution (`<(...)`, `>(...)`)
/// - Here-strings (`<<<`)
/// - Background execution (trailing `&`)
/// - Output/input redirection (`>`, `>>`, `<`) outside of quotes
///
/// Falls back to the regex-based detector if tree-sitter parsing fails.
pub fn contains_forbidden_constructs(cmd: &str) -> bool {
    let tree = match parse_bash(cmd) {
        Some(t) => t,
        None => return contains_forbidden_constructs_regex(cmd),
    };
    let root = tree.root_node();

    // Check for ERROR nodes — if the parser couldn't fully understand the command,
    // fall back to regex for safety (fail-closed).
    if root.has_error() {
        return contains_forbidden_constructs_regex(cmd);
    }

    // Walk the entire AST looking for forbidden node types
    let mut cursor = root.walk();
    if walk_for_forbidden(&mut cursor) {
        return true;
    }

    // Check for command chaining and background execution at the top level.
    // In bash tree-sitter grammar, a `list` node with `;`, `&&`, `||` operators
    // or a trailing `&` represents command chaining/backgrounding.
    check_top_level_chaining(&root)
}

/// Recursively walk the AST looking for forbidden node types and redirections.
fn walk_for_forbidden(cursor: &mut tree_sitter::TreeCursor) -> bool {
    loop {
        let node = cursor.node();
        let kind = node.kind();

        // Check forbidden node types
        if FORBIDDEN_NODE_KINDS.contains(&kind) {
            return true;
        }

        // Check for redirections — `file_redirect` or `heredoc_redirect` nodes
        // indicate I/O redirection (`>`, `>>`, `<`, `<<`).
        // We allow heredoc_body but not the redirect itself for security.
        if kind == "file_redirect" || kind == "heredoc_redirect" {
            return true;
        }

        // Recurse into children
        if cursor.goto_first_child() {
            if walk_for_forbidden(cursor) {
                return true;
            }
            cursor.goto_parent();
        }

        if !cursor.goto_next_sibling() {
            break;
        }
    }
    false
}

/// Check if the root program node contains command chaining or backgrounding.
///
/// In the bash grammar:
/// - A `list` node represents chained commands (`;`, `&&`, `||`, `&`)
/// - Multiple top-level statements indicate implicit chaining via newlines
fn check_top_level_chaining(root: &tree_sitter::Node) -> bool {
    // If root has a `list` child, that means chaining operators are present
    let mut cursor = root.walk();
    if cursor.goto_first_child() {
        loop {
            let node = cursor.node();

            // `list` node means `;`, `&&`, `||`, or `&` was used
            if node.kind() == "list" {
                return true;
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    // Check for multiple top-level statements (implicit chaining via newlines
    // is allowed in scripts but we want single commands for tool execution).
    // A single piped command (`cmd1 | cmd2`) is a single `pipeline` node, so
    // that's fine. We only block if there are 2+ top-level *statement* nodes.
    let statement_count = (0..root.child_count())
        .filter(|i| {
            let child = root.child(*i).unwrap();
            !child.is_extra() && child.kind() != "\n"
        })
        .count();

    statement_count > 1
}

/// Detect shell redirection operators outside of quoted strings.
///
/// This is the legacy regex-based approach, used as a fallback when
/// tree-sitter parsing fails.
pub fn has_redirection_operator(cmd: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_char = '\0';
    let bytes = cmd.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            '\'' if !in_double && prev_char != '\\' => in_single = !in_single,
            '"' if !in_single && prev_char != '\\' => in_double = !in_double,
            '>' | '<' if !in_single && !in_double => return true,
            _ => {}
        }
        prev_char = ch;
        i += 1;
    }
    false
}

/// Legacy regex-based detection of forbidden shell tokens.
/// Used as fallback when tree-sitter parsing fails (fail-closed).
fn contains_forbidden_constructs_regex(cmd: &str) -> bool {
    let forbidden = [
        "\n", "\r", ";", "&&", "||", // Command chaining
        "`", "$(", // Subshell / command substitution
        "<(", ">(",  // Process substitution
        "<<<", // Here-string
    ];
    if forbidden.iter().any(|needle| cmd.contains(needle)) {
        return true;
    }

    // Background execution: trailing `&` (but not `&&` which is already checked)
    let trimmed = cmd.trim();
    if trimmed.ends_with('&') && !trimmed.ends_with("&&") {
        return true;
    }

    has_redirection_operator(trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Simple commands (should pass) ──

    #[test]
    fn simple_command_allowed() {
        assert!(!contains_forbidden_constructs("ls -la"));
        assert!(!contains_forbidden_constructs("echo hello"));
        assert!(!contains_forbidden_constructs("cargo test --workspace"));
    }

    #[test]
    fn pipeline_allowed() {
        assert!(!contains_forbidden_constructs("ls | grep foo"));
        assert!(!contains_forbidden_constructs(
            "cat file.txt | head -20 | wc -l"
        ));
    }

    #[test]
    fn single_command_with_args_allowed() {
        assert!(!contains_forbidden_constructs("git log --oneline -10"));
        assert!(!contains_forbidden_constructs("npm install express"));
    }

    // ── Dangerous constructs (should block) ──

    #[test]
    fn blocks_command_chaining_semicolon() {
        assert!(contains_forbidden_constructs("echo a; echo b"));
    }

    #[test]
    fn blocks_command_chaining_and() {
        assert!(contains_forbidden_constructs("make && make install"));
    }

    #[test]
    fn blocks_command_chaining_or() {
        assert!(contains_forbidden_constructs("test -f x || exit 1"));
    }

    #[test]
    fn blocks_command_substitution_dollar() {
        assert!(contains_forbidden_constructs("echo $(whoami)"));
    }

    #[test]
    fn blocks_command_substitution_backtick() {
        assert!(contains_forbidden_constructs("echo `whoami`"));
    }

    #[test]
    fn blocks_process_substitution() {
        assert!(contains_forbidden_constructs("diff <(ls /a) <(ls /b)"));
        assert!(contains_forbidden_constructs("tee >(grep foo)"));
    }

    #[test]
    fn blocks_here_string() {
        assert!(contains_forbidden_constructs("cat <<< 'hello'"));
    }

    #[test]
    fn blocks_output_redirection() {
        assert!(contains_forbidden_constructs("echo hello > /tmp/out"));
        assert!(contains_forbidden_constructs("echo hello >> /tmp/out"));
    }

    #[test]
    fn blocks_input_redirection() {
        assert!(contains_forbidden_constructs("cat < /etc/passwd"));
    }

    #[test]
    fn blocks_background_execution() {
        assert!(contains_forbidden_constructs("sleep 100 &"));
    }

    // ── Edge cases ──

    #[test]
    fn quoted_angle_brackets_not_blocked() {
        // Inside quotes, > and < are not redirections
        assert!(!contains_forbidden_constructs("awk '{if ($1 > 5) print}'"));
        assert!(!contains_forbidden_constructs(r#"echo "a > b""#));
    }

    #[test]
    fn grep_with_pipe_allowed() {
        assert!(!contains_forbidden_constructs(
            "grep -r 'pattern' src/ | head -20"
        ));
    }

    #[test]
    fn command_with_flags_and_quotes_allowed() {
        assert!(!contains_forbidden_constructs(r#"git commit -m "fix bug""#));
    }

    #[test]
    fn complex_pipeline_allowed() {
        assert!(!contains_forbidden_constructs(
            "find . -name '*.rs' | xargs grep 'TODO' | wc -l"
        ));
    }

    // ── Regression tests matching existing policy tests ──

    #[test]
    fn regression_blocks_same_as_policy() {
        // These should all be blocked, matching the behavior from policy tests
        assert!(contains_forbidden_constructs("echo hello > /tmp/out"));
        assert!(contains_forbidden_constructs("echo hello >> /tmp/out"));
        assert!(contains_forbidden_constructs("cat < /etc/passwd"));
        assert!(contains_forbidden_constructs("sleep 100 &"));
        assert!(contains_forbidden_constructs("diff <(ls /a) <(ls /b)"));
        assert!(contains_forbidden_constructs("tee >(grep foo)"));
        assert!(contains_forbidden_constructs("cat <<< 'hello'"));
    }
}
