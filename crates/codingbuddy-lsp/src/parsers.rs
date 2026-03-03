use std::path::PathBuf;

use crate::{Diagnostic, Severity};

/// Parse `cargo check --message-format=json` JSON lines output.
///
/// Each line is a JSON object. We look for objects with `reason: "compiler-message"`
/// containing a `message` object with `level`, `message`, and `spans`.
pub fn parse_cargo_check(output: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if value.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }

        let Some(message_obj) = value.get("message") else {
            continue;
        };

        let level = message_obj
            .get("level")
            .and_then(|l| l.as_str())
            .unwrap_or("");
        let severity = match level {
            "error" => Severity::Error,
            "warning" => Severity::Warning,
            _ => continue,
        };

        let message_text = message_obj
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();

        if message_text.is_empty() {
            continue;
        }

        // Extract the primary span for location info.
        let spans = message_obj.get("spans").and_then(|s| s.as_array());

        let (file, line_num, col) = if let Some(spans) = spans {
            // Prefer the primary span; fall back to the first span.
            let primary = spans
                .iter()
                .find(|s| s.get("is_primary").and_then(|p| p.as_bool()) == Some(true))
                .or_else(|| spans.first());

            if let Some(span) = primary {
                let file = span.get("file_name").and_then(|f| f.as_str()).unwrap_or("");
                let line = span.get("line_start").and_then(|l| l.as_u64()).unwrap_or(0) as u32;
                let col = span
                    .get("column_start")
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as u32;
                (file.to_string(), line, col)
            } else {
                (String::new(), 0, 0)
            }
        } else {
            (String::new(), 0, 0)
        };

        diagnostics.push(Diagnostic {
            file: PathBuf::from(file),
            line: line_num,
            col,
            severity,
            message: message_text,
        });
    }

    diagnostics
}

/// Parse TypeScript compiler output.
///
/// Format: `file(line,col): error TS1234: message`
pub fn parse_tsc(output: &str) -> Vec<Diagnostic> {
    let re = match regex::Regex::new(r"^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+TS\d+:\s+(.+)$") {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let mut diagnostics = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if let Some(caps) = re.captures(line) {
            let file = caps.get(1).map_or("", |m| m.as_str());
            let line_num: u32 = caps
                .get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let col: u32 = caps
                .get(3)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let severity = match caps.get(4).map_or("", |m| m.as_str()) {
                "error" => Severity::Error,
                "warning" => Severity::Warning,
                _ => continue,
            };
            let message = caps.get(5).map_or("", |m| m.as_str()).to_string();

            diagnostics.push(Diagnostic {
                file: PathBuf::from(file),
                line: line_num,
                col,
                severity,
                message,
            });
        }
    }

    diagnostics
}

/// Parse Python compile error output from `python3 -m py_compile`.
///
/// Typical format on stderr:
/// ```text
///   File "file.py", line 5
///     x = (
///         ^
/// SyntaxError: unexpected EOF while parsing
/// ```
///
/// Or the single-line format:
/// ```text
/// py_compile.PyCompileError:   File "file.py", line 5
/// ```
pub fn parse_python_compile(stderr: &str) -> Vec<Diagnostic> {
    // Match: File "path", line N
    let file_line_re = match regex::Regex::new(r#"File "(.+?)", line (\d+)"#) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    // Match: SyntaxError: ... or IndentationError: ... or similar
    let error_re = match regex::Regex::new(r"^(\w*Error):\s*(.+)$") {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let mut diagnostics = Vec::new();
    let mut current_file = String::new();
    let mut current_line: u32 = 0;

    for line in stderr.lines() {
        let trimmed = line.trim();

        if let Some(caps) = file_line_re.captures(trimmed) {
            current_file = caps.get(1).map_or("", |m| m.as_str()).to_string();
            current_line = caps
                .get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
        }

        if let Some(caps) = error_re.captures(trimmed) {
            let error_type = caps.get(1).map_or("", |m| m.as_str());
            let error_msg = caps.get(2).map_or("", |m| m.as_str());
            let message = format!("{error_type}: {error_msg}");

            if !current_file.is_empty() {
                diagnostics.push(Diagnostic {
                    file: PathBuf::from(&current_file),
                    line: current_line,
                    col: 0,
                    severity: Severity::Error,
                    message,
                });
            }
        }
    }

    diagnostics
}

/// Parse `go vet` output.
///
/// Format: `file.go:line:col: message`
/// or: `file.go:line: message` (no column)
pub fn parse_go_vet(output: &str) -> Vec<Diagnostic> {
    let re = match regex::Regex::new(r"^(.+?\.go):(\d+):(?:(\d+):)?\s+(.+)$") {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let mut diagnostics = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if let Some(caps) = re.captures(line) {
            let file = caps.get(1).map_or("", |m| m.as_str());
            let line_num: u32 = caps
                .get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let col: u32 = caps
                .get(3)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let message = caps.get(4).map_or("", |m| m.as_str()).to_string();

            diagnostics.push(Diagnostic {
                file: PathBuf::from(file),
                line: line_num,
                col,
                severity: Severity::Warning,
                message,
            });
        }
    }

    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cargo_check_json() {
        let output = r#"{"reason":"compiler-message","package_id":"foo 0.1.0","manifest_path":"/tmp/foo/Cargo.toml","target":{"kind":["lib"],"crate_types":["lib"],"name":"foo","src_path":"/tmp/foo/src/lib.rs","edition":"2021","doc":true,"doctest":true,"test":true},"message":{"rendered":"error[E0308]: mismatched types\n","children":[],"code":{"code":"E0308","explanation":null},"level":"error","message":"expected `u32`, found `String`","spans":[{"byte_end":100,"byte_start":90,"column_end":15,"column_start":10,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":42,"line_start":42,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;

        let diagnostics = parse_cargo_check(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].file, PathBuf::from("src/lib.rs"));
        assert_eq!(diagnostics[0].line, 42);
        assert_eq!(diagnostics[0].col, 10);
        assert!(matches!(diagnostics[0].severity, Severity::Error));
        assert_eq!(diagnostics[0].message, "expected `u32`, found `String`");
    }

    #[test]
    fn test_parse_cargo_check_warning() {
        let output = r#"{"reason":"compiler-message","package_id":"foo 0.1.0","manifest_path":"/tmp/foo/Cargo.toml","target":{"kind":["lib"],"crate_types":["lib"],"name":"foo","src_path":"/tmp/foo/src/lib.rs","edition":"2021","doc":true,"doctest":true,"test":true},"message":{"rendered":"warning: unused variable\n","children":[],"code":null,"level":"warning","message":"unused variable `x`","spans":[{"byte_end":50,"byte_start":49,"column_end":2,"column_start":1,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":55,"line_start":55,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;

        let diagnostics = parse_cargo_check(output);
        assert_eq!(diagnostics.len(), 1);
        assert!(matches!(diagnostics[0].severity, Severity::Warning));
        assert_eq!(diagnostics[0].message, "unused variable `x`");
        assert_eq!(diagnostics[0].line, 55);
    }

    #[test]
    fn test_parse_cargo_check_ignores_non_messages() {
        let output = r#"{"reason":"build-script-executed","package_id":"foo 0.1.0"}
{"reason":"compiler-artifact","package_id":"foo 0.1.0"}"#;

        let diagnostics = parse_cargo_check(output);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_parse_cargo_check_malformed_json() {
        let output = "this is not json\n{also not valid";
        let diagnostics = parse_cargo_check(output);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_parse_tsc_output() {
        let output =
            "src/app.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'.";
        let diagnostics = parse_tsc(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].file, PathBuf::from("src/app.ts"));
        assert_eq!(diagnostics[0].line, 10);
        assert_eq!(diagnostics[0].col, 5);
        assert!(matches!(diagnostics[0].severity, Severity::Error));
        assert_eq!(
            diagnostics[0].message,
            "Type 'string' is not assignable to type 'number'."
        );
    }

    #[test]
    fn test_parse_tsc_multiple() {
        let output = "\
src/app.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'.
src/utils.tsx(3,1): warning TS6133: 'x' is declared but its value is never read.";

        let diagnostics = parse_tsc(output);
        assert_eq!(diagnostics.len(), 2);
        assert!(matches!(diagnostics[0].severity, Severity::Error));
        assert!(matches!(diagnostics[1].severity, Severity::Warning));
    }

    #[test]
    fn test_parse_tsc_no_match() {
        let output = "Successfully compiled 5 files.";
        let diagnostics = parse_tsc(output);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_parse_python_compile() {
        let stderr = r#"  File "test.py", line 5
    x = (
        ^
SyntaxError: unexpected EOF while parsing"#;

        let diagnostics = parse_python_compile(stderr);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].file, PathBuf::from("test.py"));
        assert_eq!(diagnostics[0].line, 5);
        assert!(matches!(diagnostics[0].severity, Severity::Error));
        assert!(diagnostics[0].message.contains("SyntaxError"));
        assert!(diagnostics[0].message.contains("unexpected EOF"));
    }

    #[test]
    fn test_parse_python_compile_indentation_error() {
        let stderr = r#"  File "app.py", line 10
    def foo():
    ^
IndentationError: unexpected indent"#;

        let diagnostics = parse_python_compile(stderr);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("IndentationError"));
    }

    #[test]
    fn test_parse_python_compile_no_error() {
        let stderr = "";
        let diagnostics = parse_python_compile(stderr);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_parse_go_vet() {
        let output = "main.go:15:2: unreachable code";
        let diagnostics = parse_go_vet(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].file, PathBuf::from("main.go"));
        assert_eq!(diagnostics[0].line, 15);
        assert_eq!(diagnostics[0].col, 2);
        assert!(matches!(diagnostics[0].severity, Severity::Warning));
        assert_eq!(diagnostics[0].message, "unreachable code");
    }

    #[test]
    fn test_parse_go_vet_no_column() {
        let output = "main.go:20: unused result of fmt.Sprintf";
        let diagnostics = parse_go_vet(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].line, 20);
        assert_eq!(diagnostics[0].col, 0);
    }

    #[test]
    fn test_parse_go_vet_multiple() {
        let output = "\
main.go:15:2: unreachable code
handler.go:42:10: printf format %d has arg of wrong type";

        let diagnostics = parse_go_vet(output);
        assert_eq!(diagnostics.len(), 2);
    }

    #[test]
    fn test_parse_go_vet_empty() {
        let output = "";
        let diagnostics = parse_go_vet(output);
        assert!(diagnostics.is_empty());
    }
}
