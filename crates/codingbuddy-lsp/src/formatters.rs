use crate::{Diagnostic, Severity};

/// Format a list of diagnostics into a human-readable string suitable for LLM consumption.
///
/// Returns an empty string if there are no diagnostics.
///
/// Output format:
/// ```text
/// Diagnostics found after edit:
///
/// [ERROR] file.rs:42:10 -- expected `u32`, found `String`
/// [WARN] file.rs:55:1 -- unused variable `x`
///
/// Fix these issues before proceeding.
/// ```
pub fn format_diagnostics_for_llm(diagnostics: &[Diagnostic]) -> String {
    if diagnostics.is_empty() {
        return String::new();
    }

    let mut output = String::from("Diagnostics found after edit:\n\n");

    for diag in diagnostics {
        let severity_tag = match diag.severity {
            Severity::Error => "[ERROR]",
            Severity::Warning => "[WARN]",
        };

        let location = if diag.col > 0 {
            format!("{}:{}:{}", diag.file.display(), diag.line, diag.col)
        } else if diag.line > 0 {
            format!("{}:{}", diag.file.display(), diag.line)
        } else {
            format!("{}", diag.file.display())
        };

        output.push_str(&format!("{severity_tag} {location} -- {}\n", diag.message));
    }

    output.push_str("\nFix these issues before proceeding.\n");
    output
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_format_empty() {
        let result = format_diagnostics_for_llm(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_single_error() {
        let diagnostics = vec![Diagnostic {
            file: PathBuf::from("src/lib.rs"),
            line: 42,
            col: 10,
            severity: Severity::Error,
            message: "expected `u32`, found `String`".to_string(),
        }];

        let result = format_diagnostics_for_llm(&diagnostics);
        assert!(result.contains("Diagnostics found after edit:"));
        assert!(result.contains("[ERROR] src/lib.rs:42:10 -- expected `u32`, found `String`"));
        assert!(result.contains("Fix these issues before proceeding."));
    }

    #[test]
    fn test_format_mixed_severities() {
        let diagnostics = vec![
            Diagnostic {
                file: PathBuf::from("src/lib.rs"),
                line: 42,
                col: 10,
                severity: Severity::Error,
                message: "expected `u32`, found `String`".to_string(),
            },
            Diagnostic {
                file: PathBuf::from("src/lib.rs"),
                line: 55,
                col: 1,
                severity: Severity::Warning,
                message: "unused variable `x`".to_string(),
            },
        ];

        let result = format_diagnostics_for_llm(&diagnostics);
        assert!(result.contains("[ERROR]"));
        assert!(result.contains("[WARN]"));
    }

    #[test]
    fn test_format_no_column() {
        let diagnostics = vec![Diagnostic {
            file: PathBuf::from("test.py"),
            line: 5,
            col: 0,
            severity: Severity::Error,
            message: "SyntaxError: unexpected EOF".to_string(),
        }];

        let result = format_diagnostics_for_llm(&diagnostics);
        assert!(result.contains("[ERROR] test.py:5 -- SyntaxError: unexpected EOF"));
    }

    #[test]
    fn test_format_no_line() {
        let diagnostics = vec![Diagnostic {
            file: PathBuf::from("unknown.rs"),
            line: 0,
            col: 0,
            severity: Severity::Error,
            message: "compilation failed".to_string(),
        }];

        let result = format_diagnostics_for_llm(&diagnostics);
        assert!(result.contains("[ERROR] unknown.rs -- compilation failed"));
    }
}
