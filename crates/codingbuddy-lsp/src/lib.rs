pub mod formatters;
pub mod parsers;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for the post-edit LSP validator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspConfig {
    /// Global enable/disable toggle.
    pub enabled: bool,
    /// Per-language enable/disable. Keys are language names (e.g., "rust", "typescript").
    /// If a language is absent from the map, it defaults to enabled.
    pub languages: HashMap<String, bool>,
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            languages: HashMap::new(),
        }
    }
}

impl LspConfig {
    /// Check whether validation is enabled for a given language.
    /// Returns `false` if the global toggle is off, or if the language is explicitly disabled.
    /// Languages not present in the map are considered enabled.
    pub fn is_language_enabled(&self, language: &str) -> bool {
        if !self.enabled {
            return false;
        }
        *self.languages.get(language).unwrap_or(&true)
    }
}

/// Severity level for a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
}

/// A single diagnostic produced by a language tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// The file path the diagnostic refers to.
    pub file: PathBuf,
    /// Line number (1-based). 0 means unknown.
    pub line: u32,
    /// Column number (1-based). 0 means unknown.
    pub col: u32,
    /// Severity of the diagnostic.
    pub severity: Severity,
    /// Human-readable message.
    pub message: String,
}

/// Post-edit validator that runs language-specific checks on edited files.
pub struct EditValidator {
    /// Workspace root directory (used for Rust `cargo check`, etc.).
    pub workspace: PathBuf,
    /// Configuration controlling which languages are validated.
    pub config: LspConfig,
}

impl EditValidator {
    /// Create a new `EditValidator`.
    pub fn new(workspace: PathBuf, config: LspConfig) -> Self {
        Self { workspace, config }
    }

    /// Check a single file for diagnostics based on its extension.
    ///
    /// Returns an empty vec if the language is disabled, the tool is not installed,
    /// or the file extension is not recognized.
    pub fn check_file(&self, path: &Path) -> Result<Vec<Diagnostic>> {
        let Some(language) = detect_language(path) else {
            return Ok(Vec::new());
        };

        if !self.config.is_language_enabled(language) {
            return Ok(Vec::new());
        }

        match language {
            "rust" => self.check_rust(),
            "typescript" => self.check_typescript(path),
            "python" => self.check_python(path),
            "go" => self.check_go(path),
            _ => Ok(Vec::new()),
        }
    }

    /// Format diagnostics for LLM consumption. Delegates to the formatters module.
    pub fn format_for_llm(diagnostics: &[Diagnostic]) -> String {
        formatters::format_diagnostics_for_llm(diagnostics)
    }

    /// Run `cargo check --message-format=json` in the workspace directory.
    fn check_rust(&self) -> Result<Vec<Diagnostic>> {
        if !is_command_available("cargo") {
            return Ok(Vec::new());
        }

        let output = Command::new("cargo")
            .arg("check")
            .arg("--message-format=json")
            .current_dir(&self.workspace)
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                Ok(parsers::parse_cargo_check(&stdout))
            }
            Err(_) => Ok(Vec::new()),
        }
    }

    /// Run `tsc --noEmit --pretty false` on a TypeScript file.
    fn check_typescript(&self, path: &Path) -> Result<Vec<Diagnostic>> {
        if !is_command_available("tsc") {
            return Ok(Vec::new());
        }

        let output = Command::new("tsc")
            .arg("--noEmit")
            .arg("--pretty")
            .arg("false")
            .arg(path)
            .current_dir(&self.workspace)
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let combined = format!("{stdout}\n{stderr}");
                Ok(parsers::parse_tsc(&combined))
            }
            Err(_) => Ok(Vec::new()),
        }
    }

    /// Run `python3 -m py_compile <file>` on a Python file.
    fn check_python(&self, path: &Path) -> Result<Vec<Diagnostic>> {
        if !is_command_available("python3") {
            return Ok(Vec::new());
        }

        let output = Command::new("python3")
            .arg("-m")
            .arg("py_compile")
            .arg(path)
            .current_dir(&self.workspace)
            .output();

        match output {
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Ok(parsers::parse_python_compile(&stderr))
            }
            Err(_) => Ok(Vec::new()),
        }
    }

    /// Run `go vet` on the package containing the edited Go file.
    fn check_go(&self, path: &Path) -> Result<Vec<Diagnostic>> {
        if !is_command_available("go") {
            return Ok(Vec::new());
        }

        // Determine the directory containing the Go file for `go vet`.
        let dir = path.parent().unwrap_or(&self.workspace);

        let output = Command::new("go")
            .arg("vet")
            .arg("./...")
            .current_dir(dir)
            .output();

        match output {
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Ok(parsers::parse_go_vet(&stderr))
            }
            Err(_) => Ok(Vec::new()),
        }
    }
}

/// Detect the language of a file based on its extension.
///
/// Returns `None` for unrecognized extensions.
pub fn detect_language(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?;
    match ext {
        "rs" => Some("rust"),
        "ts" | "tsx" => Some("typescript"),
        "py" => Some("python"),
        "go" => Some("go"),
        _ => None,
    }
}

/// Check whether a command is available on the system PATH.
///
/// Uses `which` on Unix-like systems and `where` on Windows.
pub fn is_command_available(cmd: &str) -> bool {
    #[cfg(unix)]
    let check = Command::new("which").arg(cmd).output();

    #[cfg(windows)]
    let check = Command::new("where").arg(cmd).output();

    match check {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language_rust() {
        assert_eq!(detect_language(Path::new("src/lib.rs")), Some("rust"));
        assert_eq!(detect_language(Path::new("main.rs")), Some("rust"));
    }

    #[test]
    fn test_detect_language_typescript() {
        assert_eq!(detect_language(Path::new("src/app.ts")), Some("typescript"));
        assert_eq!(
            detect_language(Path::new("component.tsx")),
            Some("typescript")
        );
    }

    #[test]
    fn test_detect_language_python() {
        assert_eq!(detect_language(Path::new("script.py")), Some("python"));
    }

    #[test]
    fn test_detect_language_go() {
        assert_eq!(detect_language(Path::new("main.go")), Some("go"));
    }

    #[test]
    fn test_detect_language_unknown() {
        assert_eq!(detect_language(Path::new("style.css")), None);
        assert_eq!(detect_language(Path::new("README.md")), None);
        assert_eq!(detect_language(Path::new("data.xyz")), None);
    }

    #[test]
    fn test_detect_language_no_extension() {
        assert_eq!(detect_language(Path::new("Makefile")), None);
        assert_eq!(detect_language(Path::new(".")), None);
    }

    #[test]
    fn test_parse_cargo_check_json() {
        let output = r#"{"reason":"compiler-message","package_id":"foo 0.1.0","manifest_path":"/tmp/foo/Cargo.toml","target":{"kind":["lib"],"crate_types":["lib"],"name":"foo","src_path":"/tmp/foo/src/lib.rs","edition":"2021","doc":true,"doctest":true,"test":true},"message":{"rendered":"error[E0308]: mismatched types\n","children":[],"code":{"code":"E0308","explanation":null},"level":"error","message":"expected `u32`, found `String`","spans":[{"byte_end":100,"byte_start":90,"column_end":15,"column_start":10,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":42,"line_start":42,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;

        let diagnostics = parsers::parse_cargo_check(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].file, PathBuf::from("src/lib.rs"));
        assert_eq!(diagnostics[0].line, 42);
        assert_eq!(diagnostics[0].col, 10);
        assert!(matches!(diagnostics[0].severity, Severity::Error));
        assert_eq!(diagnostics[0].message, "expected `u32`, found `String`");
    }

    #[test]
    fn test_parse_tsc_output() {
        let output =
            "src/app.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'.";
        let diagnostics = parsers::parse_tsc(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].file, PathBuf::from("src/app.ts"));
        assert_eq!(diagnostics[0].line, 10);
        assert_eq!(diagnostics[0].col, 5);
        assert!(matches!(diagnostics[0].severity, Severity::Error));
    }

    #[test]
    fn test_parse_python_compile() {
        let stderr = r#"  File "test.py", line 5
    x = (
        ^
SyntaxError: unexpected EOF while parsing"#;

        let diagnostics = parsers::parse_python_compile(stderr);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].file, PathBuf::from("test.py"));
        assert_eq!(diagnostics[0].line, 5);
        assert!(diagnostics[0].message.contains("SyntaxError"));
    }

    #[test]
    fn test_parse_go_vet() {
        let output = "main.go:15:2: unreachable code";
        let diagnostics = parsers::parse_go_vet(output);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].file, PathBuf::from("main.go"));
        assert_eq!(diagnostics[0].line, 15);
        assert_eq!(diagnostics[0].col, 2);
    }

    #[test]
    fn test_format_diagnostics() {
        let diagnostics = vec![
            Diagnostic {
                file: PathBuf::from("file.rs"),
                line: 42,
                col: 10,
                severity: Severity::Error,
                message: "expected `u32`, found `String`".to_string(),
            },
            Diagnostic {
                file: PathBuf::from("file.rs"),
                line: 55,
                col: 1,
                severity: Severity::Warning,
                message: "unused variable `x`".to_string(),
            },
        ];

        let result = EditValidator::format_for_llm(&diagnostics);
        assert!(result.contains("Diagnostics found after edit:"));
        assert!(result.contains("[ERROR] file.rs:42:10 -- expected `u32`, found `String`"));
        assert!(result.contains("[WARN] file.rs:55:1 -- unused variable `x`"));
        assert!(result.contains("Fix these issues before proceeding."));
    }

    #[test]
    fn test_empty_diagnostics() {
        let result = EditValidator::format_for_llm(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_config_disables_language() {
        let mut config = LspConfig::default();
        config.languages.insert("rust".to_string(), false);

        let validator = EditValidator::new(PathBuf::from("/tmp"), config);
        let diagnostics = validator.check_file(Path::new("src/lib.rs")).unwrap();
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_config_global_disable() {
        let config = LspConfig {
            enabled: false,
            languages: HashMap::new(),
        };

        let validator = EditValidator::new(PathBuf::from("/tmp"), config);
        let diagnostics = validator.check_file(Path::new("src/lib.rs")).unwrap();
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_config_default_enables_all() {
        let config = LspConfig::default();
        assert!(config.is_language_enabled("rust"));
        assert!(config.is_language_enabled("typescript"));
        assert!(config.is_language_enabled("python"));
        assert!(config.is_language_enabled("go"));
        assert!(config.is_language_enabled("anything"));
    }

    #[test]
    fn test_unrecognized_extension_returns_empty() {
        let config = LspConfig::default();
        let validator = EditValidator::new(PathBuf::from("/tmp"), config);
        let diagnostics = validator.check_file(Path::new("style.css")).unwrap();
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_is_command_available_cargo() {
        // cargo should be available in the test environment
        assert!(is_command_available("cargo"));
    }

    #[test]
    fn test_is_command_not_available() {
        assert!(!is_command_available("nonexistent_tool_xyz_12345"));
    }
}
