//! Tree-sitter based tag extraction for building repo maps.
//!
//! Extracts function, class, struct, trait, enum, and interface definitions
//! from source files using tree-sitter grammars. Results are cached in SQLite
//! keyed by file path and modification time.

use anyhow::{Result, anyhow};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::Path;
use std::time::SystemTime;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor};

/// A single extracted tag (symbol definition).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag {
    pub name: String,
    pub kind: TagKind,
    pub line: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    Function,
    Class,
    Struct,
    Trait,
    Enum,
    Interface,
    Method,
    Impl,
}

impl TagKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "fn",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Trait => "trait",
            Self::Enum => "enum",
            Self::Interface => "interface",
            Self::Method => "method",
            Self::Impl => "impl",
        }
    }
}

/// Tree-sitter tag extractor with SQLite cache.
pub struct TagExtractor {
    cache_db: Connection,
    languages: HashMap<String, (Language, String)>, // ext -> (language, query_src)
}

impl TagExtractor {
    /// Create a new tag extractor with cache stored in `cache_dir`.
    pub fn new(cache_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;
        let db_path = cache_dir.join("tags_cache.db");
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS tag_cache (
                path TEXT NOT NULL,
                mtime_secs INTEGER NOT NULL,
                tags_json TEXT NOT NULL,
                PRIMARY KEY (path)
            );",
        )?;

        let languages = Self::build_language_map();
        Ok(Self {
            cache_db: conn,
            languages,
        })
    }

    /// Create a new tag extractor with an in-memory cache (for testing).
    pub fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS tag_cache (
                path TEXT NOT NULL,
                mtime_secs INTEGER NOT NULL,
                tags_json TEXT NOT NULL,
                PRIMARY KEY (path)
            );",
        )?;
        let languages = Self::build_language_map();
        Ok(Self {
            cache_db: conn,
            languages,
        })
    }

    fn build_language_map() -> HashMap<String, (Language, String)> {
        let mut map = HashMap::new();

        map.insert(
            "rs".to_string(),
            (tree_sitter_rust::LANGUAGE.into(), RUST_QUERY.to_string()),
        );

        map.insert(
            "py".to_string(),
            (
                tree_sitter_python::LANGUAGE.into(),
                PYTHON_QUERY.to_string(),
            ),
        );
        map.insert(
            "pyi".to_string(),
            (
                tree_sitter_python::LANGUAGE.into(),
                PYTHON_QUERY.to_string(),
            ),
        );

        map.insert(
            "js".to_string(),
            (
                tree_sitter_javascript::LANGUAGE.into(),
                JS_QUERY.to_string(),
            ),
        );
        map.insert(
            "jsx".to_string(),
            (
                tree_sitter_javascript::LANGUAGE.into(),
                JS_QUERY.to_string(),
            ),
        );
        map.insert(
            "mjs".to_string(),
            (
                tree_sitter_javascript::LANGUAGE.into(),
                JS_QUERY.to_string(),
            ),
        );

        map.insert(
            "ts".to_string(),
            (
                tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
                TS_QUERY.to_string(),
            ),
        );

        map.insert(
            "tsx".to_string(),
            (
                tree_sitter_typescript::LANGUAGE_TSX.into(),
                TSX_QUERY.to_string(),
            ),
        );

        map.insert(
            "go".to_string(),
            (tree_sitter_go::LANGUAGE.into(), GO_QUERY.to_string()),
        );

        map.insert(
            "java".to_string(),
            (tree_sitter_java::LANGUAGE.into(), JAVA_QUERY.to_string()),
        );

        map.insert(
            "cs".to_string(),
            (
                tree_sitter_c_sharp::LANGUAGE.into(),
                CSHARP_QUERY.to_string(),
            ),
        );

        map.insert(
            "hs".to_string(),
            (
                tree_sitter_haskell::LANGUAGE.into(),
                HASKELL_QUERY.to_string(),
            ),
        );

        map.insert(
            "swift".to_string(),
            (tree_sitter_swift::LANGUAGE.into(), SWIFT_QUERY.to_string()),
        );

        map.insert(
            "zig".to_string(),
            (tree_sitter_zig::LANGUAGE.into(), ZIG_QUERY.to_string()),
        );

        map
    }

    /// Extract tags from a file, using cache if available and fresh.
    pub fn extract_tags(&self, path: &Path) -> Result<Vec<Tag>> {
        let path_str = path.to_string_lossy().to_string();
        let mtime = file_mtime(path)?;

        // Check cache
        if let Ok(cached) = self.get_cached(&path_str, mtime) {
            return Ok(cached);
        }

        // Extract fresh
        let tags = self.extract_tags_uncached(path)?;

        // Store in cache
        let _ = self.set_cached(&path_str, mtime, &tags);

        Ok(tags)
    }

    /// Extract tags without cache lookup.
    pub fn extract_tags_uncached(&self, path: &Path) -> Result<Vec<Tag>> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let (language, query_src) = self
            .languages
            .get(ext)
            .ok_or_else(|| anyhow!("unsupported language extension: {ext}"))?;

        let source = std::fs::read_to_string(path)?;
        extract_tags_from_source(&source, language.clone(), query_src)
    }

    /// Check if a file extension is supported.
    pub fn supports_extension(&self, ext: &str) -> bool {
        self.languages.contains_key(ext)
    }

    /// Get supported extensions.
    pub fn supported_extensions(&self) -> Vec<String> {
        self.languages.keys().cloned().collect()
    }

    fn get_cached(&self, path: &str, mtime: u64) -> Result<Vec<Tag>> {
        let mut stmt = self
            .cache_db
            .prepare("SELECT tags_json FROM tag_cache WHERE path = ?1 AND mtime_secs = ?2")?;
        let json: String =
            stmt.query_row(rusqlite::params![path, mtime as i64], |row| row.get(0))?;
        let tags: Vec<CachedTag> = serde_json::from_str(&json)?;
        Ok(tags.into_iter().map(|ct| ct.into()).collect())
    }

    fn set_cached(&self, path: &str, mtime: u64, tags: &[Tag]) -> Result<()> {
        let cached: Vec<CachedTag> = tags.iter().map(CachedTag::from).collect();
        let json = serde_json::to_string(&cached)?;
        self.cache_db.execute(
            "INSERT OR REPLACE INTO tag_cache (path, mtime_secs, tags_json) VALUES (?1, ?2, ?3)",
            rusqlite::params![path, mtime as i64, json],
        )?;
        Ok(())
    }

    /// Clear the entire cache.
    pub fn clear_cache(&self) -> Result<()> {
        self.cache_db.execute("DELETE FROM tag_cache", [])?;
        Ok(())
    }
}

/// Extract tags from source code string using a tree-sitter language and query.
pub fn extract_tags_from_source(
    source: &str,
    language: Language,
    query_src: &str,
) -> Result<Vec<Tag>> {
    let mut parser = Parser::new();
    parser.set_language(&language)?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow!("tree-sitter parse failed"))?;

    let query = Query::new(&language, query_src)?;
    let mut cursor = QueryCursor::new();
    let source_bytes = source.as_bytes();

    let mut tags = Vec::new();
    let capture_names: Vec<String> = query
        .capture_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut matches = cursor.matches(&query, tree.root_node(), source_bytes);
    while let Some(m) = matches.next() {
        let mut name: Option<String> = None;
        let mut kind: Option<TagKind> = None;
        let mut line: usize = 0;

        for capture in m.captures {
            let capture_name = &capture_names[capture.index as usize];
            let text = capture.node.utf8_text(source_bytes).unwrap_or_default();

            if capture_name == "name" || capture_name.ends_with(".name") {
                name = Some(text.to_string());
                line = capture.node.start_position().row + 1;
            } else if capture_name == "definition" || capture_name.starts_with("definition.") {
                let tag_kind = match capture_name.as_str() {
                    "definition.function" | "definition" => TagKind::Function,
                    "definition.class" => TagKind::Class,
                    "definition.struct" => TagKind::Struct,
                    "definition.trait" => TagKind::Trait,
                    "definition.enum" => TagKind::Enum,
                    "definition.interface" => TagKind::Interface,
                    "definition.method" => TagKind::Method,
                    "definition.impl" => TagKind::Impl,
                    _ => TagKind::Function,
                };
                kind = Some(tag_kind);
                if name.is_none() {
                    // For some patterns, the definition capture IS the name
                    let t = text.to_string();
                    if !t.is_empty() && t.len() < 100 {
                        name = Some(t);
                        line = capture.node.start_position().row + 1;
                    }
                }
            }
        }

        if let (Some(name), Some(kind)) = (name, kind)
            && !name.is_empty()
        {
            tags.push(Tag { name, kind, line });
        }
    }

    tags.dedup_by(|a, b| a.name == b.name && a.kind == b.kind && a.line == b.line);
    Ok(tags)
}

/// Convert tags to symbol hint strings for repo map consumption.
pub fn tags_to_symbol_hints(tags: &[Tag], max: usize) -> Vec<String> {
    tags.iter()
        .take(max)
        .map(|t| format!("{}:{}", t.kind.as_str(), t.name))
        .collect()
}

fn file_mtime(path: &Path) -> Result<u64> {
    let metadata = std::fs::metadata(path)?;
    let mtime = metadata
        .modified()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Ok(mtime)
}

// ── Serialization helpers for cache ────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedTag {
    name: String,
    kind: String,
    line: usize,
}

impl From<CachedTag> for Tag {
    fn from(ct: CachedTag) -> Self {
        let kind = match ct.kind.as_str() {
            "fn" => TagKind::Function,
            "class" => TagKind::Class,
            "struct" => TagKind::Struct,
            "trait" => TagKind::Trait,
            "enum" => TagKind::Enum,
            "interface" => TagKind::Interface,
            "method" => TagKind::Method,
            "impl" => TagKind::Impl,
            _ => TagKind::Function,
        };
        Tag {
            name: ct.name,
            kind,
            line: ct.line,
        }
    }
}

impl From<&Tag> for CachedTag {
    fn from(t: &Tag) -> Self {
        CachedTag {
            name: t.name.clone(),
            kind: t.kind.as_str().to_string(),
            line: t.line,
        }
    }
}

// ── Tree-sitter queries per language ───────────────────────────────────

const RUST_QUERY: &str = r#"
(function_item name: (identifier) @name) @definition.function
(struct_item name: (type_identifier) @name) @definition.struct
(enum_item name: (type_identifier) @name) @definition.enum
(trait_item name: (type_identifier) @name) @definition.trait
(impl_item type: (type_identifier) @name) @definition.impl
"#;

const PYTHON_QUERY: &str = r#"
(function_definition name: (identifier) @name) @definition.function
(class_definition name: (identifier) @name) @definition.class
"#;

const JS_QUERY: &str = r#"
(function_declaration name: (identifier) @name) @definition.function
(class_declaration name: (identifier) @name) @definition.class
(method_definition name: (property_identifier) @name) @definition.method
(arrow_function) @definition.function
(variable_declarator name: (identifier) @name value: (arrow_function)) @definition.function
"#;

const TS_QUERY: &str = r#"
(function_declaration name: (identifier) @name) @definition.function
(class_declaration name: (type_identifier) @name) @definition.class
(method_definition name: (property_identifier) @name) @definition.method
(interface_declaration name: (type_identifier) @name) @definition.interface
(type_alias_declaration name: (type_identifier) @name) @definition.interface
(enum_declaration name: (identifier) @name) @definition.enum
"#;

const TSX_QUERY: &str = r#"
(function_declaration name: (identifier) @name) @definition.function
(class_declaration name: (type_identifier) @name) @definition.class
(method_definition name: (property_identifier) @name) @definition.method
(interface_declaration name: (type_identifier) @name) @definition.interface
(type_alias_declaration name: (type_identifier) @name) @definition.interface
(enum_declaration name: (identifier) @name) @definition.enum
"#;

const GO_QUERY: &str = r#"
(function_declaration name: (identifier) @name) @definition.function
(method_declaration name: (field_identifier) @name) @definition.method
(type_declaration (type_spec name: (type_identifier) @name)) @definition.struct
"#;

const JAVA_QUERY: &str = r#"
(class_declaration name: (identifier) @name) @definition.class
(method_declaration name: (identifier) @name) @definition.method
(interface_declaration name: (identifier) @name) @definition.interface
(enum_declaration name: (identifier) @name) @definition.enum
"#;

const CSHARP_QUERY: &str = r#"
(class_declaration name: (identifier) @name) @definition.class
(method_declaration name: (identifier) @name) @definition.method
(interface_declaration name: (identifier) @name) @definition.interface
(enum_declaration name: (identifier) @name) @definition.enum
(struct_declaration name: (identifier) @name) @definition.struct
"#;

const HASKELL_QUERY: &str = r#"
(function name: (variable) @name) @definition.function
(type_synomym name: (name) @name) @definition.interface
(newtype name: (name) @name) @definition.struct
(data_type name: (name) @name) @definition.enum
(class name: (name) @name) @definition.class
"#;

const SWIFT_QUERY: &str = r#"
(function_declaration name: (simple_identifier) @name) @definition.function
(class_declaration name: (type_identifier) @name) @definition.class
(protocol_declaration name: (type_identifier) @name) @definition.interface
"#;

const ZIG_QUERY: &str = r#"
(function_declaration name: (identifier) @name) @definition.function
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_rust_tags() {
        let source = r#"
fn hello() {}
pub fn world() {}
struct Foo {}
enum Bar { A, B }
trait Baz {}
impl Foo {}
"#;
        let lang: Language = tree_sitter_rust::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, RUST_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"hello"),
            "should find hello; got: {:?}",
            names
        );
        assert!(
            names.contains(&"world"),
            "should find world; got: {:?}",
            names
        );
        assert!(names.contains(&"Foo"), "should find Foo; got: {:?}", names);
        assert!(names.contains(&"Bar"), "should find Bar; got: {:?}", names);
        assert!(names.contains(&"Baz"), "should find Baz; got: {:?}", names);
    }

    #[test]
    fn extract_python_tags() {
        let source = r#"
def hello():
    pass

class MyClass:
    def method(self):
        pass
"#;
        let lang: Language = tree_sitter_python::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, PYTHON_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"hello"),
            "should find hello; got: {:?}",
            names
        );
        assert!(
            names.contains(&"MyClass"),
            "should find MyClass; got: {:?}",
            names
        );
        assert!(
            names.contains(&"method"),
            "should find method; got: {:?}",
            names
        );
    }

    #[test]
    fn extract_js_tags() {
        let source = r#"
function greet() {}
class Widget {}
"#;
        let lang: Language = tree_sitter_javascript::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, JS_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"greet"),
            "should find greet; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Widget"),
            "should find Widget; got: {:?}",
            names
        );
    }

    #[test]
    fn extract_ts_tags() {
        let source = r#"
function greet(): void {}
class Widget {}
interface Props {}
enum Color { Red, Green }
"#;
        let lang: Language = tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into();
        let tags = extract_tags_from_source(source, lang, TS_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"greet"),
            "should find greet; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Widget"),
            "should find Widget; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Props"),
            "should find Props; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Color"),
            "should find Color; got: {:?}",
            names
        );
    }

    #[test]
    fn extract_go_tags() {
        let source = r#"
package main

func Hello() {}

type Server struct {}

func (s *Server) Start() {}
"#;
        let lang: Language = tree_sitter_go::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, GO_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"Hello"),
            "should find Hello; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Server"),
            "should find Server; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Start"),
            "should find Start; got: {:?}",
            names
        );
    }

    #[test]
    fn extract_java_tags() {
        let source = r#"
public class App {
    public void run() {}
}

interface Service {}
enum Status { OK, ERROR }
"#;
        let lang: Language = tree_sitter_java::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, JAVA_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"App"), "should find App; got: {:?}", names);
        assert!(names.contains(&"run"), "should find run; got: {:?}", names);
        assert!(
            names.contains(&"Service"),
            "should find Service; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Status"),
            "should find Status; got: {:?}",
            names
        );
    }

    #[test]
    fn extract_csharp_tags() {
        let source = r#"
public class App {
    public void Run() {}
}

interface IService {}
enum Status { Ok, Error }
struct Point {}
"#;
        let lang: Language = tree_sitter_c_sharp::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, CSHARP_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"App"), "should find App; got: {:?}", names);
        assert!(names.contains(&"Run"), "should find Run; got: {:?}", names);
        assert!(
            names.contains(&"IService"),
            "should find IService; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Status"),
            "should find Status; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Point"),
            "should find Point; got: {:?}",
            names
        );
    }

    #[test]
    fn extract_haskell_tags() {
        let source = r#"
module Main where

hello :: Int -> Int
hello x = x + 1

data Color = Red | Green | Blue

type Name = String
"#;
        let lang: Language = tree_sitter_haskell::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, HASKELL_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"hello"),
            "should find hello; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Color"),
            "should find Color; got: {:?}",
            names
        );
    }

    #[test]
    fn extract_swift_tags() {
        let source = r#"
func greet() {}
class Widget {}
protocol Drawable {}
struct Point {}
enum Direction { case north, south }
"#;
        let lang: Language = tree_sitter_swift::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, SWIFT_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"greet"),
            "should find greet; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Widget"),
            "should find Widget; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Drawable"),
            "should find Drawable; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Point"),
            "should find Point; got: {:?}",
            names
        );
        assert!(
            names.contains(&"Direction"),
            "should find Direction; got: {:?}",
            names
        );
    }

    #[test]
    fn extract_zig_tags() {
        let source = r#"
const std = @import("std");

fn hello() void {}

pub fn main() !void {
    std.debug.print("hello\n", .{});
}
"#;
        let lang: Language = tree_sitter_zig::LANGUAGE.into();
        let tags = extract_tags_from_source(source, lang, ZIG_QUERY).unwrap();
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"hello"),
            "should find hello; got: {:?}",
            names
        );
        assert!(
            names.contains(&"main"),
            "should find main; got: {:?}",
            names
        );
    }

    #[test]
    fn cache_round_trip() {
        let extractor = TagExtractor::new_in_memory().unwrap();
        let tags = vec![
            Tag {
                name: "hello".to_string(),
                kind: TagKind::Function,
                line: 1,
            },
            Tag {
                name: "Foo".to_string(),
                kind: TagKind::Struct,
                line: 5,
            },
        ];
        extractor.set_cached("/test.rs", 12345, &tags).unwrap();
        let cached = extractor.get_cached("/test.rs", 12345).unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].name, "hello");
        assert_eq!(cached[1].name, "Foo");
    }

    #[test]
    fn cache_miss_on_mtime_change() {
        let extractor = TagExtractor::new_in_memory().unwrap();
        let tags = vec![Tag {
            name: "hello".to_string(),
            kind: TagKind::Function,
            line: 1,
        }];
        extractor.set_cached("/test.rs", 12345, &tags).unwrap();
        assert!(extractor.get_cached("/test.rs", 99999).is_err());
    }

    #[test]
    fn tags_to_hints() {
        let tags = vec![
            Tag {
                name: "hello".to_string(),
                kind: TagKind::Function,
                line: 1,
            },
            Tag {
                name: "Foo".to_string(),
                kind: TagKind::Struct,
                line: 5,
            },
            Tag {
                name: "Bar".to_string(),
                kind: TagKind::Enum,
                line: 10,
            },
        ];
        let hints = tags_to_symbol_hints(&tags, 2);
        assert_eq!(hints, vec!["fn:hello", "struct:Foo"]);
    }

    #[test]
    fn supported_extensions_check() {
        let extractor = TagExtractor::new_in_memory().unwrap();
        assert!(extractor.supports_extension("rs"));
        assert!(extractor.supports_extension("py"));
        assert!(extractor.supports_extension("js"));
        assert!(extractor.supports_extension("ts"));
        assert!(extractor.supports_extension("tsx"));
        assert!(extractor.supports_extension("go"));
        assert!(extractor.supports_extension("java"));
        assert!(extractor.supports_extension("cs"));
        assert!(extractor.supports_extension("hs"));
        assert!(extractor.supports_extension("swift"));
        assert!(extractor.supports_extension("zig"));
        assert!(!extractor.supports_extension("rb"));
        assert!(!extractor.supports_extension("kt")); // Kotlin not supported (incompatible API)
    }
}
