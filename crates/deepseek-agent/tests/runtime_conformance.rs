use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out)?;
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

#[test]
fn legacy_runtime_symbols_are_not_present_in_agent_runtime() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files)?;

    // Conformance deny-list for old runtime paths.
    let forbidden = [
        "mode_router",
        "r1_drive_tools",
        "dsml_rescue",
        "RouterEscalation",
        "R1DriveTools",
    ];

    let mut violations = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)?;
        for symbol in forbidden {
            if text.contains(symbol) {
                violations.push(format!("{} contains {}", file.display(), symbol));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "legacy runtime conformance violation(s):\n{}",
        violations.join("\n")
    );
    Ok(())
}
