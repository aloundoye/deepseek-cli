use anyhow::Result;
use deepseek_agent::AgentEngine;
use std::path::Path;

pub fn run_replay_smoke(workspace: &Path) -> Result<String> {
    let engine = AgentEngine::new(workspace)?;
    let _ = engine.run_once("replay test", false)?;
    engine.resume()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_smoke() {
        let wd = Path::new(".");
        let result = run_replay_smoke(wd);
        assert!(result.is_ok(), "replay smoke failed: {:?}", result.err());
    }
}
