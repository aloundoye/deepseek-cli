use anyhow::Result;
use chrono::Utc;
use deepseek_core::{EventEnvelope, EventKind, Plan, Session, SessionState, runtime_dir};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const MIGRATIONS: &[(i64, &str)] = &[
    (
        1,
        "CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            seq_no INTEGER NOT NULL,
            at TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            workspace_root TEXT NOT NULL,
            baseline_commit TEXT,
            status TEXT NOT NULL,
            budgets TEXT NOT NULL,
            active_plan_id TEXT,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS plans (
            plan_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            goal TEXT NOT NULL,
            payload TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );",
    ),
    (
        2,
        "CREATE TABLE IF NOT EXISTS plugin_state (
            plugin_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            path TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            manifest_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS approvals_ledger (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            invocation_id TEXT NOT NULL,
            approved_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS verification_runs (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            command TEXT NOT NULL,
            success INTEGER NOT NULL,
            output TEXT NOT NULL,
            run_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS router_stats (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            selected_model TEXT NOT NULL,
            score REAL NOT NULL,
            reasons TEXT NOT NULL,
            recorded_at TEXT NOT NULL
         );",
    ),
    (
        3,
        "CREATE TABLE IF NOT EXISTS usage_ledger (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            unit TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            recorded_at TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_usage_ledger_session_time ON usage_ledger(session_id, recorded_at);
         CREATE TABLE IF NOT EXISTS context_compactions (
            summary_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            from_turn INTEGER NOT NULL,
            to_turn INTEGER NOT NULL,
            token_delta_estimate INTEGER NOT NULL,
            replay_pointer TEXT NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS autopilot_runs (
            run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            status TEXT NOT NULL,
            stop_reason TEXT,
            completed_iterations INTEGER NOT NULL,
            failed_iterations INTEGER NOT NULL,
            consecutive_failures INTEGER NOT NULL,
            last_error TEXT,
            stop_file TEXT NOT NULL,
            heartbeat_file TEXT NOT NULL,
            tools INTEGER NOT NULL,
            max_think INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS hook_executions (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            hook_path TEXT NOT NULL,
            success INTEGER NOT NULL,
            timed_out INTEGER NOT NULL,
            exit_code INTEGER,
            recorded_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS plugin_catalog_cache (
            plugin_id TEXT NOT NULL,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            description TEXT NOT NULL,
            source TEXT NOT NULL,
            signature TEXT,
            verified INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (plugin_id, version)
         );",
    ),
    (
        4,
        "CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            snapshot_path TEXT NOT NULL,
            files_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS transcript_exports (
            export_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            format TEXT NOT NULL,
            output_path TEXT NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS mcp_servers (
            server_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            transport TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS mcp_tools_cache (
            server_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            description TEXT NOT NULL,
            schema_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (server_id, tool_name)
         );
         CREATE TABLE IF NOT EXISTS subagent_runs (
            run_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            goal TEXT NOT NULL,
            status TEXT NOT NULL,
            output TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS cost_ledger (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            estimated_cost_usd REAL NOT NULL,
            recorded_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS profile_runs (
            profile_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            elapsed_ms INTEGER NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS memory_versions (
            version_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            content TEXT NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL
         );",
    ),
    (
        5,
        "CREATE TABLE IF NOT EXISTS background_jobs (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            reference TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS replay_cassettes (
            cassette_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            deterministic INTEGER NOT NULL,
            events_count INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS skill_registry (
            skill_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS provider_metrics (
            id INTEGER PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            cache_key TEXT,
            cache_hit INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            recorded_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS marketplace_catalog (
            plugin_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            source TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS visual_artifacts (
            artifact_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            mime TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS remote_env_profiles (
            profile_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            auth_mode TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );",
    ),
    (
        6,
        "CREATE TABLE IF NOT EXISTS task_queue (
            task_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            outcome TEXT,
            artifact_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS web_search_cache (
            query_hash TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            results_json TEXT NOT NULL,
            results_count INTEGER NOT NULL,
            cached_at TEXT NOT NULL,
            ttl_seconds INTEGER NOT NULL DEFAULT 900
         );
         CREATE TABLE IF NOT EXISTS session_locks (
            session_id TEXT PRIMARY KEY,
            lock_holder TEXT NOT NULL,
            locked_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS review_runs (
            review_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            preset TEXT NOT NULL,
            target TEXT NOT NULL,
            findings_json TEXT NOT NULL,
            findings_count INTEGER NOT NULL,
            critical_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            files_json TEXT NOT NULL,
            created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS permission_mode_log (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            from_mode TEXT NOT NULL,
            to_mode TEXT NOT NULL,
            changed_at TEXT NOT NULL
         );",
    ),
];

#[derive(Debug, Clone)]
pub struct PluginStateRecord {
    pub plugin_id: String,
    pub name: String,
    pub version: String,
    pub path: String,
    pub enabled: bool,
    pub manifest_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummary {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub records: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageByUnitSummary {
    pub unit: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCompactionRecord {
    pub summary_id: Uuid,
    pub session_id: Uuid,
    pub from_turn: u64,
    pub to_turn: u64,
    pub token_delta_estimate: i64,
    pub replay_pointer: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopilotRunRecord {
    pub run_id: Uuid,
    pub session_id: Uuid,
    pub prompt: String,
    pub status: String,
    pub stop_reason: Option<String>,
    pub completed_iterations: u64,
    pub failed_iterations: u64,
    pub consecutive_failures: u64,
    pub last_error: Option<String>,
    pub stop_file: String,
    pub heartbeat_file: String,
    pub tools: bool,
    pub max_think: bool,
    pub started_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCatalogEntryRecord {
    pub plugin_id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub source: String,
    pub signature: Option<String>,
    pub verified: bool,
    pub metadata_json: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointRecord {
    pub checkpoint_id: Uuid,
    pub reason: String,
    pub snapshot_path: String,
    pub files_count: u64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptExportRecord {
    pub export_id: Uuid,
    pub session_id: Uuid,
    pub format: String,
    pub output_path: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerRecord {
    pub server_id: String,
    pub name: String,
    pub transport: String,
    pub endpoint: String,
    pub enabled: bool,
    pub metadata_json: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCacheRecord {
    pub server_id: String,
    pub tool_name: String,
    pub description: String,
    pub schema_json: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentRunRecord {
    pub run_id: Uuid,
    pub name: String,
    pub goal: String,
    pub status: String,
    pub output: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileRunRecord {
    pub profile_id: Uuid,
    pub session_id: Uuid,
    pub summary: String,
    pub elapsed_ms: u64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRunRecord {
    pub command: String,
    pub success: bool,
    pub output: String,
    pub run_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundJobRecord {
    pub job_id: Uuid,
    pub kind: String,
    pub reference: String,
    pub status: String,
    pub metadata_json: String,
    pub started_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayCassetteRecord {
    pub cassette_id: Uuid,
    pub session_id: Uuid,
    pub deterministic: bool,
    pub events_count: u64,
    pub payload_json: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRegistryRecord {
    pub skill_id: String,
    pub name: String,
    pub path: String,
    pub enabled: bool,
    pub metadata_json: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderMetricRecord {
    pub provider: String,
    pub model: String,
    pub cache_key: Option<String>,
    pub cache_hit: bool,
    pub latency_ms: u64,
    pub recorded_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceCatalogRecord {
    pub plugin_id: String,
    pub name: String,
    pub version: String,
    pub source: String,
    pub metadata_json: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualArtifactRecord {
    pub artifact_id: Uuid,
    pub path: String,
    pub mime: String,
    pub metadata_json: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteEnvProfileRecord {
    pub profile_id: Uuid,
    pub name: String,
    pub endpoint: String,
    pub auth_mode: String,
    pub metadata_json: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskQueueRecord {
    pub task_id: Uuid,
    pub session_id: Uuid,
    pub title: String,
    pub priority: u32,
    pub status: String,
    pub outcome: Option<String>,
    pub artifact_path: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchCacheRecord {
    pub query_hash: String,
    pub query: String,
    pub results_json: String,
    pub results_count: u64,
    pub cached_at: String,
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLockRecord {
    pub session_id: Uuid,
    pub lock_holder: String,
    pub locked_at: String,
    pub heartbeat_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewRunRecord {
    pub review_id: Uuid,
    pub session_id: Uuid,
    pub preset: String,
    pub target: String,
    pub findings_json: String,
    pub findings_count: u64,
    pub critical_count: u64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub artifact_id: Uuid,
    pub task_id: Uuid,
    pub artifact_path: String,
    pub files_json: String,
    pub created_at: String,
}

pub struct Store {
    pub root: PathBuf,
    db_path: PathBuf,
    events_path: PathBuf,
}

impl Store {
    pub fn new(workspace: &Path) -> Result<Self> {
        let root = runtime_dir(workspace);
        fs::create_dir_all(&root)?;
        let db_path = root.join("store.sqlite");
        let events_path = root.join("events.jsonl");
        let store = Self {
            root,
            db_path,
            events_path,
        };
        store.init_db()?;
        Ok(store)
    }

    pub fn db(&self) -> Result<Connection> {
        Ok(Connection::open(&self.db_path)?)
    }

    pub fn next_seq_no(&self, session_id: Uuid) -> Result<u64> {
        let conn = self.db()?;
        let mut stmt =
            conn.prepare("SELECT COALESCE(MAX(seq_no), 0) FROM events WHERE session_id = ?1")?;
        let current: i64 = stmt.query_row([session_id.to_string()], |r| r.get(0))?;
        Ok((current as u64) + 1)
    }

    pub fn append_event(&self, event: &EventEnvelope) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.events_path)?;
        writeln!(file, "{}", serde_json::to_string(event)?)?;

        let conn = self.db()?;
        conn.execute(
            "INSERT INTO events (session_id, seq_no, at, kind, payload) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                event.session_id.to_string(),
                event.seq_no as i64,
                event.at.to_rfc3339(),
                event_kind_name(&event.kind),
                serde_json::to_string(&event.kind)?,
            ],
        )?;
        self.project_event(&conn, event)?;
        Ok(())
    }

    pub fn save_session(&self, session: &Session) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, workspace_root, baseline_commit, status, budgets, active_plan_id, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session.session_id.to_string(),
                session.workspace_root,
                session.baseline_commit,
                serde_json::to_string(&session.status)?,
                serde_json::to_string(&session.budgets)?,
                session.active_plan_id.map(|id| id.to_string()),
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn load_latest_session(&self) -> Result<Option<Session>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT session_id, workspace_root, baseline_commit, status, budgets, active_plan_id
             FROM sessions ORDER BY updated_at DESC LIMIT 1",
        )?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(Session {
                session_id: Uuid::parse_str(row.get::<_, String>(0)?.as_str())?,
                workspace_root: row.get(1)?,
                baseline_commit: row.get(2)?,
                status: serde_json::from_str(&row.get::<_, String>(3)?)?,
                budgets: serde_json::from_str(&row.get::<_, String>(4)?)?,
                active_plan_id: row
                    .get::<_, Option<String>>(5)?
                    .map(|v| Uuid::parse_str(&v))
                    .transpose()?,
            }));
        }
        Ok(None)
    }

    pub fn load_session(&self, session_id: Uuid) -> Result<Option<Session>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT session_id, workspace_root, baseline_commit, status, budgets, active_plan_id
             FROM sessions WHERE session_id = ?1",
        )?;
        let mut rows = stmt.query(params![session_id.to_string()])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(Session {
                session_id: Uuid::parse_str(row.get::<_, String>(0)?.as_str())?,
                workspace_root: row.get(1)?,
                baseline_commit: row.get(2)?,
                status: serde_json::from_str(&row.get::<_, String>(3)?)?,
                budgets: serde_json::from_str(&row.get::<_, String>(4)?)?,
                active_plan_id: row
                    .get::<_, Option<String>>(5)?
                    .map(|v| Uuid::parse_str(&v))
                    .transpose()?,
            }));
        }
        Ok(None)
    }

    pub fn save_plan(&self, session_id: Uuid, plan: &Plan) -> Result<()> {
        let plan_dir = self.root.join("plans");
        fs::create_dir_all(&plan_dir)?;
        fs::write(
            plan_dir.join(format!("{}.json", plan.plan_id)),
            serde_json::to_vec_pretty(plan)?,
        )?;

        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO plans (plan_id, session_id, version, goal, payload, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                plan.plan_id.to_string(),
                session_id.to_string(),
                plan.version as i64,
                plan.goal,
                serde_json::to_string(plan)?,
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn set_plugin_state(&self, record: &PluginStateRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO plugin_state (plugin_id, name, version, path, enabled, manifest_json, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                record.plugin_id,
                record.name,
                record.version,
                record.path,
                if record.enabled { 1 } else { 0 },
                record.manifest_json,
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn remove_plugin_state(&self, plugin_id: &str) -> Result<()> {
        let conn = self.db()?;
        conn.execute("DELETE FROM plugin_state WHERE plugin_id = ?1", [plugin_id])?;
        Ok(())
    }

    pub fn list_plugin_states(&self) -> Result<Vec<PluginStateRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT plugin_id, name, version, path, enabled, manifest_json
             FROM plugin_state ORDER BY plugin_id ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PluginStateRecord {
                plugin_id: r.get(0)?,
                name: r.get(1)?,
                version: r.get(2)?,
                path: r.get(3)?,
                enabled: r.get::<_, i64>(4)? != 0,
                manifest_json: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn set_plugin_catalog_entries(
        &self,
        source: &str,
        entries: &[PluginCatalogEntryRecord],
    ) -> Result<()> {
        let conn = self.db()?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "DELETE FROM plugin_catalog_cache WHERE source = ?1",
            [source.to_string()],
        )?;
        for entry in entries {
            conn.execute(
                "INSERT OR REPLACE INTO plugin_catalog_cache
                 (plugin_id, name, version, description, source, signature, verified, metadata_json, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    entry.plugin_id,
                    entry.name,
                    entry.version,
                    entry.description,
                    source,
                    entry.signature,
                    if entry.verified { 1 } else { 0 },
                    entry.metadata_json,
                    now,
                ],
            )?;
        }
        Ok(())
    }

    pub fn list_plugin_catalog_entries(&self) -> Result<Vec<PluginCatalogEntryRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT plugin_id, name, version, description, source, signature, verified, metadata_json, updated_at
             FROM plugin_catalog_cache ORDER BY plugin_id ASC, version DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PluginCatalogEntryRecord {
                plugin_id: r.get(0)?,
                name: r.get(1)?,
                version: r.get(2)?,
                description: r.get(3)?,
                source: r.get(4)?,
                signature: r.get(5)?,
                verified: r.get::<_, i64>(6)? != 0,
                metadata_json: r.get(7)?,
                updated_at: r.get(8)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn upsert_autopilot_run(&self, record: &AutopilotRunRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO autopilot_runs
             (run_id, session_id, prompt, status, stop_reason, completed_iterations, failed_iterations,
              consecutive_failures, last_error, stop_file, heartbeat_file, tools, max_think, started_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                record.run_id.to_string(),
                record.session_id.to_string(),
                record.prompt,
                record.status,
                record.stop_reason,
                record.completed_iterations as i64,
                record.failed_iterations as i64,
                record.consecutive_failures as i64,
                record.last_error,
                record.stop_file,
                record.heartbeat_file,
                if record.tools { 1 } else { 0 },
                if record.max_think { 1 } else { 0 },
                record.started_at,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn load_autopilot_run(&self, run_id: Uuid) -> Result<Option<AutopilotRunRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT run_id, session_id, prompt, status, stop_reason, completed_iterations, failed_iterations,
                    consecutive_failures, last_error, stop_file, heartbeat_file, tools, max_think, started_at, updated_at
             FROM autopilot_runs WHERE run_id = ?1",
        )?;
        let mut rows = stmt.query([run_id.to_string()])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(AutopilotRunRecord {
                run_id: Uuid::parse_str(row.get::<_, String>(0)?.as_str())?,
                session_id: Uuid::parse_str(row.get::<_, String>(1)?.as_str())?,
                prompt: row.get(2)?,
                status: row.get(3)?,
                stop_reason: row.get(4)?,
                completed_iterations: row.get::<_, i64>(5)? as u64,
                failed_iterations: row.get::<_, i64>(6)? as u64,
                consecutive_failures: row.get::<_, i64>(7)? as u64,
                last_error: row.get(8)?,
                stop_file: row.get(9)?,
                heartbeat_file: row.get(10)?,
                tools: row.get::<_, i64>(11)? != 0,
                max_think: row.get::<_, i64>(12)? != 0,
                started_at: row.get(13)?,
                updated_at: row.get(14)?,
            }));
        }
        Ok(None)
    }

    pub fn load_latest_autopilot_run(&self) -> Result<Option<AutopilotRunRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT run_id, session_id, prompt, status, stop_reason, completed_iterations, failed_iterations,
                    consecutive_failures, last_error, stop_file, heartbeat_file, tools, max_think, started_at, updated_at
             FROM autopilot_runs ORDER BY updated_at DESC LIMIT 1",
        )?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(AutopilotRunRecord {
                run_id: Uuid::parse_str(row.get::<_, String>(0)?.as_str())?,
                session_id: Uuid::parse_str(row.get::<_, String>(1)?.as_str())?,
                prompt: row.get(2)?,
                status: row.get(3)?,
                stop_reason: row.get(4)?,
                completed_iterations: row.get::<_, i64>(5)? as u64,
                failed_iterations: row.get::<_, i64>(6)? as u64,
                consecutive_failures: row.get::<_, i64>(7)? as u64,
                last_error: row.get(8)?,
                stop_file: row.get(9)?,
                heartbeat_file: row.get(10)?,
                tools: row.get::<_, i64>(11)? != 0,
                max_think: row.get::<_, i64>(12)? != 0,
                started_at: row.get(13)?,
                updated_at: row.get(14)?,
            }));
        }
        Ok(None)
    }

    pub fn usage_summary(
        &self,
        session_id: Option<Uuid>,
        last_hours: Option<i64>,
    ) -> Result<UsageSummary> {
        let conn = self.db()?;
        let since = last_hours.map(|h| Utc::now() - chrono::Duration::hours(h));
        let (input_tokens, output_tokens, records) = match (session_id, since) {
            (Some(session_id), Some(since)) => conn.query_row(
                "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(COUNT(1),0)
                 FROM usage_ledger WHERE session_id = ?1 AND recorded_at >= ?2",
                params![session_id.to_string(), since.to_rfc3339()],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)),
            )?,
            (Some(session_id), None) => conn.query_row(
                "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(COUNT(1),0)
                 FROM usage_ledger WHERE session_id = ?1",
                [session_id.to_string()],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)),
            )?,
            (None, Some(since)) => conn.query_row(
                "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(COUNT(1),0)
                 FROM usage_ledger WHERE recorded_at >= ?1",
                [since.to_rfc3339()],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)),
            )?,
            (None, None) => conn.query_row(
                "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(COUNT(1),0)
                 FROM usage_ledger",
                [],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)),
            )?,
        };
        Ok(UsageSummary {
            input_tokens: input_tokens.max(0) as u64,
            output_tokens: output_tokens.max(0) as u64,
            records: records.max(0) as u64,
        })
    }

    pub fn usage_by_unit(&self, session_id: Uuid) -> Result<Vec<UsageByUnitSummary>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT unit, COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0)
             FROM usage_ledger WHERE session_id = ?1 GROUP BY unit ORDER BY unit",
        )?;
        let rows = stmt.query_map([session_id.to_string()], |r| {
            Ok(UsageByUnitSummary {
                unit: r.get(0)?,
                input_tokens: r.get::<_, i64>(1)?.max(0) as u64,
                output_tokens: r.get::<_, i64>(2)?.max(0) as u64,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn list_recent_verification_runs(
        &self,
        session_id: Uuid,
        limit: usize,
    ) -> Result<Vec<VerificationRunRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT command, success, output, run_at
             FROM verification_runs
             WHERE session_id = ?1
             ORDER BY run_at DESC
             LIMIT ?2",
        )?;
        let mut out = Vec::new();
        let rows = stmt.query_map(params![session_id.to_string(), limit as i64], |r| {
            Ok(VerificationRunRecord {
                command: r.get(0)?,
                success: r.get::<_, i64>(1)? != 0,
                output: r.get(2)?,
                run_at: r.get(3)?,
            })
        })?;
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn list_context_compactions(
        &self,
        session_id: Option<Uuid>,
    ) -> Result<Vec<ContextCompactionRecord>> {
        let conn = self.db()?;
        let sql = if session_id.is_some() {
            "SELECT summary_id, session_id, from_turn, to_turn, token_delta_estimate, replay_pointer, created_at
             FROM context_compactions WHERE session_id = ?1 ORDER BY created_at DESC"
        } else {
            "SELECT summary_id, session_id, from_turn, to_turn, token_delta_estimate, replay_pointer, created_at
             FROM context_compactions ORDER BY created_at DESC"
        };
        let mut stmt = conn.prepare(sql)?;
        let mut out = Vec::new();
        if let Some(session_id) = session_id {
            let rows = stmt.query_map([session_id.to_string()], |r| {
                Ok(ContextCompactionRecord {
                    summary_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    from_turn: r.get::<_, i64>(2)? as u64,
                    to_turn: r.get::<_, i64>(3)? as u64,
                    token_delta_estimate: r.get(4)?,
                    replay_pointer: r.get(5)?,
                    created_at: r.get(6)?,
                })
            })?;
            for row in rows {
                out.push(row?);
            }
        } else {
            let rows = stmt.query_map([], |r| {
                Ok(ContextCompactionRecord {
                    summary_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    from_turn: r.get::<_, i64>(2)? as u64,
                    to_turn: r.get::<_, i64>(3)? as u64,
                    token_delta_estimate: r.get(4)?,
                    replay_pointer: r.get(5)?,
                    created_at: r.get(6)?,
                })
            })?;
            for row in rows {
                out.push(row?);
            }
        }
        Ok(out)
    }

    pub fn insert_checkpoint(&self, record: &CheckpointRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (checkpoint_id, reason, snapshot_path, files_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.checkpoint_id.to_string(),
                record.reason,
                record.snapshot_path,
                record.files_count as i64,
                record.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn load_checkpoint(&self, checkpoint_id: Uuid) -> Result<Option<CheckpointRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT checkpoint_id, reason, snapshot_path, files_count, created_at
             FROM checkpoints WHERE checkpoint_id = ?1",
        )?;
        let mut rows = stmt.query([checkpoint_id.to_string()])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(CheckpointRecord {
                checkpoint_id: Uuid::parse_str(row.get::<_, String>(0)?.as_str())?,
                reason: row.get(1)?,
                snapshot_path: row.get(2)?,
                files_count: row.get::<_, i64>(3)? as u64,
                created_at: row.get(4)?,
            }));
        }
        Ok(None)
    }

    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT checkpoint_id, reason, snapshot_path, files_count, created_at
             FROM checkpoints ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(CheckpointRecord {
                checkpoint_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                reason: r.get(1)?,
                snapshot_path: r.get(2)?,
                files_count: r.get::<_, i64>(3)? as u64,
                created_at: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn insert_transcript_export(&self, record: &TranscriptExportRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO transcript_exports (export_id, session_id, format, output_path, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.export_id.to_string(),
                record.session_id.to_string(),
                record.format,
                record.output_path,
                record.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn upsert_mcp_server(&self, record: &McpServerRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO mcp_servers (server_id, name, transport, endpoint, enabled, metadata_json, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                record.server_id,
                record.name,
                record.transport,
                record.endpoint,
                if record.enabled { 1 } else { 0 },
                record.metadata_json,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn remove_mcp_server(&self, server_id: &str) -> Result<()> {
        let conn = self.db()?;
        conn.execute("DELETE FROM mcp_servers WHERE server_id = ?1", [server_id])?;
        conn.execute(
            "DELETE FROM mcp_tools_cache WHERE server_id = ?1",
            [server_id],
        )?;
        Ok(())
    }

    pub fn list_mcp_servers(&self) -> Result<Vec<McpServerRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT server_id, name, transport, endpoint, enabled, metadata_json, updated_at
             FROM mcp_servers ORDER BY server_id ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(McpServerRecord {
                server_id: r.get(0)?,
                name: r.get(1)?,
                transport: r.get(2)?,
                endpoint: r.get(3)?,
                enabled: r.get::<_, i64>(4)? != 0,
                metadata_json: r.get(5)?,
                updated_at: r.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn upsert_mcp_tool_cache(&self, record: &McpToolCacheRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO mcp_tools_cache (server_id, tool_name, description, schema_json, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.server_id,
                record.tool_name,
                record.description,
                record.schema_json,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_mcp_tool_cache(&self) -> Result<Vec<McpToolCacheRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT server_id, tool_name, description, schema_json, updated_at
             FROM mcp_tools_cache ORDER BY server_id ASC, tool_name ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(McpToolCacheRecord {
                server_id: r.get(0)?,
                tool_name: r.get(1)?,
                description: r.get(2)?,
                schema_json: r.get(3)?,
                updated_at: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn replace_mcp_tool_cache_for_server(
        &self,
        server_id: &str,
        records: &[McpToolCacheRecord],
    ) -> Result<()> {
        let mut conn = self.db()?;
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM mcp_tools_cache WHERE server_id = ?1",
            [server_id],
        )?;
        for record in records {
            tx.execute(
                "INSERT OR REPLACE INTO mcp_tools_cache (server_id, tool_name, description, schema_json, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    record.server_id,
                    record.tool_name,
                    record.description,
                    record.schema_json,
                    record.updated_at,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_subagent_run(&self, record: &SubagentRunRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO subagent_runs (run_id, name, goal, status, output, error, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                record.run_id.to_string(),
                record.name,
                record.goal,
                record.status,
                record.output,
                record.error,
                record.created_at,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn insert_cost_ledger(
        &self,
        session_id: Uuid,
        input_tokens: u64,
        output_tokens: u64,
        estimated_cost_usd: f64,
    ) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT INTO cost_ledger (session_id, input_tokens, output_tokens, estimated_cost_usd, recorded_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                session_id.to_string(),
                input_tokens as i64,
                output_tokens as i64,
                estimated_cost_usd,
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn insert_profile_run(&self, record: &ProfileRunRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO profile_runs (profile_id, session_id, summary, elapsed_ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.profile_id.to_string(),
                record.session_id.to_string(),
                record.summary,
                record.elapsed_ms as i64,
                record.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn insert_memory_version(
        &self,
        version_id: Uuid,
        path: &str,
        content: &str,
        note: &str,
        created_at: &str,
    ) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO memory_versions (version_id, path, content, note, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![version_id.to_string(), path, content, note, created_at],
        )?;
        Ok(())
    }

    pub fn upsert_background_job(&self, record: &BackgroundJobRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO background_jobs (job_id, kind, reference, status, metadata_json, started_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                record.job_id.to_string(),
                record.kind,
                record.reference,
                record.status,
                record.metadata_json,
                record.started_at,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn load_background_job(&self, job_id: Uuid) -> Result<Option<BackgroundJobRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT job_id, kind, reference, status, metadata_json, started_at, updated_at
             FROM background_jobs WHERE job_id = ?1",
        )?;
        let mut rows = stmt.query([job_id.to_string()])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(BackgroundJobRecord {
                job_id: Uuid::parse_str(row.get::<_, String>(0)?.as_str())?,
                kind: row.get(1)?,
                reference: row.get(2)?,
                status: row.get(3)?,
                metadata_json: row.get(4)?,
                started_at: row.get(5)?,
                updated_at: row.get(6)?,
            }));
        }
        Ok(None)
    }

    pub fn list_background_jobs(&self) -> Result<Vec<BackgroundJobRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT job_id, kind, reference, status, metadata_json, started_at, updated_at
             FROM background_jobs ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(BackgroundJobRecord {
                job_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                kind: r.get(1)?,
                reference: r.get(2)?,
                status: r.get(3)?,
                metadata_json: r.get(4)?,
                started_at: r.get(5)?,
                updated_at: r.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn insert_replay_cassette(&self, record: &ReplayCassetteRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO replay_cassettes (cassette_id, session_id, deterministic, events_count, payload_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                record.cassette_id.to_string(),
                record.session_id.to_string(),
                if record.deterministic { 1 } else { 0 },
                record.events_count as i64,
                record.payload_json,
                record.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_replay_cassettes(
        &self,
        session_id: Option<Uuid>,
        limit: usize,
    ) -> Result<Vec<ReplayCassetteRecord>> {
        let conn = self.db()?;
        let bounded_limit = limit.clamp(1, 500) as i64;
        let mut out = Vec::new();
        if let Some(session_id) = session_id {
            let mut stmt = conn.prepare(
                "SELECT cassette_id, session_id, deterministic, events_count, payload_json, created_at
                 FROM replay_cassettes
                 WHERE session_id = ?1
                 ORDER BY created_at DESC
                 LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![session_id.to_string(), bounded_limit], |r| {
                Ok(ReplayCassetteRecord {
                    cassette_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    deterministic: r.get::<_, i64>(2)? != 0,
                    events_count: r.get::<_, i64>(3)? as u64,
                    payload_json: r.get(4)?,
                    created_at: r.get(5)?,
                })
            })?;
            for row in rows {
                out.push(row?);
            }
            return Ok(out);
        }

        let mut stmt = conn.prepare(
            "SELECT cassette_id, session_id, deterministic, events_count, payload_json, created_at
             FROM replay_cassettes
             ORDER BY created_at DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![bounded_limit], |r| {
            Ok(ReplayCassetteRecord {
                cassette_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                deterministic: r.get::<_, i64>(2)? != 0,
                events_count: r.get::<_, i64>(3)? as u64,
                payload_json: r.get(4)?,
                created_at: r.get(5)?,
            })
        })?;
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn set_skill_registry(&self, record: &SkillRegistryRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO skill_registry (skill_id, name, path, enabled, metadata_json, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                record.skill_id,
                record.name,
                record.path,
                if record.enabled { 1 } else { 0 },
                record.metadata_json,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_skill_registry(&self) -> Result<Vec<SkillRegistryRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT skill_id, name, path, enabled, metadata_json, updated_at
             FROM skill_registry ORDER BY skill_id ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(SkillRegistryRecord {
                skill_id: r.get(0)?,
                name: r.get(1)?,
                path: r.get(2)?,
                enabled: r.get::<_, i64>(3)? != 0,
                metadata_json: r.get(4)?,
                updated_at: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn remove_skill_registry(&self, skill_id: &str) -> Result<()> {
        let conn = self.db()?;
        conn.execute("DELETE FROM skill_registry WHERE skill_id = ?1", [skill_id])?;
        Ok(())
    }

    pub fn insert_provider_metric(&self, record: &ProviderMetricRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT INTO provider_metrics (provider, model, cache_key, cache_hit, latency_ms, recorded_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                record.provider,
                record.model,
                record.cache_key,
                if record.cache_hit { 1 } else { 0 },
                record.latency_ms as i64,
                record.recorded_at,
            ],
        )?;
        Ok(())
    }

    pub fn upsert_marketplace_catalog(&self, record: &MarketplaceCatalogRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO marketplace_catalog (plugin_id, name, version, source, metadata_json, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                record.plugin_id,
                record.name,
                record.version,
                record.source,
                record.metadata_json,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_marketplace_catalog(&self) -> Result<Vec<MarketplaceCatalogRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT plugin_id, name, version, source, metadata_json, updated_at
             FROM marketplace_catalog ORDER BY plugin_id ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(MarketplaceCatalogRecord {
                plugin_id: r.get(0)?,
                name: r.get(1)?,
                version: r.get(2)?,
                source: r.get(3)?,
                metadata_json: r.get(4)?,
                updated_at: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn insert_visual_artifact(&self, record: &VisualArtifactRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO visual_artifacts (artifact_id, path, mime, metadata_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.artifact_id.to_string(),
                record.path,
                record.mime,
                record.metadata_json,
                record.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_visual_artifacts(&self, limit: usize) -> Result<Vec<VisualArtifactRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT artifact_id, path, mime, metadata_json, created_at
             FROM visual_artifacts ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit.max(1) as i64], |r| {
            Ok(VisualArtifactRecord {
                artifact_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                path: r.get(1)?,
                mime: r.get(2)?,
                metadata_json: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn upsert_remote_env_profile(&self, record: &RemoteEnvProfileRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO remote_env_profiles (profile_id, name, endpoint, auth_mode, metadata_json, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                record.profile_id.to_string(),
                record.name,
                record.endpoint,
                record.auth_mode,
                record.metadata_json,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_remote_env_profiles(&self) -> Result<Vec<RemoteEnvProfileRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT profile_id, name, endpoint, auth_mode, metadata_json, updated_at
             FROM remote_env_profiles ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(RemoteEnvProfileRecord {
                profile_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                name: r.get(1)?,
                endpoint: r.get(2)?,
                auth_mode: r.get(3)?,
                metadata_json: r.get(4)?,
                updated_at: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn load_remote_env_profile(
        &self,
        profile_id: Uuid,
    ) -> Result<Option<RemoteEnvProfileRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT profile_id, name, endpoint, auth_mode, metadata_json, updated_at
             FROM remote_env_profiles WHERE profile_id = ?1",
        )?;
        let mut rows = stmt.query([profile_id.to_string()])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(RemoteEnvProfileRecord {
                profile_id: Uuid::parse_str(row.get::<_, String>(0)?.as_str())?,
                name: row.get(1)?,
                endpoint: row.get(2)?,
                auth_mode: row.get(3)?,
                metadata_json: row.get(4)?,
                updated_at: row.get(5)?,
            }));
        }
        Ok(None)
    }

    pub fn remove_remote_env_profile(&self, profile_id: Uuid) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "DELETE FROM remote_env_profiles WHERE profile_id = ?1",
            [profile_id.to_string()],
        )?;
        Ok(())
    }

    // --- Session Locking ---

    pub fn try_acquire_session_lock(&self, session_id: Uuid, holder: &str) -> Result<bool> {
        let conn = self.db()?;
        let now = Utc::now().to_rfc3339();
        // Try to acquire; fail if another holder has it and heartbeat is recent (< 60s)
        let existing: Option<(String, String)> = conn
            .prepare("SELECT lock_holder, heartbeat_at FROM session_locks WHERE session_id = ?1")?
            .query_row([session_id.to_string()], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .ok();
        if let Some((existing_holder, heartbeat)) = existing
            && existing_holder != holder
        {
            // Check if heartbeat is stale (> 60 seconds)
            if let Ok(hb) = chrono::DateTime::parse_from_rfc3339(&heartbeat) {
                let age = Utc::now().signed_duration_since(hb.with_timezone(&Utc));
                if age.num_seconds() < 60 {
                    return Ok(false); // Lock held by another process
                }
            }
        }
        conn.execute(
            "INSERT OR REPLACE INTO session_locks (session_id, lock_holder, locked_at, heartbeat_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![session_id.to_string(), holder, &now, &now],
        )?;
        Ok(true)
    }

    pub fn release_session_lock(&self, session_id: Uuid, holder: &str) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "DELETE FROM session_locks WHERE session_id = ?1 AND lock_holder = ?2",
            params![session_id.to_string(), holder],
        )?;
        Ok(())
    }

    pub fn heartbeat_session_lock(&self, session_id: Uuid, holder: &str) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "UPDATE session_locks SET heartbeat_at = ?1 WHERE session_id = ?2 AND lock_holder = ?3",
            params![Utc::now().to_rfc3339(), session_id.to_string(), holder],
        )?;
        Ok(())
    }

    // --- Session Forking ---

    pub fn fork_session(&self, from_session_id: Uuid) -> Result<Session> {
        let source = self
            .load_session(from_session_id)?
            .ok_or_else(|| anyhow::anyhow!("session not found: {from_session_id}"))?;
        let new_id = Uuid::now_v7();
        let forked = Session {
            session_id: new_id,
            workspace_root: source.workspace_root.clone(),
            baseline_commit: source.baseline_commit.clone(),
            status: SessionState::Idle,
            budgets: source.budgets.clone(),
            active_plan_id: None,
        };
        self.save_session(&forked)?;
        // Record fork events in both sessions
        let seq = self.next_seq_no(from_session_id)?;
        self.append_event(&EventEnvelope {
            seq_no: seq,
            at: Utc::now(),
            session_id: from_session_id,
            kind: EventKind::SessionForkedV1 {
                from_session_id,
                to_session_id: new_id,
            },
        })?;
        Ok(forked)
    }

    // --- Task Queue ---

    pub fn insert_task(&self, record: &TaskQueueRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO task_queue (task_id, session_id, title, priority, status, outcome, artifact_path, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                record.task_id.to_string(),
                record.session_id.to_string(),
                record.title,
                record.priority as i64,
                record.status,
                record.outcome,
                record.artifact_path,
                record.created_at,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn update_task_status(
        &self,
        task_id: Uuid,
        status: &str,
        outcome: Option<&str>,
    ) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "UPDATE task_queue SET status = ?1, outcome = ?2, updated_at = ?3 WHERE task_id = ?4",
            params![
                status,
                outcome,
                Utc::now().to_rfc3339(),
                task_id.to_string()
            ],
        )?;
        Ok(())
    }

    pub fn list_tasks(&self, session_id: Option<Uuid>) -> Result<Vec<TaskQueueRecord>> {
        let conn = self.db()?;
        let mut out = Vec::new();
        if let Some(sid) = session_id {
            let mut stmt = conn.prepare(
                "SELECT task_id, session_id, title, priority, status, outcome, artifact_path, created_at, updated_at
                 FROM task_queue WHERE session_id = ?1 ORDER BY priority DESC, created_at ASC",
            )?;
            let rows = stmt.query_map([sid.to_string()], |r| {
                Ok(TaskQueueRecord {
                    task_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    title: r.get(2)?,
                    priority: r.get::<_, i64>(3)? as u32,
                    status: r.get(4)?,
                    outcome: r.get(5)?,
                    artifact_path: r.get(6)?,
                    created_at: r.get(7)?,
                    updated_at: r.get(8)?,
                })
            })?;
            for row in rows {
                out.push(row?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT task_id, session_id, title, priority, status, outcome, artifact_path, created_at, updated_at
                 FROM task_queue ORDER BY priority DESC, created_at ASC",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok(TaskQueueRecord {
                    task_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                        .unwrap_or_else(|_| Uuid::nil()),
                    title: r.get(2)?,
                    priority: r.get::<_, i64>(3)? as u32,
                    status: r.get(4)?,
                    outcome: r.get(5)?,
                    artifact_path: r.get(6)?,
                    created_at: r.get(7)?,
                    updated_at: r.get(8)?,
                })
            })?;
            for row in rows {
                out.push(row?);
            }
        }
        Ok(out)
    }

    // --- Web Search Cache ---

    pub fn get_web_search_cache(&self, query_hash: &str) -> Result<Option<WebSearchCacheRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT query_hash, query, results_json, results_count, cached_at, ttl_seconds
             FROM web_search_cache WHERE query_hash = ?1",
        )?;
        let mut rows = stmt.query([query_hash])?;
        if let Some(row) = rows.next()? {
            let cached_at: String = row.get(4)?;
            let ttl: i64 = row.get(5)?;
            // Check if cache is still valid
            if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&cached_at) {
                let age = Utc::now().signed_duration_since(ts.with_timezone(&Utc));
                if age.num_seconds() > ttl {
                    // Expired
                    conn.execute(
                        "DELETE FROM web_search_cache WHERE query_hash = ?1",
                        [query_hash],
                    )?;
                    return Ok(None);
                }
            }
            return Ok(Some(WebSearchCacheRecord {
                query_hash: row.get(0)?,
                query: row.get(1)?,
                results_json: row.get(2)?,
                results_count: row.get::<_, i64>(3)? as u64,
                cached_at,
                ttl_seconds: ttl as u64,
            }));
        }
        Ok(None)
    }

    pub fn set_web_search_cache(&self, record: &WebSearchCacheRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO web_search_cache (query_hash, query, results_json, results_count, cached_at, ttl_seconds)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                record.query_hash,
                record.query,
                record.results_json,
                record.results_count as i64,
                record.cached_at,
                record.ttl_seconds as i64,
            ],
        )?;
        Ok(())
    }

    // --- Review Runs ---

    pub fn insert_review_run(&self, record: &ReviewRunRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO review_runs (review_id, session_id, preset, target, findings_json, findings_count, critical_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                record.review_id.to_string(),
                record.session_id.to_string(),
                record.preset,
                record.target,
                record.findings_json,
                record.findings_count as i64,
                record.critical_count as i64,
                record.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_review_runs(&self, session_id: Uuid) -> Result<Vec<ReviewRunRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT review_id, session_id, preset, target, findings_json, findings_count, critical_count, created_at
             FROM review_runs WHERE session_id = ?1 ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([session_id.to_string()], |r| {
            Ok(ReviewRunRecord {
                review_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                preset: r.get(2)?,
                target: r.get(3)?,
                findings_json: r.get(4)?,
                findings_count: r.get::<_, i64>(5)? as u64,
                critical_count: r.get::<_, i64>(6)? as u64,
                created_at: r.get(7)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // --- Artifacts ---

    pub fn insert_artifact(&self, record: &ArtifactRecord) -> Result<()> {
        let conn = self.db()?;
        conn.execute(
            "INSERT OR REPLACE INTO artifacts (artifact_id, task_id, artifact_path, files_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.artifact_id.to_string(),
                record.task_id.to_string(),
                record.artifact_path,
                record.files_json,
                record.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_artifacts_for_task(&self, task_id: Uuid) -> Result<Vec<ArtifactRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT artifact_id, task_id, artifact_path, files_json, created_at
             FROM artifacts WHERE task_id = ?1 ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([task_id.to_string()], |r| {
            Ok(ArtifactRecord {
                artifact_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                task_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                artifact_path: r.get(2)?,
                files_json: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn list_all_artifacts(&self, limit: usize) -> Result<Vec<ArtifactRecord>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT artifact_id, task_id, artifact_path, files_json, created_at
             FROM artifacts ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok(ArtifactRecord {
                artifact_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                task_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str())
                    .unwrap_or_else(|_| Uuid::nil()),
                artifact_path: r.get(2)?,
                files_json: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // --- Sessions by workspace (for --continue) ---

    pub fn load_latest_session_for_workspace(
        &self,
        workspace_root: &str,
    ) -> Result<Option<Session>> {
        let conn = self.db()?;
        let mut stmt = conn.prepare(
            "SELECT session_id, workspace_root, baseline_commit, status, budgets, active_plan_id
             FROM sessions WHERE workspace_root = ?1 ORDER BY updated_at DESC LIMIT 1",
        )?;
        let mut rows = stmt.query([workspace_root])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(Session {
                session_id: Uuid::parse_str(row.get::<_, String>(0)?.as_str())?,
                workspace_root: row.get(1)?,
                baseline_commit: row.get(2)?,
                status: serde_json::from_str(&row.get::<_, String>(3)?)?,
                budgets: serde_json::from_str(&row.get::<_, String>(4)?)?,
                active_plan_id: row
                    .get::<_, Option<String>>(5)?
                    .map(|v| Uuid::parse_str(&v))
                    .transpose()?,
            }));
        }
        Ok(None)
    }

    pub fn rebuild_from_events(&self, session_id: Uuid) -> Result<RebuildProjection> {
        if !self.events_path.exists() {
            return Ok(RebuildProjection::default());
        }
        let file = OpenOptions::new().read(true).open(&self.events_path)?;
        let reader = BufReader::new(file);

        let mut projection = RebuildProjection::default();
        for line in reader.lines() {
            let line = line?;
            let event: EventEnvelope = serde_json::from_str(&line)?;
            if event.session_id != session_id {
                continue;
            }
            apply_projection(&mut projection, &event);
        }

        Ok(projection)
    }

    fn init_db(&self) -> Result<()> {
        let conn = self.db()?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
             );",
        )?;

        for (version, sql) in MIGRATIONS {
            let already: i64 = conn.query_row(
                "SELECT COUNT(1) FROM schema_migrations WHERE version = ?1",
                [*version],
                |r| r.get(0),
            )?;
            if already == 0 {
                conn.execute_batch(sql)?;
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?1, ?2)",
                    params![version, Utc::now().to_rfc3339()],
                )?;
            }
        }
        Ok(())
    }

    fn project_event(&self, conn: &Connection, event: &EventEnvelope) -> Result<()> {
        match &event.kind {
            EventKind::SessionStateChangedV1 { to, .. } => {
                conn.execute(
                    "UPDATE sessions SET status = ?1, updated_at = ?2 WHERE session_id = ?3",
                    params![
                        serde_json::to_string(to)?,
                        Utc::now().to_rfc3339(),
                        event.session_id.to_string()
                    ],
                )?;
            }
            EventKind::PlanCreatedV1 { plan } | EventKind::PlanRevisedV1 { plan } => {
                self.save_plan(event.session_id, plan)?;
            }
            EventKind::ToolApprovedV1 { invocation_id } => {
                conn.execute(
                    "INSERT INTO approvals_ledger (session_id, invocation_id, approved_at) VALUES (?1, ?2, ?3)",
                    params![
                        event.session_id.to_string(),
                        invocation_id.to_string(),
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::VerificationRunV1 {
                command,
                success,
                output,
            } => {
                conn.execute(
                    "INSERT INTO verification_runs (session_id, command, success, output, run_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        event.session_id.to_string(),
                        command,
                        if *success { 1 } else { 0 },
                        output,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::RouterDecisionV1 { decision } => {
                conn.execute(
                    "INSERT INTO router_stats (session_id, decision_id, selected_model, score, reasons, recorded_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        event.session_id.to_string(),
                        decision.decision_id.to_string(),
                        decision.selected_model,
                        decision.score,
                        decision.reason_codes.join(","),
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::UsageUpdatedV1 {
                unit,
                model,
                input_tokens,
                output_tokens,
            } => {
                conn.execute(
                    "INSERT INTO usage_ledger (session_id, unit, model, input_tokens, output_tokens, recorded_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        event.session_id.to_string(),
                        serde_json::to_string(unit)?,
                        model,
                        *input_tokens as i64,
                        *output_tokens as i64,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::ContextCompactedV1 {
                summary_id,
                from_turn,
                to_turn,
                token_delta_estimate,
                replay_pointer,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO context_compactions
                     (summary_id, session_id, from_turn, to_turn, token_delta_estimate, replay_pointer, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        summary_id.to_string(),
                        event.session_id.to_string(),
                        *from_turn as i64,
                        *to_turn as i64,
                        *token_delta_estimate,
                        replay_pointer,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::AutopilotRunStartedV1 { run_id, prompt } => {
                conn.execute(
                    "INSERT OR REPLACE INTO autopilot_runs
                     (run_id, session_id, prompt, status, stop_reason, completed_iterations, failed_iterations,
                      consecutive_failures, last_error, stop_file, heartbeat_file, tools, max_think, started_at, updated_at)
                     VALUES (?1, ?2, ?3, 'running', NULL, 0, 0, 0, NULL, '', '', 0, 0, ?4, ?5)",
                    params![
                        run_id.to_string(),
                        event.session_id.to_string(),
                        prompt,
                        Utc::now().to_rfc3339(),
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::AutopilotRunHeartbeatV1 {
                run_id,
                completed_iterations,
                failed_iterations,
                consecutive_failures,
                last_error,
            } => {
                conn.execute(
                    "UPDATE autopilot_runs
                     SET completed_iterations = ?1, failed_iterations = ?2, consecutive_failures = ?3, last_error = ?4, updated_at = ?5
                     WHERE run_id = ?6",
                    params![
                        *completed_iterations as i64,
                        *failed_iterations as i64,
                        *consecutive_failures as i64,
                        last_error,
                        Utc::now().to_rfc3339(),
                        run_id.to_string(),
                    ],
                )?;
            }
            EventKind::AutopilotRunStoppedV1 {
                run_id,
                stop_reason,
                completed_iterations,
                failed_iterations,
            } => {
                conn.execute(
                    "UPDATE autopilot_runs
                     SET status = 'stopped', stop_reason = ?1, completed_iterations = ?2, failed_iterations = ?3, updated_at = ?4
                     WHERE run_id = ?5",
                    params![
                        stop_reason,
                        *completed_iterations as i64,
                        *failed_iterations as i64,
                        Utc::now().to_rfc3339(),
                        run_id.to_string(),
                    ],
                )?;
            }
            EventKind::HookExecutedV1 {
                phase,
                hook_path,
                success,
                timed_out,
                exit_code,
            } => {
                conn.execute(
                    "INSERT INTO hook_executions (session_id, phase, hook_path, success, timed_out, exit_code, recorded_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        event.session_id.to_string(),
                        phase,
                        hook_path,
                        if *success { 1 } else { 0 },
                        if *timed_out { 1 } else { 0 },
                        exit_code,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::CheckpointCreatedV1 {
                checkpoint_id,
                reason,
                files_count,
                snapshot_path,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoints (checkpoint_id, reason, snapshot_path, files_count, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        checkpoint_id.to_string(),
                        reason,
                        snapshot_path,
                        *files_count as i64,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::CheckpointRewoundV1 {
                checkpoint_id,
                reason,
            } => {
                conn.execute(
                    "UPDATE checkpoints SET reason = ?1 WHERE checkpoint_id = ?2",
                    params![reason, checkpoint_id.to_string()],
                )?;
            }
            EventKind::TranscriptExportedV1 {
                export_id,
                format,
                output_path,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO transcript_exports (export_id, session_id, format, output_path, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        export_id.to_string(),
                        event.session_id.to_string(),
                        format,
                        output_path,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::McpServerAddedV1 {
                server_id,
                transport,
                endpoint,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO mcp_servers (server_id, name, transport, endpoint, enabled, metadata_json, updated_at)
                     VALUES (?1, ?2, ?3, ?4, 1, '{}', ?5)",
                    params![
                        server_id,
                        server_id,
                        transport,
                        endpoint,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::McpServerRemovedV1 { server_id } => {
                conn.execute("DELETE FROM mcp_servers WHERE server_id = ?1", [server_id])?;
                conn.execute(
                    "DELETE FROM mcp_tools_cache WHERE server_id = ?1",
                    [server_id],
                )?;
            }
            EventKind::McpToolDiscoveredV1 {
                server_id,
                tool_name,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO mcp_tools_cache (server_id, tool_name, description, schema_json, updated_at)
                     VALUES (?1, ?2, '', '{}', ?3)",
                    params![server_id, tool_name, Utc::now().to_rfc3339()],
                )?;
            }
            EventKind::SubagentSpawnedV1 { run_id, name, goal } => {
                conn.execute(
                    "INSERT OR REPLACE INTO subagent_runs (run_id, name, goal, status, output, error, created_at, updated_at)
                     VALUES (?1, ?2, ?3, 'running', NULL, NULL, ?4, ?5)",
                    params![
                        run_id.to_string(),
                        name,
                        goal,
                        Utc::now().to_rfc3339(),
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::SubagentCompletedV1 { run_id, output } => {
                conn.execute(
                    "UPDATE subagent_runs SET status='completed', output=?1, error=NULL, updated_at=?2 WHERE run_id=?3",
                    params![output, Utc::now().to_rfc3339(), run_id.to_string()],
                )?;
            }
            EventKind::SubagentFailedV1 { run_id, error } => {
                conn.execute(
                    "UPDATE subagent_runs SET status='failed', output=NULL, error=?1, updated_at=?2 WHERE run_id=?3",
                    params![error, Utc::now().to_rfc3339(), run_id.to_string()],
                )?;
            }
            EventKind::CostUpdatedV1 {
                input_tokens,
                output_tokens,
                estimated_cost_usd,
            } => {
                conn.execute(
                    "INSERT INTO cost_ledger (session_id, input_tokens, output_tokens, estimated_cost_usd, recorded_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        event.session_id.to_string(),
                        *input_tokens as i64,
                        *output_tokens as i64,
                        *estimated_cost_usd,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::ProfileCapturedV1 {
                profile_id,
                summary,
                elapsed_ms,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO profile_runs (profile_id, session_id, summary, elapsed_ms, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        profile_id.to_string(),
                        event.session_id.to_string(),
                        summary,
                        *elapsed_ms as i64,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::MemorySyncedV1 {
                version_id,
                path,
                note,
            } => {
                let content = fs::read_to_string(path).unwrap_or_default();
                conn.execute(
                    "INSERT OR REPLACE INTO memory_versions (version_id, path, content, note, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        version_id.to_string(),
                        path,
                        content,
                        note,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::BackgroundJobStartedV1 {
                job_id,
                kind,
                reference,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO background_jobs (job_id, kind, reference, status, metadata_json, started_at, updated_at)
                     VALUES (?1, ?2, ?3, 'running', '{}', ?4, ?5)",
                    params![
                        job_id.to_string(),
                        kind,
                        reference,
                        Utc::now().to_rfc3339(),
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::BackgroundJobResumedV1 { job_id, reference } => {
                conn.execute(
                    "UPDATE background_jobs SET status='running', reference=?1, updated_at=?2 WHERE job_id=?3",
                    params![reference, Utc::now().to_rfc3339(), job_id.to_string()],
                )?;
            }
            EventKind::BackgroundJobStoppedV1 { job_id, reason } => {
                conn.execute(
                    "UPDATE background_jobs SET status='stopped', metadata_json=?1, updated_at=?2 WHERE job_id=?3",
                    params![
                        serde_json::json!({"reason": reason}).to_string(),
                        Utc::now().to_rfc3339(),
                        job_id.to_string()
                    ],
                )?;
            }
            EventKind::SkillLoadedV1 {
                skill_id,
                source_path,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO skill_registry (skill_id, name, path, enabled, metadata_json, updated_at)
                     VALUES (?1, ?2, ?3, 1, '{}', ?4)",
                    params![skill_id, skill_id, source_path, Utc::now().to_rfc3339()],
                )?;
            }
            EventKind::ReplayExecutedV1 {
                session_id,
                deterministic,
                events_replayed,
            } => {
                conn.execute(
                    "INSERT INTO replay_cassettes (cassette_id, session_id, deterministic, events_count, payload_json, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        Uuid::now_v7().to_string(),
                        session_id.to_string(),
                        if *deterministic { 1 } else { 0 },
                        *events_replayed as i64,
                        "{}",
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::PromptCacheHitV1 { cache_key, model } => {
                conn.execute(
                    "INSERT INTO provider_metrics (provider, model, cache_key, cache_hit, latency_ms, recorded_at)
                     VALUES ('deepseek', ?1, ?2, 1, 0, ?3)",
                    params![model, cache_key, Utc::now().to_rfc3339()],
                )?;
            }
            EventKind::OffPeakScheduledV1 {
                reason,
                resume_after,
            } => {
                conn.execute(
                    "INSERT INTO provider_metrics (provider, model, cache_key, cache_hit, latency_ms, recorded_at)
                     VALUES ('scheduler', 'off_peak', ?1, 0, 0, ?2)",
                    params![format!("{reason}:{resume_after}"), Utc::now().to_rfc3339()],
                )?;
            }
            EventKind::VisualArtifactCapturedV1 {
                artifact_id,
                path,
                mime,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO visual_artifacts (artifact_id, path, mime, metadata_json, created_at)
                     VALUES (?1, ?2, ?3, '{}', ?4)",
                    params![
                        artifact_id.to_string(),
                        path,
                        mime,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::RemoteEnvConfiguredV1 {
                profile_id,
                name,
                endpoint,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO remote_env_profiles (profile_id, name, endpoint, auth_mode, metadata_json, updated_at)
                     VALUES (?1, ?2, ?3, 'token', '{}', ?4)",
                    params![
                        profile_id.to_string(),
                        name,
                        endpoint,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::TeleportBundleCreatedV1 { bundle_id, path } => {
                conn.execute(
                    "INSERT INTO replay_cassettes (cassette_id, session_id, deterministic, events_count, payload_json, created_at)
                     VALUES (?1, ?2, 1, 0, ?3, ?4)",
                    params![
                        bundle_id.to_string(),
                        event.session_id.to_string(),
                        serde_json::json!({"path": path}).to_string(),
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::PermissionModeChangedV1 { from, to } => {
                conn.execute(
                    "INSERT INTO permission_mode_log (session_id, from_mode, to_mode, changed_at)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![
                        event.session_id.to_string(),
                        from,
                        to,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::TaskCreatedV1 {
                task_id,
                title,
                priority,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO task_queue (task_id, session_id, title, priority, status, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, 'pending', ?5, ?6)",
                    params![
                        task_id.to_string(),
                        event.session_id.to_string(),
                        title,
                        *priority as i64,
                        Utc::now().to_rfc3339(),
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::TaskCompletedV1 { task_id, outcome } => {
                conn.execute(
                    "UPDATE task_queue SET status = 'completed', outcome = ?1, updated_at = ?2 WHERE task_id = ?3",
                    params![outcome, Utc::now().to_rfc3339(), task_id.to_string()],
                )?;
            }
            EventKind::ReviewStartedV1 {
                review_id,
                preset,
                target,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO review_runs (review_id, session_id, preset, target, findings_json, findings_count, critical_count, created_at)
                     VALUES (?1, ?2, ?3, ?4, '[]', 0, 0, ?5)",
                    params![
                        review_id.to_string(),
                        event.session_id.to_string(),
                        preset,
                        target,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            EventKind::ReviewCompletedV1 {
                review_id,
                findings_count,
                critical_count,
            } => {
                conn.execute(
                    "UPDATE review_runs SET findings_count = ?1, critical_count = ?2 WHERE review_id = ?3",
                    params![
                        *findings_count as i64,
                        *critical_count as i64,
                        review_id.to_string()
                    ],
                )?;
            }
            EventKind::ArtifactBundledV1 {
                task_id,
                artifact_path,
                files,
            } => {
                conn.execute(
                    "INSERT OR REPLACE INTO artifacts (artifact_id, task_id, artifact_path, files_json, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        Uuid::now_v7().to_string(),
                        task_id.to_string(),
                        artifact_path,
                        serde_json::to_string(files)?,
                        Utc::now().to_rfc3339(),
                    ],
                )?;
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct RebuildProjection {
    pub transcript: Vec<String>,
    pub latest_plan: Option<Plan>,
    pub step_status: Vec<(Uuid, bool, String)>,
    pub router_models: Vec<String>,
    pub staged_patches: Vec<Uuid>,
    pub applied_patches: Vec<Uuid>,
    pub tool_invocations: Vec<Uuid>,
    pub approved_invocations: Vec<Uuid>,
    pub state: Option<SessionState>,
    pub plugin_events: Vec<String>,
    pub usage_input_tokens: u64,
    pub usage_output_tokens: u64,
    pub compaction_events: usize,
    pub autopilot_runs: Vec<Uuid>,
    pub permission_mode: Option<String>,
    pub task_ids: Vec<Uuid>,
    pub review_ids: Vec<Uuid>,
}

fn apply_projection(proj: &mut RebuildProjection, event: &EventEnvelope) {
    match &event.kind {
        EventKind::TurnAddedV1 { role, content } => {
            proj.transcript.push(format!("{role}: {content}"))
        }
        EventKind::PlanCreatedV1 { plan } | EventKind::PlanRevisedV1 { plan } => {
            proj.latest_plan = Some(plan.clone())
        }
        EventKind::StepMarkedV1 {
            step_id,
            done,
            note,
        } => proj.step_status.push((*step_id, *done, note.clone())),
        EventKind::RouterDecisionV1 { decision } => {
            proj.router_models.push(decision.selected_model.clone())
        }
        EventKind::PatchStagedV1 { patch_id, .. } => proj.staged_patches.push(*patch_id),
        EventKind::PatchAppliedV1 {
            patch_id, applied, ..
        } => {
            if *applied {
                proj.applied_patches.push(*patch_id)
            }
        }
        EventKind::ToolProposedV1 { proposal } => {
            proj.tool_invocations.push(proposal.invocation_id)
        }
        EventKind::ToolApprovedV1 { invocation_id } => {
            proj.approved_invocations.push(*invocation_id)
        }
        EventKind::SessionStateChangedV1 { to, .. } => proj.state = Some(to.clone()),
        EventKind::PluginInstalledV1 { plugin_id, .. }
        | EventKind::PluginRemovedV1 { plugin_id }
        | EventKind::PluginEnabledV1 { plugin_id }
        | EventKind::PluginDisabledV1 { plugin_id } => proj.plugin_events.push(plugin_id.clone()),
        EventKind::UsageUpdatedV1 {
            input_tokens,
            output_tokens,
            ..
        } => {
            proj.usage_input_tokens = proj.usage_input_tokens.saturating_add(*input_tokens);
            proj.usage_output_tokens = proj.usage_output_tokens.saturating_add(*output_tokens);
        }
        EventKind::ContextCompactedV1 { .. } => {
            proj.compaction_events = proj.compaction_events.saturating_add(1)
        }
        EventKind::AutopilotRunStartedV1 { run_id, .. } => proj.autopilot_runs.push(*run_id),
        EventKind::PermissionModeChangedV1 { to, .. } => proj.permission_mode = Some(to.clone()),
        EventKind::TaskCreatedV1 { task_id, .. } => proj.task_ids.push(*task_id),
        EventKind::ReviewStartedV1 { review_id, .. } => proj.review_ids.push(*review_id),
        _ => {}
    }
}

fn event_kind_name(kind: &EventKind) -> &'static str {
    match kind {
        EventKind::TurnAddedV1 { .. } => "TurnAdded@v1",
        EventKind::SessionStateChangedV1 { .. } => "SessionStateChanged@v1",
        EventKind::PlanCreatedV1 { .. } => "PlanCreated@v1",
        EventKind::PlanRevisedV1 { .. } => "PlanRevised@v1",
        EventKind::StepMarkedV1 { .. } => "StepMarked@v1",
        EventKind::RouterDecisionV1 { .. } => "RouterDecision@v1",
        EventKind::RouterEscalationV1 { .. } => "RouterEscalation@v1",
        EventKind::ToolProposedV1 { .. } => "ToolProposed@v1",
        EventKind::ToolApprovedV1 { .. } => "ToolApproved@v1",
        EventKind::ToolResultV1 { .. } => "ToolResult@v1",
        EventKind::PatchStagedV1 { .. } => "PatchStaged@v1",
        EventKind::PatchAppliedV1 { .. } => "PatchApplied@v1",
        EventKind::VerificationRunV1 { .. } => "VerificationRun@v1",
        EventKind::PluginInstalledV1 { .. } => "PluginInstalled@v1",
        EventKind::PluginRemovedV1 { .. } => "PluginRemoved@v1",
        EventKind::PluginEnabledV1 { .. } => "PluginEnabled@v1",
        EventKind::PluginDisabledV1 { .. } => "PluginDisabled@v1",
        EventKind::UsageUpdatedV1 { .. } => "UsageUpdated@v1",
        EventKind::ContextCompactedV1 { .. } => "ContextCompacted@v1",
        EventKind::AutopilotRunStartedV1 { .. } => "AutopilotRunStarted@v1",
        EventKind::AutopilotRunHeartbeatV1 { .. } => "AutopilotRunHeartbeat@v1",
        EventKind::AutopilotRunStoppedV1 { .. } => "AutopilotRunStopped@v1",
        EventKind::PluginCatalogSyncedV1 { .. } => "PluginCatalogSynced@v1",
        EventKind::PluginVerifiedV1 { .. } => "PluginVerified@v1",
        EventKind::HookExecutedV1 { .. } => "HookExecuted@v1",
        EventKind::SessionForkedV1 { .. } => "SessionForked@v1",
        EventKind::CheckpointCreatedV1 { .. } => "CheckpointCreated@v1",
        EventKind::CheckpointRewoundV1 { .. } => "CheckpointRewound@v1",
        EventKind::TranscriptExportedV1 { .. } => "TranscriptExported@v1",
        EventKind::McpServerAddedV1 { .. } => "McpServerAdded@v1",
        EventKind::McpServerRemovedV1 { .. } => "McpServerRemoved@v1",
        EventKind::McpToolDiscoveredV1 { .. } => "McpToolDiscovered@v1",
        EventKind::SubagentSpawnedV1 { .. } => "SubagentSpawned@v1",
        EventKind::SubagentCompletedV1 { .. } => "SubagentCompleted@v1",
        EventKind::SubagentFailedV1 { .. } => "SubagentFailed@v1",
        EventKind::CostUpdatedV1 { .. } => "CostUpdated@v1",
        EventKind::EffortChangedV1 { .. } => "EffortChanged@v1",
        EventKind::ProfileCapturedV1 { .. } => "ProfileCaptured@v1",
        EventKind::MemorySyncedV1 { .. } => "MemorySynced@v1",
        EventKind::BackgroundJobStartedV1 { .. } => "BackgroundJobStarted@v1",
        EventKind::BackgroundJobResumedV1 { .. } => "BackgroundJobResumed@v1",
        EventKind::BackgroundJobStoppedV1 { .. } => "BackgroundJobStopped@v1",
        EventKind::SkillLoadedV1 { .. } => "SkillLoaded@v1",
        EventKind::ReplayExecutedV1 { .. } => "ReplayExecuted@v1",
        EventKind::PromptCacheHitV1 { .. } => "PromptCacheHit@v1",
        EventKind::OffPeakScheduledV1 { .. } => "OffPeakScheduled@v1",
        EventKind::VisualArtifactCapturedV1 { .. } => "VisualArtifactCaptured@v1",
        EventKind::RemoteEnvConfiguredV1 { .. } => "RemoteEnvConfigured@v1",
        EventKind::TeleportBundleCreatedV1 { .. } => "TeleportBundleCreated@v1",
        EventKind::TelemetryEventV1 { .. } => "TelemetryEvent@v1",
        EventKind::PermissionModeChangedV1 { .. } => "PermissionModeChanged@v1",
        EventKind::WebSearchExecutedV1 { .. } => "WebSearchExecuted@v1",
        EventKind::ReviewStartedV1 { .. } => "ReviewStarted@v1",
        EventKind::ReviewCompletedV1 { .. } => "ReviewCompleted@v1",
        EventKind::TaskCreatedV1 { .. } => "TaskCreated@v1",
        EventKind::TaskCompletedV1 { .. } => "TaskCompleted@v1",
        EventKind::ArtifactBundledV1 { .. } => "ArtifactBundled@v1",
        EventKind::SessionStartedV1 { .. } => "SessionStarted@v1",
        EventKind::SessionResumedV1 { .. } => "SessionResumed@v1",
        EventKind::ToolDeniedV1 { .. } => "ToolDenied@v1",
        EventKind::NotebookEditedV1 { .. } => "NotebookEdited@v1",
        EventKind::PdfTextExtractedV1 { .. } => "PdfTextExtracted@v1",
        EventKind::IdeSessionStartedV1 { .. } => "IdeSessionStarted@v1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_core::{SessionBudgets, SessionState};

    #[test]
    fn rebuild_projection_from_events_is_deterministic() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-store-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("temp workspace");
        let store = Store::new(&workspace).expect("store");
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 30,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        store.save_session(&session).expect("save session");

        let ev = EventEnvelope {
            seq_no: 1,
            at: Utc::now(),
            session_id: session.session_id,
            kind: EventKind::TurnAddedV1 {
                role: "user".to_string(),
                content: "hello".to_string(),
            },
        };
        store.append_event(&ev).expect("append");
        let p1 = store
            .rebuild_from_events(session.session_id)
            .expect("rebuild 1");
        let p2 = store
            .rebuild_from_events(session.session_id)
            .expect("rebuild 2");
        assert_eq!(p1.transcript, p2.transcript);
    }

    #[test]
    fn rebuild_handles_mixed_old_and_new_event_types() {
        let workspace =
            std::env::temp_dir().join(format!("deepseek-store-test-{}", Uuid::now_v7()));
        fs::create_dir_all(&workspace).expect("temp workspace");
        let store = Store::new(&workspace).expect("store");
        let session = Session {
            session_id: Uuid::now_v7(),
            workspace_root: workspace.to_string_lossy().to_string(),
            baseline_commit: None,
            status: SessionState::Idle,
            budgets: SessionBudgets {
                per_turn_seconds: 30,
                max_think_tokens: 1000,
            },
            active_plan_id: None,
        };
        store.save_session(&session).expect("save session");

        store
            .append_event(&EventEnvelope {
                seq_no: 1,
                at: Utc::now(),
                session_id: session.session_id,
                kind: EventKind::TurnAddedV1 {
                    role: "user".to_string(),
                    content: "compat".to_string(),
                },
            })
            .expect("append old");
        store
            .append_event(&EventEnvelope {
                seq_no: 2,
                at: Utc::now(),
                session_id: session.session_id,
                kind: EventKind::PluginEnabledV1 {
                    plugin_id: "demo".to_string(),
                },
            })
            .expect("append new");

        let rebuilt = store
            .rebuild_from_events(session.session_id)
            .expect("rebuild mixed");
        assert_eq!(rebuilt.transcript.len(), 1);
        assert_eq!(rebuilt.plugin_events, vec!["demo".to_string()]);
    }
}
