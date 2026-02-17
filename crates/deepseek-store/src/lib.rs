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
                    summary_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str()).unwrap_or_else(|_| Uuid::nil()),
                    session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str()).unwrap_or_else(|_| Uuid::nil()),
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
                    summary_id: Uuid::parse_str(r.get::<_, String>(0)?.as_str()).unwrap_or_else(|_| Uuid::nil()),
                    session_id: Uuid::parse_str(r.get::<_, String>(1)?.as_str()).unwrap_or_else(|_| Uuid::nil()),
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
        EventKind::TelemetryEventV1 { .. } => "TelemetryEvent@v1",
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
