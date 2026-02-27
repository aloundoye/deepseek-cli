use deepseek_local_ml::privacy::*;

fn make_router(policy: PrivacyPolicy) -> PrivacyRouter {
    let config = PrivacyConfig {
        enabled: true,
        policy,
        ..Default::default()
    };
    PrivacyRouter::new(config).unwrap()
}

#[test]
fn sensitive_path_detected() {
    let router = make_router(PrivacyPolicy::Redact);
    assert!(router.is_sensitive_path("project/.env"));
    assert!(router.is_sensitive_path("certs/server.pem"));
    assert!(router.is_sensitive_path("keys/id_rsa.pub"));
    assert!(!router.is_sensitive_path("src/main.rs"));
    assert!(!router.is_sensitive_path("README.md"));
}

#[test]
fn sensitive_content_detected() {
    let router = make_router(PrivacyPolicy::Redact);
    let content = "The AWS key is AKIAIOSFODNN7EXAMPLE and the secret is stored elsewhere.";
    let matches = router.scan_content(content);
    assert!(
        !matches.is_empty(),
        "AKIA pattern should be detected, found {} matches",
        matches.len()
    );
    assert!(
        matches.iter().any(|m| m.pattern == "aws_key"),
        "should match aws_key pattern"
    );
}

#[test]
fn block_cloud_prevents_sending() {
    let router = make_router(PrivacyPolicy::BlockCloud);
    let content = "SECRET_KEY=sk-1234567890abcdefghijklmnopqrstuv";
    let result = router.apply_policy(content, Some("config/.env"));
    assert_eq!(result, PrivacyResult::Blocked);
}

#[test]
fn redact_replaces_secrets() {
    let router = make_router(PrivacyPolicy::Redact);
    let content = "API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234\nNORMAL_LINE=hello\n";
    let result = router.apply_policy(content, None);
    match result {
        PrivacyResult::Redacted(redacted) => {
            assert!(
                redacted.contains("[REDACTED:"),
                "secrets should be replaced with [REDACTED:*]"
            );
            assert!(
                !redacted.contains("sk-abcdef"),
                "original secret should not appear in redacted output"
            );
            assert!(
                redacted.contains("NORMAL_LINE"),
                "non-secret content should be preserved"
            );
        }
        other => panic!("expected Redacted, got {:?}", other),
    }
}

#[test]
fn clean_content_passes() {
    let router = make_router(PrivacyPolicy::Redact);
    let content = r#"
fn main() {
    let x = 42;
    println!("Hello, world!");
    let v: Vec<i32> = vec![1, 2, 3];
}
"#;
    let result = router.apply_policy(content, Some("src/main.rs"));
    assert!(
        matches!(result, PrivacyResult::Clean(_)),
        "normal Rust code should pass as clean"
    );
}

#[test]
fn session_log_redacted_when_configured() {
    let config = PrivacyConfig {
        enabled: true,
        store_raw_in_logs: false,
        ..Default::default()
    };
    let router = PrivacyRouter::new(config).unwrap();
    assert!(
        router.should_redact_logs(),
        "logs should be redacted when store_raw_in_logs=false"
    );

    let config2 = PrivacyConfig {
        enabled: true,
        store_raw_in_logs: true,
        ..Default::default()
    };
    let router2 = PrivacyRouter::new(config2).unwrap();
    assert!(
        !router2.should_redact_logs(),
        "logs should not be redacted when store_raw_in_logs=true"
    );
}

#[test]
fn private_key_pattern_detected() {
    let router = make_router(PrivacyPolicy::Redact);
    let content =
        "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0B\n-----END PRIVATE KEY-----";
    let matches = router.scan_content(content);
    assert!(!matches.is_empty(), "private key block should be detected");
}

#[test]
fn github_token_pattern_detected() {
    let router = make_router(PrivacyPolicy::Redact);
    let content = "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij";
    let matches = router.scan_content(content);
    assert!(!matches.is_empty(), "GitHub token should be detected");
}

#[test]
fn connection_string_detected() {
    let router = make_router(PrivacyPolicy::Redact);
    let content = r#"DATABASE_URL=postgres://user:password@localhost:5432/mydb"#;
    let matches = router.scan_content(content);
    assert!(
        !matches.is_empty(),
        "postgres connection string should be detected"
    );
}
