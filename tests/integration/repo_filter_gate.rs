//! `allow_repositories` / `exclude_repositories` must keep git-ai from writing
//! authorship notes for a filtered repository and from pushing notes to a
//! remote the filters reject.
//!
//! Remotes use real-looking URLs (the filters match on remote URLs) whose
//! transport is rewritten to local bare repos with `url.<path>.insteadOf`.

use crate::repos::test_file::ExpectedLineExt;
use crate::repos::test_repo::{DaemonTestScope, TestRepo};
use crate::test_utils::isolated_metrics_db_path;
use git_ai::authorship::authorship_log_serialization::generate_session_id;
use git_ai::metrics::db::MetricsDatabase;
use git_ai::metrics::{EventAttributes, MetricEvent, PosEncoded, SessionEventValues};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::time::UNIX_EPOCH;

const GITHUB_URL: &str = "https://github.com/acme/filtered.git";
const GITLAB_URL: &str = "https://gitlab.example.com/acme/filtered.git";

/// Merge repository filters into the git-ai config the CLI and the daemon read.
fn set_repository_filters(repo: &TestRepo, allow: &[&str], exclude: &[&str]) {
    let mut homes = vec![repo.test_home_path().clone(), repo.daemon_home_path()];
    homes.dedup();
    for home in homes {
        let config_path = home.join(".git-ai/config.json");
        let mut config: serde_json::Map<String, serde_json::Value> =
            fs::read_to_string(&config_path)
                .ok()
                .and_then(|raw| serde_json::from_str(&raw).ok())
                .unwrap_or_default();
        config.insert("allow_repositories".to_string(), json!(allow));
        config.insert("exclude_repositories".to_string(), json!(exclude));
        fs::create_dir_all(config_path.parent().unwrap()).unwrap();
        fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();
    }
}

/// Point remote `name` at `url` while sending its transport to `bare`.
fn set_remote(repo: &TestRepo, name: &str, url: &str, bare: &Path) {
    if repo.git_og(&["remote", "get-url", name]).is_ok() {
        repo.git_og(&["remote", "set-url", name, url]).unwrap();
    } else {
        repo.git_og(&["remote", "add", name, url]).unwrap();
    }
    repo.git_og(&[
        "config",
        &format!("url.{}.insteadOf", bare.to_str().unwrap()),
        url,
    ])
    .unwrap();
}

fn notes_refs(repo: &TestRepo) -> String {
    repo.git_og(&["for-each-ref", "--format=%(refname)", "refs/notes/"])
        .unwrap()
        .trim()
        .to_string()
}

fn head_sha(repo: &TestRepo) -> String {
    repo.git_og(&["rev-parse", "HEAD"])
        .unwrap()
        .trim()
        .to_string()
}

fn commit_all(repo: &TestRepo, message: &str) -> String {
    repo.git(&["add", "-A"]).unwrap();
    repo.git(&["commit", "-m", message]).unwrap();
    head_sha(repo)
}

fn filtered_repo_with_remote() -> (TestRepo, TestRepo) {
    let (local, upstream) = TestRepo::new_with_remote_with_daemon_scope(DaemonTestScope::Dedicated);
    set_remote(&local, "origin", GITHUB_URL, upstream.path());
    set_repository_filters(&local, &[], &["https://github.com/*"]);
    (local, upstream)
}

#[test]
fn excluded_repository_commit_writes_no_authorship_note() {
    let (local, _upstream) = filtered_repo_with_remote();

    fs::write(local.path().join("human.txt"), "written by a human\n").unwrap();
    let sha = commit_all(&local, "human change in excluded repo");

    assert_eq!(
        local.read_authorship_note(&sha),
        None,
        "no authorship note may be written in a repository matched by exclude_repositories"
    );
}

#[test]
fn excluded_repository_checkpoint_is_skipped_and_commit_writes_no_note() {
    let (local, _upstream) = filtered_repo_with_remote();

    let mut file = local.filename("ai.txt");
    file.set_contents(vec!["line from an agent".ai()]);
    let sha = commit_all(&local, "agent change in excluded repo");

    assert_eq!(
        local.read_authorship_note(&sha),
        None,
        "the checkpoint was skipped by the filter; post-commit must not write a note anyway"
    );
}

#[test]
fn excluded_repository_push_sends_no_notes_to_remote() {
    let (local, upstream) = filtered_repo_with_remote();

    fs::write(local.path().join("human.txt"), "written by a human\n").unwrap();
    commit_all(&local, "change in excluded repo");
    local.git(&["push", "origin", "HEAD:main"]).unwrap();
    local.sync_daemon_force();

    assert_eq!(
        notes_refs(&upstream),
        "",
        "push to an excluded remote must not sync refs/notes/ai"
    );
}

/// Filters are evaluated per repository (any matching remote allows it), but a
/// push targets one remote. A repo allowed via its GitLab origin must still
/// not push notes to a GitHub remote that the allowlist does not cover.
#[test]
fn push_to_remote_outside_allowlist_sends_no_notes_even_if_repo_is_allowed() {
    let (local, github) = TestRepo::new_with_remote_with_daemon_scope(DaemonTestScope::Dedicated);
    let gitlab = TestRepo::new_bare_with_daemon_scope(DaemonTestScope::NoDaemon);
    set_remote(&local, "origin", GITLAB_URL, gitlab.path());
    set_remote(&local, "upstream", GITHUB_URL, github.path());
    set_repository_filters(&local, &["https://gitlab.example.com/*"], &[]);

    let mut file = local.filename("ai.txt");
    file.set_contents(vec!["line from an agent".ai()]);
    let sha = commit_all(&local, "agent change in allowed repo");
    assert!(
        local.read_authorship_note(&sha).is_some(),
        "precondition: the repo is allowed via origin, so a note is expected locally"
    );

    local.git(&["push", "upstream", "HEAD:main"]).unwrap();
    local.sync_daemon_force();

    assert_eq!(
        notes_refs(&github),
        "",
        "notes must not be pushed to a remote outside allow_repositories"
    );
}

fn file_mtime_secs(path: &Path) -> u32 {
    fs::metadata(path)
        .unwrap()
        .modified()
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .min(u32::MAX as u64) as u32
}

/// Transcript streaming keeps recording SessionEvents for excluded repos (they
/// are only dropped at upload). Post-commit attribution recovery must not turn
/// them into an AI-attributed note for the excluded repo.
#[test]
fn excluded_repository_commit_does_not_recover_ai_session_into_note() {
    let (_metrics_db_dir, metrics_db_path) = isolated_metrics_db_path();
    let local =
        TestRepo::new_with_daemon_env(&[("GIT_AI_TEST_METRICS_DB_PATH", metrics_db_path.as_str())]);
    local
        .git_og(&["remote", "add", "origin", GITHUB_URL])
        .unwrap();
    set_repository_filters(&local, &[], &["https://github.com/*"]);
    local
        .git(&["commit", "--allow-empty", "-m", "initial"])
        .unwrap();

    let file_path = local.path().join("generated.txt");
    fs::write(&file_path, "generated by an agent\n").unwrap();
    let external_session_id = "cursor-session-in-excluded-repo";
    let session_id = generate_session_id(external_session_id, "cursor");
    let event = MetricEvent::from_values_with_timestamp(
        SessionEventValues::with_ids(
            json!({ "type": "assistant", "session_id": external_session_id }),
            Some("event-excluded".to_string()),
            None,
            Some("tool-use-excluded".to_string()),
        ),
        EventAttributes::with_version("test")
            .tool("cursor")
            .model("claude-sonnet-5")
            .external_session_id(external_session_id)
            .session_id(&session_id)
            .trace_id("trace-excluded")
            .repo_url("https://github.com/acme/filtered")
            .to_sparse(),
        Some(file_mtime_secs(&file_path)),
    );
    MetricsDatabase::open_at_path(Path::new(&metrics_db_path))
        .unwrap()
        .insert_events(&[serde_json::to_string(&event).unwrap()])
        .unwrap();

    let sha = commit_all(&local, "uncheckpointed agent change in excluded repo");

    let note = local.read_authorship_note(&sha);
    assert_eq!(
        note,
        None,
        "no note may be written for an excluded repo; recovered session {session_id} present: {}",
        note.as_deref().is_some_and(|n| n.contains(&session_id))
    );
}
