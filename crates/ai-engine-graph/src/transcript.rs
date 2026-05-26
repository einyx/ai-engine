//! Transcript scanner: one `Session` per `*.jsonl`, with tool-use counts
//! aggregated by tool name (`Bash`, `Skill`, `Agent`, ...).

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Session {
    pub session_id: String,
    pub label: String,
    /// tool name -> invocation count
    pub commands: BTreeMap<String, u32>,
}

/// Parse one transcript file. Returns `None` if no `sessionId` is found.
pub(crate) fn parse_transcript(path: &Path) -> Option<Session> {
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(file);

    let mut session_id: Option<String> = None;
    let mut label: Option<String> = None;
    let mut commands: BTreeMap<String, u32> = BTreeMap::new();

    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };

        if session_id.is_none() {
            if let Some(id) = v.get("sessionId").and_then(|x| x.as_str()) {
                session_id = Some(id.to_string());
            }
        }

        let kind = v.get("type").and_then(|x| x.as_str());
        let content = v.get("message").and_then(|m| m.get("content"));

        // First user prompt that is a plain string becomes the label.
        if kind == Some("user") && label.is_none() {
            if let Some(s) = content.and_then(|c| c.as_str()) {
                let s = s.trim();
                if !s.is_empty() {
                    label = Some(truncate(s, 60));
                }
            }
        }

        // Count tool_use blocks in assistant content lists.
        if kind == Some("assistant") {
            if let Some(blocks) = content.and_then(|c| c.as_array()) {
                for b in blocks {
                    if b.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        if let Some(name) = b.get("name").and_then(|n| n.as_str()) {
                            *commands.entry(name.to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
    }

    let session_id = session_id?;
    Some(Session {
        label: label.unwrap_or_else(|| session_id.clone()),
        session_id,
        commands,
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

/// Scan all `*.jsonl` in `dir`. Missing dir → empty.
pub(crate) fn scan_transcript_dir(dir: &Path) -> Vec<Session> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            if let Some(s) = parse_transcript(&path) {
                out.push(s);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixtures() -> &'static Path {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/transcripts"))
    }

    #[test]
    fn parses_session_label_and_command_counts() {
        let s = parse_transcript(&fixtures().join("sess-111.jsonl")).unwrap();
        assert_eq!(s.session_id, "sess-111");
        assert_eq!(s.label, "build me a graph view");
        assert_eq!(s.commands.get("Bash"), Some(&2));
        assert_eq!(s.commands.get("Skill"), Some(&1));
    }

    #[test]
    fn missing_dir_returns_empty() {
        assert!(scan_transcript_dir(Path::new("/no/such/dir")).is_empty());
    }

    #[test]
    fn scan_dir_finds_session() {
        let sessions = scan_transcript_dir(fixtures());
        assert_eq!(sessions.len(), 1);
    }
}
