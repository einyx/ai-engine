//! Memory-file scanner: parses YAML-frontmatter `.md` files and `[[wikilinks]]`.
//! Hand-rolled parsing (only ~4 fields + link scan) to avoid a YAML dependency.

use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Memory {
    pub name: String,
    pub description: String,
    pub mem_type: Option<String>,
    pub origin_session: Option<String>,
    pub links: Vec<String>,
    /// First ~400 chars of the body, for the UI detail panel.
    pub body: String,
}

/// Parse a single memory file. Returns `None` if it lacks a frontmatter block
/// or a `name`.
pub(crate) fn parse_memory_file(path: &Path) -> Option<Memory> {
    let text = fs::read_to_string(path).ok()?;
    let rest = text.strip_prefix("---")?;
    let end = rest.find("\n---")?;
    let frontmatter = &rest[..end];
    let body = &rest[end + 4..];

    let mut name = None;
    let mut description = String::new();
    let mut mem_type = None;
    let mut origin_session = None;

    for line in frontmatter.lines() {
        let trimmed = line.trim();
        if let Some(v) = trimmed.strip_prefix("name:") {
            name = Some(v.trim().to_string());
        } else if let Some(v) = trimmed.strip_prefix("description:") {
            description = v.trim().to_string();
        } else if let Some(v) = trimmed.strip_prefix("type:") {
            mem_type = Some(v.trim().to_string());
        } else if let Some(v) = trimmed.strip_prefix("originSessionId:") {
            origin_session = Some(v.trim().to_string());
        }
    }

    Some(Memory {
        name: name?,
        description,
        mem_type,
        origin_session,
        links: extract_wikilinks(body),
        body: body_snippet(body),
    })
}

/// First ~400 chars of the trimmed body, for the UI detail panel.
fn body_snippet(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() <= 400 {
        trimmed.to_string()
    } else {
        let head: String = trimmed.chars().take(400).collect();
        format!("{head}…")
    }
}

/// Collect distinct `[[target]]` references in order of first appearance.
fn extract_wikilinks(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(open) = rest.find("[[") {
        rest = &rest[open + 2..];
        if let Some(close) = rest.find("]]") {
            let link = rest[..close].trim().to_string();
            if !link.is_empty() && !out.contains(&link) {
                out.push(link);
            }
            rest = &rest[close + 2..];
        } else {
            break;
        }
    }
    out
}

/// Scan all `*.md` in `dir` (skips `MEMORY.md` index). Missing dir → empty.
pub(crate) fn scan_memory_dir(dir: &Path) -> Vec<Memory> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        if path.file_name().and_then(|n| n.to_str()) == Some("MEMORY.md") {
            continue;
        }
        if let Some(m) = parse_memory_file(&path) {
            out.push(m);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixtures() -> &'static Path {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/memory"))
    }

    #[test]
    fn parses_frontmatter_fields() {
        let m = parse_memory_file(&fixtures().join("a.md")).unwrap();
        assert_eq!(m.name, "feedback-plan-docs");
        assert_eq!(m.mem_type.as_deref(), Some("feedback"));
        assert_eq!(m.origin_session.as_deref(), Some("sess-111"));
        assert_eq!(m.links, vec!["other-memory".to_string()]);
    }

    #[test]
    fn scan_dir_returns_all_memories() {
        let mems = scan_memory_dir(fixtures());
        assert_eq!(mems.len(), 2);
    }

    #[test]
    fn missing_dir_returns_empty() {
        assert!(scan_memory_dir(Path::new("/no/such/dir")).is_empty());
    }
}
