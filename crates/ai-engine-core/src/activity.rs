//! In-memory ring buffer of recent gateway chat requests.
//!
//! The terminal log stage records one [`ChatEvent`] per inference request; the
//! `/graph` endpoint reads the recent window to render chat → model → provider
//! activity alongside the static knowledge graph. Bounded and best-effort —
//! purely for visualization, never persisted.

use std::collections::VecDeque;
use std::sync::Mutex;

/// One completed gateway inference request.
#[derive(Debug, Clone)]
pub struct ChatEvent {
    pub request_id: String,
    /// RFC3339 timestamp.
    pub ts: String,
    pub model: String,
    /// Provider id that actually served the request.
    pub provider: String,
    /// Output (completion) tokens, when known.
    pub tokens: u32,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// HTTP status of the response.
    pub status: u16,
    /// Short preview of the user's prompt (truncated), when available.
    pub prompt: Option<String>,
}

/// Fixed-capacity, newest-last log of recent chat events.
pub struct ActivityLog {
    inner: Mutex<VecDeque<ChatEvent>>,
    cap: usize,
}

impl ActivityLog {
    pub fn new(cap: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(cap)),
            cap,
        }
    }

    /// Append an event, evicting the oldest beyond capacity.
    pub fn record(&self, ev: ChatEvent) {
        if let Ok(mut q) = self.inner.lock() {
            if q.len() >= self.cap {
                q.pop_front();
            }
            q.push_back(ev);
        }
    }

    /// Snapshot of the current window, oldest first.
    pub fn recent(&self) -> Vec<ChatEvent> {
        self.inner
            .lock()
            .map(|q| q.iter().cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for ActivityLog {
    /// Keeps the last 100 requests — enough for a lively graph without unbounded growth.
    fn default() -> Self {
        Self::new(100)
    }
}
