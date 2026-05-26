//! Serves the React SPA from assets embedded at compile time. The Vite build
//! writes into `assets/`; if that dir only holds `.gitkeep`, requests fall back
//! to a minimal placeholder page so the binary still runs without a frontend
//! build (Node is a build-time-only dependency).

use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "assets/"]
struct Assets;

const PLACEHOLDER: &str =
    "<!doctype html><title>ai-engine</title><h1>ai-engine</h1><p>UI not built. \
Run the frontend build to embed it.</p>";

// Content-hashed build assets (e.g. `assets/index-abc123.js`) are safe to cache
// forever — their URL changes whenever the content does.
const IMMUTABLE: &str = "public, max-age=31536000, immutable";
// `index.html` is NOT hashed and points at the current asset hashes, so it must
// always be revalidated; otherwise a stale index pins the browser to old JS.
const NO_CACHE: &str = "no-cache";

fn serve_path(path: &str) -> Response {
    if path == "index.html" {
        return serve_index();
    }
    match Assets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            (
                [
                    (header::CONTENT_TYPE, mime.as_ref()),
                    (header::CACHE_CONTROL, IMMUTABLE),
                ],
                content.data.into_owned(),
            )
                .into_response()
        }
        None => serve_index(),
    }
}

fn serve_index() -> Response {
    match Assets::get("index.html") {
        Some(content) => (
            [
                (header::CONTENT_TYPE, "text/html"),
                (header::CACHE_CONTROL, NO_CACHE),
            ],
            content.data.into_owned(),
        )
            .into_response(),
        None => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/html"),
                (header::CACHE_CONTROL, NO_CACHE),
            ],
            PLACEHOLDER,
        )
            .into_response(),
    }
}

async fn asset_handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    if path.is_empty() {
        serve_index()
    } else {
        serve_path(path)
    }
}

/// SPA router: serves embedded assets; unknown paths fall back to `index.html`
/// (client-side routing) via the fallback handler.
pub fn static_router() -> Router {
    Router::new().fallback(get(asset_handler))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_served_when_no_index() {
        let r = serve_index();
        assert_eq!(r.status(), StatusCode::OK);
    }
}
