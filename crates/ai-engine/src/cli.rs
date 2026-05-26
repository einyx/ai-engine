use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "ai-engine",
    about = "AI gateway: OpenAI / Anthropic / OpenAI-compatible (Ollama, vLLM, LM Studio, OpenRouter) proxy",
    version
)]
pub struct Cli {
    /// Path to the configuration file.
    #[arg(short, long, default_value = "ai-engine.toml")]
    pub config: PathBuf,

    /// Validate the configuration and exit without serving.
    #[arg(long)]
    pub check: bool,

    /// Override the auto-detected node identifier (defaults to hostname).
    /// Used to disambiguate role in cluster mode.
    #[arg(long)]
    pub node_id: Option<String>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Advertise a local Ollama instance on mDNS so ai-engine gateways on the
    /// LAN can auto-discover and register it as a provider. Runs until killed.
    AdvertiseOllama {
        /// Base URL of the local Ollama HTTP API.
        #[arg(long, default_value = "http://localhost:11434")]
        ollama_url: String,
        /// Short host label included in the advertisement (defaults to hostname).
        #[arg(long)]
        label: Option<String>,
    },
}
