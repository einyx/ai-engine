use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "airproxy",
    about = "AI gateway: OpenAI / Anthropic / OpenAI-compatible (Ollama, vLLM, LM Studio, OpenRouter) proxy",
    version
)]
pub struct Cli {
    /// Path to the configuration file.
    #[arg(short, long, default_value = "airproxy.toml")]
    pub config: PathBuf,

    /// Validate the configuration and exit without serving.
    #[arg(long)]
    pub check: bool,
}
