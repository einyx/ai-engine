use regex::Regex;

/// Replace `${VAR}` occurrences in `src` with the process env var value.
/// Returns an error listing all missing variables.
pub fn env_substitute(src: &str) -> anyhow::Result<String> {
    let re = Regex::new(r"\$\{([A-Z_][A-Z0-9_]*)\}").expect("valid env-var regex");
    let mut missing: Vec<String> = Vec::new();
    let out = re.replace_all(src, |caps: &regex::Captures| {
        match std::env::var(&caps[1]) {
            Ok(v) => v,
            Err(_) => {
                missing.push(caps[1].to_string());
                String::new()
            }
        }
    });
    if !missing.is_empty() {
        anyhow::bail!("missing env vars: {}", missing.join(", "));
    }
    Ok(out.into_owned())
}
