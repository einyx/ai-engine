//! Render a model's GGUF-embedded Jinja chat template (HF `tokenizer.chat_template`)
//! into a prompt string, using minijinja with HF-compatibility shims.

use minijinja::{context, Environment, Error as JinjaError, ErrorKind};
use std::collections::HashMap;

/// A chat message for templating (role + content).
#[derive(Debug, Clone)]
pub struct TemplateMessage {
    pub role: String,
    pub content: String,
}

/// Render `messages` through the given Jinja `chat_template` source.
/// `bos_token`/`eos_token` are passed as template variables (many HF templates
/// reference them). `add_generation_prompt=true` makes the output end at the
/// assistant turn so the model continues as the assistant.
pub fn render_chat_template(
    chat_template: &str,
    messages: &[TemplateMessage],
    bos_token: &str,
    eos_token: &str,
) -> anyhow::Result<String> {
    let mut env = Environment::new();
    // pycompat: enables Python str methods (.strip(), .split(), etc.) that HF
    // templates use.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // HF templates call raise_exception(msg) for validation (e.g. role ordering).
    env.add_function("raise_exception", |msg: String| -> Result<String, JinjaError> {
        Err(JinjaError::new(ErrorKind::InvalidOperation, msg))
    });
    // Some templates call strftime_now(fmt); provide a stub returning empty.
    env.add_function("strftime_now", |_fmt: String| -> Result<String, JinjaError> {
        Ok(String::new())
    });
    env.add_template("chat", chat_template)
        .map_err(|e| anyhow::anyhow!("chat_template parse: {e}"))?;
    let tmpl = env.get_template("chat").unwrap();

    // messages as a list of maps (templates index msg['role'] / msg['content']).
    let msgs: Vec<HashMap<&str, &str>> = messages
        .iter()
        .map(|m| {
            let mut h = HashMap::new();
            h.insert("role", m.role.as_str());
            h.insert("content", m.content.as_str());
            h
        })
        .collect();

    let rendered = tmpl
        .render(context! {
            messages => msgs,
            add_generation_prompt => true,
            bos_token => bos_token,
            eos_token => eos_token,
        })
        .map_err(|e| anyhow::anyhow!("chat_template render: {e}"))?;
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(s: &str) -> Vec<TemplateMessage> {
        vec![TemplateMessage { role: "user".into(), content: s.into() }]
    }

    #[test]
    fn renders_chatml_template() {
        // Minimal ChatML template (Qwen-style).
        let tmpl = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let out = render_chat_template(tmpl, &user_msg("Hello"), "", "<|im_end|>").unwrap();
        assert_eq!(out, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn renders_with_bos_token_variable() {
        let tmpl = "{{ bos_token }}{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}";
        let out = render_chat_template(tmpl, &user_msg("Hi"), "<s>", "</s>").unwrap();
        assert_eq!(out, "<s>user: Hi\n");
    }

    #[test]
    fn pycompat_strip_method_works() {
        // .strip() is a Python str method enabled by pycompat.
        let tmpl = "{{ messages[0]['content'].strip() }}";
        let out = render_chat_template(tmpl, &user_msg("  spaced  "), "", "").unwrap();
        assert_eq!(out, "spaced");
    }

    #[test]
    fn raise_exception_surfaces_as_error() {
        let tmpl = "{{ raise_exception('bad template') }}";
        let err = render_chat_template(tmpl, &user_msg("x"), "", "").unwrap_err();
        assert!(err.to_string().contains("bad template"));
    }
}
