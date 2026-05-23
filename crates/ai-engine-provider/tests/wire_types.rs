use ai_engine_provider::anthropic::{MessageContent, MessagesRequest};
use ai_engine_provider::openai::{ChatContent, ChatRequest, EmbeddingsInput, EmbeddingsRequest};

#[test]
fn openai_chat_request_passthrough_preserves_unknown_fields() {
    let raw = r#"{
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
        "stream": true,
        "made_up_future_field": [1, 2, 3]
    }"#;
    let req: ChatRequest = serde_json::from_str(raw).unwrap();
    assert_eq!(req.model, "gpt-4o");
    assert_eq!(req.stream, Some(true));
    assert_eq!(req.temperature, Some(0.5));
    let back = serde_json::to_value(&req).unwrap();
    assert!(back.get("made_up_future_field").is_some(), "extras preserved");
}

#[test]
fn openai_chat_content_string_form() {
    let raw = r#"{"role": "user", "content": "hello"}"#;
    let m: ai_engine_provider::openai::ChatMessage = serde_json::from_str(raw).unwrap();
    assert!(matches!(m.content, ChatContent::Text(ref s) if s == "hello"));
}

#[test]
fn openai_chat_content_parts_form() {
    let raw = r#"{"role": "user", "content": [{"type": "text", "text": "hi"}]}"#;
    let m: ai_engine_provider::openai::ChatMessage = serde_json::from_str(raw).unwrap();
    assert!(matches!(m.content, ChatContent::Parts(_)));
}

#[test]
fn openai_embeddings_input_single_or_many() {
    let single = r#"{"model": "text-embedding-3-small", "input": "foo"}"#;
    let req: EmbeddingsRequest = serde_json::from_str(single).unwrap();
    assert!(matches!(req.input, EmbeddingsInput::Single(ref s) if s == "foo"));

    let many = r#"{"model": "text-embedding-3-small", "input": ["a", "b"]}"#;
    let req: EmbeddingsRequest = serde_json::from_str(many).unwrap();
    assert!(matches!(req.input, EmbeddingsInput::Many(ref v) if v.len() == 2));
}

#[test]
fn anthropic_messages_request_preserves_unknown_fields() {
    let raw = r#"{
        "model": "claude-3-5-sonnet-20240620",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
        "system": "be helpful",
        "future_field": {"nested": true}
    }"#;
    let req: MessagesRequest = serde_json::from_str(raw).unwrap();
    assert_eq!(req.max_tokens, 1024);
    let back = serde_json::to_value(&req).unwrap();
    assert!(back.get("future_field").is_some());
}

#[test]
fn anthropic_message_content_string_and_blocks() {
    let raw_str = r#"{"role": "user", "content": "hi"}"#;
    let m: ai_engine_provider::anthropic::Message = serde_json::from_str(raw_str).unwrap();
    assert!(matches!(m.content, MessageContent::Text(ref s) if s == "hi"));

    let raw_blocks = r#"{"role": "user", "content": [{"type": "text", "text": "hi"}]}"#;
    let m: ai_engine_provider::anthropic::Message = serde_json::from_str(raw_blocks).unwrap();
    assert!(matches!(m.content, MessageContent::Blocks(_)));
}
