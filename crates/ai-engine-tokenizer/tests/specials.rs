use ai_engine_tokenizer::{ModelFamily, SpecialTokens};

#[test]
fn llama3_specials() {
    let s = SpecialTokens::for_family(ModelFamily::Llama3);
    assert_eq!(s.bos_token_id, 128000);   // Llama-3 <|begin_of_text|>
    assert_eq!(s.eos_token_id, 128001);   // Llama-3 <|end_of_text|>
}

#[test]
fn qwen25_specials() {
    let s = SpecialTokens::for_family(ModelFamily::Qwen25);
    assert_eq!(s.bos_token_id, 151643);   // Qwen 2.5 <|endoftext|>
    assert_eq!(s.eos_token_id, 151645);   // Qwen 2.5 <|im_end|>
}

#[test]
fn mistral_specials() {
    let s = SpecialTokens::for_family(ModelFamily::Mistral);
    assert_eq!(s.bos_token_id, 1);
    assert_eq!(s.eos_token_id, 2);
}

#[test]
fn deepseek_specials() {
    let s = SpecialTokens::for_family(ModelFamily::DeepSeekV2);
    assert_eq!(s.bos_token_id, 100000);
    assert_eq!(s.eos_token_id, 100001);
}
