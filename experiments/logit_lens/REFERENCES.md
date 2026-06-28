# Logit Lens — References

## Method

### Original Logit Lens
nostalgebraist (2020). *Interpreting GPT: the logit lens.*
LessWrong / AI Alignment Forum.
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

The logit lens projects each layer's hidden state through the final layer norm and
unembedding matrix (lm_head) to obtain an "early prediction" distribution over the
vocabulary. It reveals how a model's implicit prediction evolves across depth without
requiring any additional training.

### Tuned Lens (improved variant)
Belrose, N., Furman, Z., Smith, L., Meek, D., Tenenbholtz, G., & Nanda, N. (2023).
*Eliciting Latent Predictions from Transformers with the Tuned Lens.*
arXiv:2303.08112.
https://arxiv.org/abs/2303.08112

Learns a small affine probe per layer to better align intermediate representations
with the final layer's distribution, improving on the raw logit lens.

---

## Metric: Layer-wise Top-1 Accuracy

### LLM Microscope
Yiran Zhao et al. (2024).
*LLM Microscope: What Model Internals Reveal About Answer Correctness and Context Utilization.*
arXiv:2510.04013.
https://arxiv.org/pdf/2510.04013

At each layer l, multiplies the last-token hidden state by the LM head and applies
softmax to obtain a vocabulary distribution. Records **whether the top-1 token matches
the correct answer's first token** as the primary interpretability signal. This binary
accuracy metric is used instead of raw probability because intermediate hidden states
have different scales than the final layer, making raw softmax values unreliable.

### Do Llamas Work in English?
Wendler, C., Veselovsky, V., Monea, G., & West, R. (2024).
*Do Llamas Work in English? On the Latent Language of Multilingual Transformers.*
ACL 2024. arXiv:2402.10588.
https://arxiv.org/abs/2402.10588

Uses layer-wise top-1 logit lens accuracy to show that multilingual LLMs process
concepts in a latent "pivot language" (often English) in middle layers, regardless of
the input/output language. Directly motivates why intermediate layers in our
experiments show non-English tokens (Korean, Chinese) even on English agricultural
prompts — this is normal for multilingual base models.

---

## Implementation Notes

- **Softcapping**: Gemma-4 applies `tanh(logits / 30) * 30` before softmax
  (`final_logit_softcapping = 30.0` in `text_config`). This must be replicated in
  the logit lens projection or the final layer's distribution will not match the
  model's actual generation output.
- **Generated-position hidden states**: We extract hidden states at the first
  *generated* token position (via `model.generate(max_new_tokens=1,
  output_hidden_states=True)`) rather than an input position. This avoids ambiguity
  about which input position is the "prediction point" in a multimodal model.
- **Prompt**: Identical to the vision assessment (`assessment_classification` +
  `visual_classification_disease` from `src/config/prompts.yaml`) with
  `enable_thinking=False` to match the evaluation setup.
