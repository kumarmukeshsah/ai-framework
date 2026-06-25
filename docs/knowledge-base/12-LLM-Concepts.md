# LLM Concepts Deep Dive

## Overview

Understanding how Large Language Models work is essential for effectively using this framework. This document covers key LLM concepts that directly impact how you configure and interact with providers.

## Core Concepts

### Tokenization

Tokenization is the process of converting text into tokens — the basic units LLMs process.

```python
# The framework handles this via count_tokens()
tokens = await provider.count_tokens("Hello, world!")
# Output: 2 (varies by tokenizer)
```

**Key facts:**
- **1 token ≈ 0.75 words** for English text
- **Different models use different tokenizers**
  - GPT-4/GPT-3.5: `cl100k_base` (~100K vocab)
  - Claude: SentencePiece (~100K vocab)
  - Gemini: SentencePiece (~256K vocab)
- **Token costs vary**: Input tokens are typically cheaper than output tokens
- **Context window** = max tokens a model can process (e.g., GPT-4: 8K/32K/128K)

**Common token counts:**

| Content Type | Tokens (approx) |
|-------------|-----------------|
| "Hello, world!" | 2-4 |
| 1 page of text | ~250 |
| 1,000 words | ~1,333 |
| 10 pages | ~2,500 |
| Full novel | ~100,000+ |

### Temperature

Temperature controls randomness in model outputs:

```python
# Low temperature (deterministic) — for structured tasks
response = await provider.generate(messages, temperature=0.1)

# Medium temperature — balanced creativity
response = await provider.generate(messages, temperature=0.7)

# High temperature (creative) — for brainstorming
response = await provider.generate(messages, temperature=1.5)
```

**Effects:**
- **0.0 - 0.3**: Deterministic, focused — use for extraction, classification, coding
- **0.4 - 0.7**: Balanced — use for general conversation, analysis
- **0.8 - 1.5**: Creative, diverse — use for brainstorming, creative writing
- **> 1.5**: Highly random, may produce gibberish

```python
# Framework default
settings.llm.temperature = 0.7  # Balanced default
```

### Top-p (Nucleus Sampling)

Alternative to temperature that controls probability mass:

```python
# The framework passes this through the provider
# Low top-p: more focused
# High top-p: more diverse
```

**Relationship:** Temperature × Top-p
- **Low temp + low top-p**: Maximum determinism
- **High temp + high top-p**: Maximum creativity
- **Low temp + high top-p**: Focused but allows some diversity

### Max Tokens

Controls maximum response length:

```python
# Short response
response = await provider.generate(messages, max_tokens=100)

# Long response
response = await provider.generate(messages, max_tokens=4096)
```

**Considerations:**
- Higher = more expensive (completion tokens cost money)
- Lower = faster responses
- Can cause truncation if set too low
- Framework default: 4096

### Stop Sequences

Tells the model when to stop generating:

```python
response = await provider.generate(
    messages,
    stop_sequences=["\n\n", "END", "<|eot|>"]
)
```

**Use cases:**
- Stop at double newline for paragraph completion
- Stop at specific delimiter for structured output
- Prevent infinite generation loops

## Prompting Strategies

### System Prompts

Set the model's behavior and persona:

```python
# Framework's EvaluatorAgent uses this
SYSTEM_PROMPT = """You are an expert technical interviewer evaluating candidates.
Analyze transcripts carefully and provide structured, fair evaluations."""
```

**Best practices:**
- Be specific about the role
- Include output format requirements
- Set behavioral guardrails
- Keep concise but complete

### Few-Shot Prompting

Providing examples in the prompt:

```python
prompt = """
Classify the sentiment of these customer reviews:

Review: "This product exceeded my expectations!"
Sentiment: Positive

Review: "Worst purchase I've ever made."
Sentiment: Negative

Review: "It works as expected, nothing special."
Sentiment: Neutral

Review: "{review_text}"
Sentiment:"""
```

### Chain-of-Thought (CoT)

Encouraging step-by-step reasoning:

```python
prompt = """
Evaluate this candidate. Think step by step:

1. Identify skills mentioned
2. Assess years of experience
3. Determine seniority level
4. Score against rubric
5. Provide recommendation

TRANSCRIPT: {transcript}

Let me work through this step by step:"""
```

### Structured Output

Using the framework's structured generation:

```python
# Framework handles JSON parsing via Pydantic
result = await provider.structured_generate(
    messages,
    response_model=CandidateEvaluation,
)
```

## Embeddings

Text embeddings convert text to vector representations:

```python
# Generate embeddings
response = await provider.embeddings([
    "Python is a programming language",
    "FastAPI is a web framework",
])

# response.embeddings[0] -> [0.012, -0.034, ...]  # 1536-dim vector
```

**Properties:**
- **Semantic similarity**: Similar text → nearby vectors
- **Fixed dimensionality**: Typically 768, 1024, 1536, or 3072
- **Model-specific**: Different models produce different vector spaces

**Common embedding models:**

| Model | Dimensions | Context | Best For |
|-------|-----------|---------|----------|
| text-embedding-3-small | 1536 | 8K | General purpose |
| text-embedding-3-large | 3072 | 8K | High accuracy |
| ada-002 | 1536 | 8K | Legacy |

## Attention & Context

### Self-Attention

The mechanism that allows models to weigh the importance of different tokens:

```
"The cat sat on the mat because it was comfortable."
                                       ↑
                              "it" attends to "cat" (not "mat")
```

### Context Window Limits

```python
# Different providers have different limits
configs = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
    "claude-3-opus": 200000,
    "gemini-1.5": 1000000,
}
```

### Positional Encoding

Models need position information since attention is position-independent:

- **Absolute**: Each position has a unique encoding
- **Relative**: Based on token distances
- **RoPE (Rotary Position Embedding)**: Used by many modern models

## Model Architectures

### Transformer Architecture

```
Input → Token Embedding → Positional Encoding → Multi-Head Attention
  → Feed-Forward → Layer Norm → Output
```

### Decoder-Only (GPT, Claude, Gemini)

```
Input → Masked Self-Attention → Feed-Forward → Output
```

Used by most modern LLMs. Generates tokens left-to-right.

### Mixture of Experts (MoE)

```
Input → Router → Expert 1 → Output (if selected)
               → Expert 3 → Output (if selected)
               → Expert 7 → Output (if selected)
```

- **GPT-4, Mixtral**: Uses MoE for efficiency
- Only activates relevant "experts" for each token
- More parameters ≠ more computation

## Sampling Methods

### Greedy Decoding
Always picks the most likely token. Deterministic but can be repetitive.

### Beam Search
Keeps multiple candidate sequences. Better quality, higher cost.

### Top-k Sampling
Samples from the k most likely tokens. Prevents rare/weird tokens.

### Temperature Sampling
Scales probabilities before sampling. Higher = more diverse.

## Performance Considerations

### Latency Factors

| Factor | Impact |
|--------|--------|
| **Input length** | Linear increase with tokens |
| **Output length** | Linear increase with generated tokens |
| **Model size** | Larger models = slower per token |
| **Batch size** | Batching multiple requests can improve throughput |
| **Quantization** | FP16 vs FP32 impacts speed/quality |

### Cost Optimization

```python
# Use smaller model for simple tasks
settings.llm.model = "gpt-4o-mini"  # Cheaper, faster

# Limit max tokens
settings.llm.max_tokens = 512  # No need for 4096 on simple tasks

# Use rule-based when possible
agent = EvaluatorAgent(use_llm=False)  # Zero cost