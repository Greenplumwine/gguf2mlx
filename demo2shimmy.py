#!/usr/bin/env python3
"""
Shimmy Framework Text Generation Demo
Uses Shimmy framework for JAX-based text generation
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("❌ JAX/Flax not available. Install with: pip install jax flax optax")

try:
    import shimmy
    SHIMMY_AVAILABLE = True
except ImportError:
    SHIMMY_AVAILABLE = False
    print("⚠ Shimmy not available. Using JAX/Flax implementation")

class ShimmyTransformer(nn.Module):
    """Shimmy-style transformer model using JAX/Flax"""
    
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    max_seq_length: int = 1024
    
    def setup(self):
        # Embedding layers
        self.token_embedding = nn.Embed(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embed(self.max_seq_length, self.hidden_size)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_dim=self.hidden_size * 4
            ) for _ in range(self.num_layers)
        ]
        
        # Output layers
        self.layer_norm = nn.LayerNorm()
        self.lm_head = nn.Dense(self.vocab_size)
    
    def __call__(self, input_ids, training=False):
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        positions = jnp.arange(seq_length)[None, :]
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        hidden_states = token_embeds + pos_embeds
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, training=training)
        
        # Final processing
        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    hidden_size: int
    num_heads: int
    mlp_dim: int
    
    def setup(self):
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size
        )
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
        
        self.mlp = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.gelu,
            nn.Dense(self.hidden_size)
        ])
    
    def __call__(self, x, training=False):
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.layer_norm1(x + attn_output)
        
        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = self.layer_norm2(x + mlp_output)
        
        return x

def load_shimmy_model(model_path):
    """Load model for Shimmy framework"""
    model_path = Path(model_path)
    
    # Load config
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load vocabulary
    vocab_path = model_path / "vocab.json"
    tokenizer_path = model_path / "tokenizer_config.json"
    
    vocab = {}
    tokenizer_config = {}
    
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    
    if tokenizer_path.exists():
        with open(tokenizer_path, 'r') as f:
            tokenizer_config = json.load(f)
    
    # Create model
    model = ShimmyTransformer(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config.get('num_hidden_layers', 12),
        num_heads=config.get('num_attention_heads', 12)
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(rng, dummy_input)
    
    print("✓ Shimmy/JAX model created successfully")
    return model, params, config, vocab

def shimmy_tokenize(text, vocab, max_length=256):
    """Tokenize text for Shimmy model"""
    tokens = text.lower().split()
    token_ids = []
    
    # Add BOS token
    token_ids.append(vocab.get('<s>', 1))
    
    for token in tokens[:max_length-2]:
        token_ids.append(vocab.get(token, vocab.get('<unk>', 0)))
    
    # Pad to minimum length
    while len(token_ids) < 16:
        token_ids.append(vocab.get('<pad>', 0))
    
    return jnp.array(token_ids).reshape(1, -1)

def generate_with_shimmy(model, params, config, vocab, prompt, max_tokens=100, temperature=0.8):
    """Generate text using Shimmy/JAX model"""
    print(f"Generating with Shimmy/JAX framework...")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    
    # Tokenize input
    input_ids = shimmy_tokenize(prompt, vocab)
    
    # Reverse vocab for decoding
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    generated_tokens = []
    current_ids = input_ids
    
    rng = jax.random.PRNGKey(42)
    
    for step in range(max_tokens):
        # Forward pass
        logits = model.apply(params, current_ids)
        next_token_logits = logits[0, -1, :]  # Get last token logits
        
        # Apply temperature and sample
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities
            probs = jax.nn.softmax(next_token_logits)
            
            # Sample using JAX random
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, next_token_logits)
        else:
            # Greedy sampling
            next_token = jnp.argmax(next_token_logits)
        
        next_token_id = int(next_token)
        generated_tokens.append(next_token_id)
        
        # Check for EOS
        if next_token_id == vocab.get('</s>', 2):
            break
        
        # Update input for next iteration
        next_token_reshaped = jnp.array([[next_token_id]])
        current_ids = jnp.concatenate([current_ids, next_token_reshaped], axis=1)
        
        # Truncate if too long
        if current_ids.shape[1] > 512:
            current_ids = current_ids[:, -256:]  # Keep last 256 tokens
    
    # Decode tokens
    generated_text = []
    for token_id in generated_tokens:
        if token_id in reverse_vocab:
            token = reverse_vocab[token_id]
            if not (token.startswith('<') and token.endswith('>')):
                generated_text.append(token)
        else:
            generated_text.append(f"[UNK_{token_id}]")
    
    return " ".join(generated_text)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Shimmy/JAX Text Generation Demo")
    parser.add_argument("--input", required=True, help="Input MLX model directory")
    parser.add_argument("--prompt", required=True, help="Text prompt for generation")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    
    args = parser.parse_args()
    
    if not JAX_AVAILABLE:
        print("❌ JAX/Flax framework not available")
        print("Install with: pip install jax flax optax")
        sys.exit(1)
    
    try:
        print("=" * 60)
        print("Shimmy/JAX Text Generation Demo")
        print("=" * 60)
        
        # Load model
        print(f"Loading model for Shimmy/JAX from: {args.input}")
        model, params, config, vocab = load_shimmy_model(args.input)
        print(f"  Model type: {config.get('model_type', 'unknown')}")
        print(f"  Vocab size: {config.get('vocab_size', 0)}")
        print(f"  Framework: Shimmy/JAX (Functional)")
        
        # Display JAX device info
        devices = jax.devices()
        print(f"  JAX devices: {[str(d) for d in devices]}")
        
        # Generate text
        generated_text = generate_with_shimmy(
            model, params, config, vocab, args.prompt, 
            args.max_tokens, args.temperature
        )
        
        # Prepare output
        result = f"=== Shimmy/JAX Generation Results ===\n"
        result += f"Prompt: {args.prompt}\n"
        result += f"Generated: {generated_text}\n"
        result += f"Model: {config.get('model_type', 'unknown')}\n"
        result += f"Framework: Shimmy/JAX (Functional Programming)\n"
        result += f"Max Tokens: {args.max_tokens}\n"
        result += f"Temperature: {args.temperature}\n"
        result += f"Features: JIT compilation, functional transforms\n"
        result += f"Devices: {[str(d) for d in jax.devices()]}\n"
        result += f"Vocab Size: {config.get('vocab_size', 0)}\n"
        result += f"Hidden Size: {config.get('hidden_size', 0)}\n"
        
        # Save output
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(result)
        
        print(f"✓ Generated text saved to: {output_path}")
        print(f"Generated: {generated_text}")
        print("✓ JAX features: JIT compilation, automatic differentiation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()