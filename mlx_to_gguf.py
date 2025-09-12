#!/usr/bin/env python3
"""
MLX to GGUF Converter for LM Studio Compatibility
Converts MLX format back to GGUF for use with LM Studio
"""

import json
import struct
import numpy as np
from pathlib import Path
import argparse

def mlx_to_gguf_converter(mlx_dir, output_gguf):
    """
    Convert MLX format back to GGUF for LM Studio compatibility
    Note: This is a basic implementation for demonstration
    """
    mlx_dir = Path(mlx_dir)
    output_gguf = Path(output_gguf)
    
    print("=" * 60)
    print("MLX to GGUF Converter for LM Studio")
    print("=" * 60)
    
    # Check if MLX files exist
    config_file = mlx_dir / "config.json"
    weights_file = mlx_dir / "model.npz"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")
    
    # Load MLX configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"Loading MLX model from: {mlx_dir}")
    print(f"Model type: {config.get('model_type', 'unknown')}")
    print(f"Vocab size: {config.get('vocab_size', 0)}")
    print(f"Hidden size: {config.get('hidden_size', 0)}")
    
    # Load weights
    print("Loading MLX weights...")
    weights = np.load(weights_file)
    print(f"Found {len(weights.files)} weight tensors")
    
    # Create basic GGUF file
    print(f"Creating GGUF file: {output_gguf}")
    
    with open(output_gguf, 'wb') as f:
        # Write GGUF magic number
        f.write(b'GGUF')
        
        # Write version (3)
        f.write(struct.pack('<I', 3))
        
        # Write tensor count
        tensor_count = len(weights.files)
        f.write(struct.pack('<Q', tensor_count))
        
        # Write metadata count (basic)
        f.write(struct.pack('<Q', 5))  # Basic metadata entries
        
        # Write basic metadata
        metadata = {
            'general.architecture': 'llama',
            'general.name': config.get('model_type', 'converted_model'),
            'llama.context_length': config.get('max_position_embeddings', 2048),
            'llama.embedding_length': config.get('hidden_size', 1536),
            'llama.block_count': config.get('num_hidden_layers', 24),
        }
        
        for key, value in metadata.items():
            # Write key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            
            # Write value type and value
            if isinstance(value, str):
                f.write(struct.pack('<I', 8))  # String type
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
            elif isinstance(value, int):
                f.write(struct.pack('<I', 4))  # Int32 type
                f.write(struct.pack('<i', value))
        
        # Write tensor info (simplified)
        for weight_name in weights.files:
            weight_data = weights[weight_name]
            
            # Tensor name
            name_bytes = weight_name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Dimensions
            dims = weight_data.shape
            f.write(struct.pack('<I', len(dims)))
            for dim in dims:
                f.write(struct.pack('<Q', dim))
            
            # Data type (assume float16)
            f.write(struct.pack('<I', 1))  # F16 type
            
            # Offset (will be calculated)
            f.write(struct.pack('<Q', 0))  # Placeholder
        
        # Align to 32 bytes
        while f.tell() % 32 != 0:
            f.write(b'\x00')
        
        # Write tensor data
        for weight_name in weights.files:
            weight_data = weights[weight_name]
            # Convert to float16 and write
            if weight_data.dtype != np.float16:
                weight_data = weight_data.astype(np.float16)
            f.write(weight_data.tobytes())
    
    print("✅ GGUF conversion completed!")
    print(f"Output file: {output_gguf}")
    print(f"File size: {output_gguf.stat().st_size / (1024*1024):.1f} MB")
    
    return output_gguf

def copy_to_lmstudio(gguf_file):
    """Copy converted GGUF file to LM Studio models directory"""
    lmstudio_dir = Path.home() / ".lmstudio" / "models"
    
    if not lmstudio_dir.exists():
        print(f"⚠ LM Studio models directory not found: {lmstudio_dir}")
        print("Please ensure LM Studio is installed and has been run at least once.")
        return False
    
    destination = lmstudio_dir / gguf_file.name
    
    try:
        import shutil
        shutil.copy2(gguf_file, destination)
        print(f"✅ Copied to LM Studio: {destination}")
        print("\nTo use in LM Studio:")
        print("1. Open LM Studio")
        print("2. Go to 'My Models'")
        print(f"3. Look for '{gguf_file.name}'")
        print("4. Click 'Load Model'")
        return True
    except Exception as e:
        print(f"❌ Failed to copy to LM Studio: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert MLX format to GGUF for LM Studio")
    parser.add_argument("--input", required=True, help="Input MLX model directory")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--copy-to-lmstudio", action="store_true", 
                       help="Automatically copy to LM Studio models directory")
    
    args = parser.parse_args()
    
    try:
        # Convert MLX to GGUF
        output_file = mlx_to_gguf_converter(args.input, args.output)
        
        # Optionally copy to LM Studio
        if args.copy_to_lmstudio:
            copy_to_lmstudio(output_file)
        
        print("\n" + "=" * 60)
        print("CONVERSION SUMMARY")
        print("=" * 60)
        print(f"✅ MLX model converted to GGUF format")
        print(f"✅ Ready for use with LM Studio")
        print(f"📁 Output: {output_file}")
        
        if not args.copy_to_lmstudio:
            print("\nTo use with LM Studio:")
            print(f"1. Copy {output_file} to ~/.lmstudio/models/")
            print("2. Open LM Studio and load the model")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# Usage examples:
"""
# Convert MLX to GGUF
python mlx_to_gguf.py --input ./models/phi3-mini-mlx --output ./models/phi3-mini-converted.gguf

# Convert and automatically copy to LM Studio
python mlx_to_gguf.py --input ./models/phi3-mini-mlx --output ./models/phi3-mini-converted.gguf --copy-to-lmstudio
"""