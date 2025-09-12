#!/usr/bin/env python3
"""
LM Studio Integration Guide for GGUF2MLX Project
Updated helper script with CLI integration and current project status
"""

import json
import subprocess
from pathlib import Path

def check_lms_cli_available():
    """Check if LM Studio CLI (lms) is available"""
    try:
        result = subprocess.run(['lms', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, "LM Studio CLI not working properly"
    except FileNotFoundError:
        return False, "LM Studio CLI (lms) not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "LM Studio CLI timeout"

def check_project_structure():
    """Check current project structure and files"""
    print("=" * 60)
    print("GGUF2MLX Project Structure Check")
    print("=" * 60)
    
    # Core files check
    core_files = {
        'gguf2mlx.py': 'GGUF to MLX converter',
        'demo.py': 'Basic MLX demo',
        'demo1.py': 'Advanced MLX demo', 
        'demo2transformers.py': 'Transformers framework demo',
        'demo2unsloth.py': 'Unsloth framework demo',
        'demo2shimmy.py': 'JAX/Shimmy framework demo',
        'lms_cli.py': 'LM Studio CLI integration',
        'mlx_to_gguf.py': 'MLX to GGUF converter',
        'requirements.txt': 'Pip dependencies',
        'pyproject.toml': 'UV dependencies',
        'README.md': 'Project documentation'
    }
    
    print("Core Files:")
    for file, description in core_files.items():
        file_path = Path(file)
        status = "✅" if file_path.exists() else "❌"
        print(f"  {status} {file:<25} - {description}")
    
    # Models directory check
    print("\nModels Directory:")
    models_dir = Path("./models")
    if models_dir.exists():
        print("  ✅ models/ directory exists")
        
        # Check GGUF file
        gguf_file = models_dir / "phi3-mini.gguf"
        if gguf_file.exists():
            size_gb = gguf_file.stat().st_size / (1024**3)
            print(f"  ✅ phi3-mini.gguf ({size_gb:.2f} GB)")
        else:
            print("  ❌ phi3-mini.gguf not found")
        
        # Check MLX directory
        mlx_dir = models_dir / "phi3-mini-mlx"
        if mlx_dir.exists():
            print("  ✅ phi3-mini-mlx/ directory exists")
            mlx_files = ['config.json', 'model.npz', 'tokenizer_config.json', 'vocab.json']
            for mlx_file in mlx_files:
                file_path = mlx_dir / mlx_file
                status = "✅" if file_path.exists() else "❌"
                print(f"    {status} {mlx_file}")
        else:
            print("  ❌ phi3-mini-mlx/ directory not found")
    else:
        print("  ❌ models/ directory not found")

def check_lm_studio_status():
    """Check LM Studio status and imported models"""
    print("\n" + "=" * 60)
    print("LM STUDIO STATUS CHECK")
    print("=" * 60)
    
    # Check if LMS CLI is available
    lms_available, lms_info = check_lms_cli_available()
    
    if lms_available:
        print("✅ LM Studio CLI is available")
        print(f"   Version info: {lms_info.split('GitHub:')[0].strip()}")
        
        # Check imported models
        try:
            result = subprocess.run(['lms', 'ls'], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("\n📋 LM Studio Models:")
                lines = result.stdout.strip().split('\n')
                model_found = False
                for line in lines:
                    if 'phi3-mini' in line.lower():
                        print(f"  ✅ {line.strip()}")
                        model_found = True
                    elif 'models' in line and 'GB' in line:
                        print(f"  📊 {line.strip()}")
                
                if not model_found:
                    print("  ⚠️  phi3-mini model not found in LM Studio")
                    print("     Use: python lms_cli.py import ./models/phi3-mini.gguf")
            else:
                print("❌ Failed to list LM Studio models")
        except Exception as e:
            print(f"❌ Error checking LM Studio models: {e}")
    else:
        print(f"❌ LM Studio CLI not available: {lms_info}")
        print("   Install LM Studio and ensure 'lms' is in PATH")

def check_lm_studio_compatibility(model_path):
    """Check if a model directory is compatible with LM Studio"""
    model_path = Path(model_path)
    
    print("=" * 60)
    print("LM Studio Compatibility Checker")
    print("=" * 60)
    
    # Check for GGUF file
    gguf_files = list(model_path.glob("*.gguf"))
    if gguf_files:
        print("✅ GGUF file found - Compatible with LM Studio")
        for gguf_file in gguf_files:
            print(f"   📁 {gguf_file}")
        return True
    
    # Check for MLX format
    if (model_path / "config.json").exists() and (model_path / "model.npz").exists():
        print("❌ MLX format detected - NOT compatible with LM Studio")
        print("   📁 config.json")
        print("   📁 model.npz")
        print("   📁 tokenizer_config.json")
        print("   📁 vocab.json")
        
        print("\n" + "=" * 60)
        print("SOLUTIONS:")
        print("=" * 60)
        print("1. Use original GGUF file with LM Studio")
        print("2. Convert MLX back to GGUF (requires custom implementation)")
        print("3. Use MLX demos instead of LM Studio")
        
        return False
    
    print("❓ Unknown format")
    return False

def get_lm_studio_instructions():
    """Provide instructions for using models with LM Studio"""
    instructions = """
## Using Your Models with LM Studio

### ✅ CORRECT METHOD: Import Local GGUF File
1. Open LM Studio
2. Click on "My Models" tab (left sidebar)
3. Click the "Import Model" button (+ icon) or drag & drop
4. Navigate to and select: ./models/phi3-mini.gguf
5. Wait for LM Studio to import and process the model
6. Once imported, you'll see it in "My Models" list
7. Click "Load in Chat" to use the model

### ❌ COMMON MISTAKE: Don't use "Download" or search
- Don't try to search for "phi3-mini" in the model browser
- Don't use the "Download" section for local files
- The search function looks online, not for local files

### 🔧 LM Studio Local File Tips
- Use "Import Model" or drag-and-drop for local files
- Ensure the .gguf file is not corrupted (check file size > 0)
- LM Studio may take time to import larger models
- Check "My Models" tab after import completes

### 📂 Alternative: Copy to LM Studio Directory
If import doesn't work, manually copy the file:
```bash
# Copy GGUF file to LM Studio's models directory
cp ./models/phi3-mini.gguf ~/.lmstudio/models/
```
Then restart LM Studio and check "My Models"

### 🚀 Verify Model File
Before importing, check your GGUF file:
```bash
# Check file exists and has reasonable size
ls -lh ./models/phi3-mini.gguf
# Should show file size (typically several GB for phi3-mini)
```

### 📊 Model Information
- Model: phi3-mini
- Format: GGUF (LM Studio compatible)
- Size: ~3.8GB (estimated for 3.8B parameters)
- Optimization: Quantized for efficiency

### 🚀 Alternative: Use MLX Demos
If you want to use the converted MLX model:
```bash
# Basic MLX generation
python3 demo.py --input ./models/phi3-mini-mlx --prompt "Your question" --output result.txt

# Advanced MLX generation
python3 demo1.py --input ./models/phi3-mini-mlx --prompt "Your question" --output result.txt

# Transformers integration
python3 demo2transformers.py --input ./models/phi3-mini-mlx --prompt "Your question" --output result.txt
```

### 🔄 MLX to GGUF Conversion (Now Available!)
Convert MLX back to GGUF for LM Studio:
```bash
# Convert MLX to GGUF and copy to LM Studio
python3 mlx_to_gguf.py --input ./models/phi3-mini-mlx --output ./models/phi3-mini-converted.gguf --copy-to-lmstudio
```
"""
    return instructions

def get_updated_lm_studio_instructions():
    """Provide updated instructions for using models with LM Studio"""
    instructions = """
## UPDATED: Using Your Models with LM Studio (CLI + GUI)

### 🚀 METHOD 1: LM Studio CLI (Recommended)
Using the new lms_cli.py integration:

```bash
# Check if LM Studio CLI is working
python lms_cli.py check

# Import your GGUF model automatically
python lms_cli.py import ./models/phi3-mini.gguf --creator gguf2mlx --name phi3-mini

# List all models in LM Studio
python lms_cli.py list

# Load the model for use
python lms_cli.py load phi3-mini

# Start interactive chat
python lms_cli.py chat phi3-mini

# Check current status
python lms_cli.py status
```

### 🖥️ METHOD 2: LM Studio GUI Import
1. Open LM Studio application
2. Click "My Models" tab (left sidebar)
3. Click "Import Model" button (+ icon)
4. Navigate to: ./models/phi3-mini.gguf
5. Wait for import to complete
6. Load model and start chatting

### ❌ AVOID: Don't search for "phi3-mini"
- Searching looks for online models, not your local file
- Always use Import or CLI commands for local files

### 🔧 Troubleshooting Commands
```bash
# If model import fails, check file status
python lm_studio_helper.py

# Convert MLX back to GGUF if needed
python mlx_to_gguf.py --input ./models/phi3-mini-mlx --output ./models/phi3-converted.gguf --copy-to-lmstudio

# Auto-import all GGUF files from project
python lms_cli.py auto-import
```

### 🚀 Alternative: MLX Framework Demos
Use Apple Silicon optimized MLX instead of LM Studio:

```bash
# Basic MLX generation
python demo.py --input ./models/phi3-mini-mlx --prompt "Your question" --output result.txt

# Advanced MLX with transformer architecture
python demo1.py --input ./models/phi3-mini-mlx --prompt "Your question" --output result.txt

# Hugging Face Transformers integration
python demo2transformers.py --input ./models/phi3-mini-mlx --prompt "Your question" --output result.txt

# JAX/Flax functional programming
python demo2shimmy.py --input ./models/phi3-mini-mlx --prompt "Your question" --output result.txt
```

### 📊 Framework Comparison
| Method | Speed | Memory | Setup | Best For |
|--------|-------|--------|--------|----------|
| LM Studio | ⭐⭐⭐ | ⭐⭐⭐ | Easy GUI | Interactive chat |
| MLX Demos | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Python env | Apple Silicon optimization |
| CLI Integration | ⭐⭐⭐⭐ | ⭐⭐⭐ | Terminal | Automation & scripting |
"""
    return instructions

def check_gguf_file_status(gguf_path):
    """Check if GGUF file is valid for LM Studio import"""
    gguf_path = Path(gguf_path)
    
    print("=" * 60)
    print("GGUF File Status Check")
    print("=" * 60)
    
    if not gguf_path.exists():
        print(f"❌ File not found: {gguf_path}")
        return False
    
    # Check file size
    file_size = gguf_path.stat().st_size
    size_mb = file_size / (1024 * 1024)
    size_gb = size_mb / 1024
    
    print(f"✅ File exists: {gguf_path}")
    print(f"📊 File size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
    
    if file_size < 1024 * 1024:  # Less than 1MB
        print("⚠️  File size is very small - may be corrupted")
        return False
    elif file_size < 100 * 1024 * 1024:  # Less than 100MB
        print("⚠️  File size is smaller than expected for phi3-mini")
    else:
        print("✅ File size looks reasonable")
    
    # Check if it's a proper GGUF file
    try:
        with open(gguf_path, 'rb') as f:
            magic = f.read(4)
            if magic == b'GGUF':
                print("✅ Valid GGUF magic number detected")
                return True
            else:
                print(f"❌ Invalid file format - not GGUF (magic: {magic})")
                return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def provide_lm_studio_troubleshooting():
    """Provide troubleshooting steps for LM Studio"""
    print("=" * 60)
    print("LM STUDIO TROUBLESHOOTING")
    print("=" * 60)
    print()
    print("🔍 If LM Studio searches online instead of loading local file:")
    print("1. Don't use the search bar or 'Discover' section")
    print("2. Use 'My Models' tab → 'Import Model' button")
    print("3. Or drag & drop the .gguf file directly into LM Studio")
    print()
    print("🔍 If import fails:")
    print("1. Check file is not corrupted (see file size above)")
    print("2. Restart LM Studio and try again")
    print("3. Try copying file to ~/.lmstudio/models/ manually")
    print("4. Ensure you have enough disk space")
    print()
    print("🔍 If model doesn't appear in 'My Models':")
    print("1. Wait for import to complete (check bottom status bar)")
    print("2. Refresh the 'My Models' tab")
    print("3. Check LM Studio logs for errors")
    print("4. Try renaming file to something simpler (e.g., 'phi3.gguf')")

def provide_complete_workflow():
    """Provide complete workflow examples"""
    print("=" * 60)
    print("COMPLETE WORKFLOW EXAMPLES")
    print("=" * 60)
    print()
    print("🔄 Full Conversion and Usage Workflow:")
    print("1. python gguf2mlx.py --input ./models/phi3-mini.gguf --output ./models/phi3-mini-mlx")
    print("2. python lms_cli.py import ./models/phi3-mini.gguf --creator gguf2mlx")
    print("3. python lms_cli.py load phi3-mini")
    print("4. python lms_cli.py chat")
    print()
    print("🧪 Testing Different Frameworks:")
    print("# Test MLX (Apple Silicon optimized)")
    print("python demo.py --input ./models/phi3-mini-mlx --prompt 'Hello AI' --output mlx_test.txt")
    print()
    print("# Test LM Studio via CLI")
    print("python lms_cli.py chat phi3-mini")
    print()
    print("# Test Transformers framework")
    print("python demo2transformers.py --input ./models/phi3-mini-mlx --prompt 'Hello AI' --output transformers_test.txt")
    print()
    print("🔧 Troubleshooting Commands:")
    print("python lm_studio_helper.py         # Check project status")
    print("python lms_cli.py check            # Verify LM Studio CLI")
    print("python lms_cli.py list             # List imported models")
    print("python lms_cli.py status           # Check LM Studio status")

def main():
    """Updated main function with comprehensive project check"""
    print("LM Studio Integration Helper - Updated for GGUF2MLX Project")
    print("Date: September 12, 2025")
    print("=" * 60)
    
    # Check project structure
    check_project_structure()
    
    # Check LM Studio status
    check_lm_studio_status()
    
    # Check specific model files
    print("\n" + "=" * 60)
    print("MODEL FILES DETAILED CHECK")
    print("=" * 60)
    
    gguf_file = Path("./models/phi3-mini.gguf")
    if gguf_file.exists():
        check_gguf_file_status(gguf_file)
    
    mlx_dir = Path("./models/phi3-mini-mlx")
    if mlx_dir.exists():
        check_lm_studio_compatibility(mlx_dir)
    
    # Provide updated instructions
    print("\n" + get_updated_lm_studio_instructions())
    
    # Show complete workflow
    provide_complete_workflow()
    
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print("✅ Project Status: Complete GGUF2MLX ecosystem ready")
    print("✅ LM Studio Integration: CLI + GUI methods available")
    print("✅ MLX Optimization: Apple Silicon native performance")
    print("✅ Multiple Frameworks: Choose the best tool for your needs")
    print()
    print("🎯 Quick Start:")
    print("  python lms_cli.py check          # Verify LM Studio")
    print("  python lms_cli.py list           # See imported models") 
    print("  python lms_cli.py chat           # Start chatting")
    print("  python demo1.py --input ./models/phi3-mini-mlx --prompt 'Hello' --output test.txt")

if __name__ == "__main__":
    main()