#!/usr/bin/env python3
"""
LM Studio CLI Integration for GGUF2MLX Project
Complete shell interface for LM Studio operations
"""

import subprocess
import json
import sys
from pathlib import Path
import argparse
import time

def check_lms_available():
    """Check if LM Studio CLI (lms) is available"""
    try:
        result = subprocess.run(['lms', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ LM Studio CLI (lms) is available")
            print(f"   {result.stdout.strip()}")
            return True
        else:
            print("❌ LM Studio CLI not working properly")
            return False
    except FileNotFoundError:
        print("❌ LM Studio CLI (lms) not found")
        print("   Please install LM Studio and ensure 'lms' is in your PATH")
        return False
    except subprocess.TimeoutExpired:
        print("❌ LM Studio CLI timeout")
        return False

def import_model_to_lmstudio(model_path, creator="gguf2mlx", model_name=None, copy_mode=True):
    """Import a model file into LM Studio using lms CLI"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not model_name:
        model_name = model_path.stem
    
    print("=" * 60)
    print("LM Studio Model Import")
    print("=" * 60)
    print(f"Model file: {model_path}")
    print(f"Creator: {creator}")
    print(f"Model name: {model_name}")
    print(f"Import mode: {'Copy' if copy_mode else 'Move'}")
    
    # Build command
    cmd = ['lms', 'import', str(model_path)]
    if copy_mode:
        cmd.append('--copy')
    
    try:
        print("\n🚀 Starting import process...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run import command
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Provide interactive responses
        # Answer "Yes" to continue
        # Choose "Interactive import"
        # Provide creator and model name
        responses = [
            "Yes\n",  # Do you wish to continue?
            "Interactive import (Recommended for custom models)\n",  # Choose categorization
            f"{creator}\n",  # Creator name
            f"{model_name}\n"  # Model name
        ]
        
        # Send responses
        for response in responses:
            process.stdin.write(response)
            process.stdin.flush()
            time.sleep(0.5)  # Small delay between responses
        
        # Wait for completion
        stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
        
        if process.returncode == 0:
            print("✅ Model imported successfully!")
            print(f"Output: {stdout}")
            return True
        else:
            print(f"❌ Import failed with return code {process.returncode}")
            print(f"Error: {stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Import timed out (5 minutes)")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def list_lmstudio_models():
    """List all models in LM Studio"""
    try:
        result = subprocess.run(['lms', 'ls'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("=" * 60)
            print("LM Studio Models")
            print("=" * 60)
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to list models: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

def load_model_in_lmstudio(model_identifier):
    """Load a model in LM Studio"""
    try:
        print(f"🚀 Loading model: {model_identifier}")
        result = subprocess.run(['lms', 'load', model_identifier], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Model loaded successfully!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to load model: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def get_lmstudio_status():
    """Get LM Studio status"""
    try:
        result = subprocess.run(['lms', 'status'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("=" * 60)
            print("LM Studio Status")
            print("=" * 60)
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to get status: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return False

def list_loaded_models():
    """List currently loaded models"""
    try:
        result = subprocess.run(['lms', 'ps'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("=" * 60)
            print("Currently Loaded Models")
            print("=" * 60)
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to list loaded models: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error listing loaded models: {e}")
        return False

def start_chat_session(model_identifier=None):
    """Start an interactive chat session"""
    try:
        cmd = ['lms', 'chat']
        if model_identifier:
            cmd.append(model_identifier)
        
        print(f"🚀 Starting chat session...")
        if model_identifier:
            print(f"   Model: {model_identifier}")
        
        # Run interactive chat
        subprocess.run(cmd)
        return True
    except Exception as e:
        print(f"❌ Error starting chat: {e}")
        return False

def auto_import_project_models():
    """Automatically import all GGUF models from the project"""
    models_dir = Path("./models")
    gguf_files = list(models_dir.glob("*.gguf"))
    
    if not gguf_files:
        print("❌ No GGUF files found in ./models/")
        return False
    
    print(f"📁 Found {len(gguf_files)} GGUF files to import:")
    for gguf_file in gguf_files:
        print(f"   - {gguf_file}")
    
    success_count = 0
    for gguf_file in gguf_files:
        model_name = gguf_file.stem
        print(f"\n📥 Importing {gguf_file.name}...")
        
        if import_model_to_lmstudio(gguf_file, "gguf2mlx", model_name):
            success_count += 1
        else:
            print(f"❌ Failed to import {gguf_file.name}")
    
    print(f"\n✅ Successfully imported {success_count}/{len(gguf_files)} models")
    return success_count > 0

def show_lms_help():
    """Show LM Studio CLI help"""
    try:
        result = subprocess.run(['lms', '--help'], 
                              capture_output=True, text=True, timeout=30)
        print("=" * 60)
        print("LM Studio CLI Help")
        print("=" * 60)
        print(result.stdout)
        return True
    except Exception as e:
        print(f"❌ Error getting help: {e}")
        return False

def main():
    """Main CLI interface for LM Studio integration"""
    parser = argparse.ArgumentParser(description="LM Studio CLI Integration for GGUF2MLX")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    subparsers.add_parser('check', help='Check if LM Studio CLI is available')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import a model file')
    import_parser.add_argument('file', help='Path to model file')
    import_parser.add_argument('--creator', default='gguf2mlx', help='Model creator name')
    import_parser.add_argument('--name', help='Model name (default: filename)')
    import_parser.add_argument('--move', action='store_true', help='Move file instead of copy')
    
    # Auto-import command
    subparsers.add_parser('auto-import', help='Import all GGUF files from ./models/')
    
    # List commands
    subparsers.add_parser('list', help='List all downloaded models')
    subparsers.add_parser('loaded', help='List currently loaded models')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load a model')
    load_parser.add_argument('model', help='Model identifier to load')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start chat session')
    chat_parser.add_argument('model', nargs='?', help='Model identifier (optional)')
    
    # Status command
    subparsers.add_parser('status', help='Show LM Studio status')
    
    # Help command
    subparsers.add_parser('help', help='Show LM Studio CLI help')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Check LMS availability for all commands except 'check'
    if args.command != 'check':
        if not check_lms_available():
            print("\n💡 Install LM Studio and ensure 'lms' CLI is available")
            return 1
    
    # Execute commands
    if args.command == 'check':
        return 0 if check_lms_available() else 1
    
    elif args.command == 'import':
        copy_mode = not args.move
        model_name = args.name or Path(args.file).stem
        return 0 if import_model_to_lmstudio(args.file, args.creator, model_name, copy_mode) else 1
    
    elif args.command == 'auto-import':
        return 0 if auto_import_project_models() else 1
    
    elif args.command == 'list':
        return 0 if list_lmstudio_models() else 1
    
    elif args.command == 'loaded':
        return 0 if list_loaded_models() else 1
    
    elif args.command == 'load':
        return 0 if load_model_in_lmstudio(args.model) else 1
    
    elif args.command == 'chat':
        return 0 if start_chat_session(args.model) else 1
    
    elif args.command == 'status':
        return 0 if get_lmstudio_status() else 1
    
    elif args.command == 'help':
        return 0 if show_lms_help() else 1
    
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    exit(main())

# Usage examples:
"""
# Check if LM Studio CLI is available
python lms_cli.py check

# Import your phi3-mini model
python lms_cli.py import ./models/phi3-mini.gguf --creator barrontang --name phi3-mini

# Auto-import all GGUF files from ./models/
python lms_cli.py auto-import

# List all models in LM Studio
python lms_cli.py list

# Load a specific model
python lms_cli.py load barrontang/phi3-mini

# Start chat with loaded model
python lms_cli.py chat

# Check LM Studio status
python lms_cli.py status

# Show LM Studio CLI help
python lms_cli.py help
"""