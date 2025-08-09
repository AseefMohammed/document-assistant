#!/usr/bin/env python3
"""
Setup script for the Fintech Document Assistant
Handles initial configuration and dependency checking
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/documents",
        "data/vector_store", 
        "logs"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_ollama():
    """Check if Ollama is available"""
    try:
        subprocess.run(["ollama", "--version"], 
                      check=True, capture_output=True)
        print("✅ Ollama is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Ollama not found")
        print("To install Ollama:")
        if platform.system() == "Darwin":  # macOS
            print("  brew install ollama")
        elif platform.system() == "Linux":
            print("  curl -fsSL https://ollama.ai/install.sh | sh")
        else:
            print("  Visit https://ollama.ai for installation instructions")
        return False


def setup_environment():
    """Setup environment file"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        print("📝 Creating .env file from template...")
        env_content = env_example.read_text()
        env_file.write_text(env_content)
        print("✅ Environment file created")
        print("📝 Please review and adjust settings in .env file")
    elif env_file.exists():
        print("✅ Environment file already exists")
    else:
        print("⚠️  No environment template found")


def download_models():
    """Download required models"""
    models_to_download = [
        "mistral:7b",
        # "phi3:mini"  # Uncomment if you want to use Phi-3
    ]
    
    if not check_ollama():
        print("⚠️  Skipping model download - Ollama not available")
        return
    
    print("🤖 Downloading language models...")
    for model in models_to_download:
        print(f"📥 Downloading {model}...")
        try:
            subprocess.run([
                "ollama", "pull", model
            ], check=True)
            print(f"✅ Downloaded {model}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to download {model}")


def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        from main import DocumentAssistant
        print("✅ Main module imports successfully")
        
        # Test basic initialization (will fail if dependencies missing)
        try:
            assistant = DocumentAssistant()
            status = assistant.get_system_status()
            print("✅ Document Assistant initializes successfully")
            
            if status.get("llm_status", {}).get("ollama_available"):
                print("✅ Ollama connection successful")
            else:
                print("⚠️  Ollama not available - responses will be limited")
                
        except Exception as e:
            print(f"⚠️  Assistant initialization issue: {e}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True


def main():
    """Main setup function"""
    print("🚀 Fintech Document Assistant Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed due to dependency installation errors")
        sys.exit(1)
    
    # Check Ollama
    check_ollama()
    
    # Download models (optional)
    download_models()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and adjust settings in .env file")
        print("2. Add documents: python main.py add-dir ./data/documents")
        print("3. Start querying: python main.py query 'What is AML compliance?'")
        print("\nFor Docker deployment:")
        print("1. docker-compose up -d")
        print("2. Wait for services to start")
        print("3. Access via the API or command line")
    else:
        print("\n⚠️  Setup completed with issues")
        print("Please check the error messages above and ensure all dependencies are properly installed")


if __name__ == "__main__":
    main()
