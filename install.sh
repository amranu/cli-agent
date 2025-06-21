#!/bin/bash
# Installation script for CLI Agent

set -e

echo "🚀 Installing CLI Agent..."

# Check if Python 3.8+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python $PYTHON_VERSION detected"
else
    echo "❌ Python $REQUIRED_VERSION or later is required. You have $PYTHON_VERSION"
    exit 1
fi

# Install the package
echo "📦 Installing CLI Agent package..."
pip install -e .

# Check if installation was successful
if command -v agent &> /dev/null; then
    echo "✅ CLI Agent installed successfully!"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Initialize configuration: agent init"
    echo "2. Edit .env file with your API keys"
    echo "3. Start chatting: agent chat"
    echo ""
    echo "📚 For more information, see: README.md"
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi