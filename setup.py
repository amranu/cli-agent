#!/usr/bin/env python3
"""Setup script for CLI Agent package."""

import os

from setuptools import find_packages, setup


# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "CLI Agent - MCP-enabled AI assistant with tool integration"


# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="mcp-cli-agent",
    version="1.1.0",
    description="MCP-enabled AI assistant with tool integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="CLI Agent Team",
    author_email="",
    url="https://github.com/amranu/cli-agent",
    packages=find_packages(),
    py_modules=["agent", "config", "mcp_deepseek_host", "mcp_gemini_host"],
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "agent=agent:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="ai assistant mcp tools cli chatbot deepseek gemini",
    project_urls={
        "Bug Reports": "https://github.com/amranu/cli-agent/issues",
        "Source": "https://github.com/amranu/cli-agent",
    },
)
