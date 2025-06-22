#!/usr/bin/env python3
"""Test runner script for cli-agent."""

import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=True):
    """Run tests with specified options."""
    cmd = ["python", "-m", "pytest"]

    # Add test path based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        cmd.append(test_type)  # Custom path

    # Add options
    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=cli_agent", "--cov=.", "--cov-report=term-missing"])

    # Add markers for different test types
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])

    print(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd)


def run_lint():
    """Run linting checks."""
    print("Running flake8...")
    subprocess.run(
        ["python", "-m", "flake8", "cli_agent/", "tests/", "--max-line-length=127"]
    )

    print("\nRunning black check...")
    subprocess.run(["python", "-m", "black", "--check", "."])

    print("\nRunning isort check...")
    subprocess.run(["python", "-m", "isort", "--check-only", "."])


def run_type_check():
    """Run type checking."""
    print("Running mypy...")
    subprocess.run(["python", "-m", "mypy", "--ignore-missing-imports", "cli_agent/"])


def main():
    """Main test runner."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [unit|integration|all|lint|type|path]")
        print("Examples:")
        print("  python run_tests.py unit          # Run unit tests")
        print("  python run_tests.py integration   # Run integration tests")
        print("  python run_tests.py all           # Run all tests")
        print("  python run_tests.py lint          # Run linting")
        print("  python run_tests.py type          # Run type checking")
        print(
            "  python run_tests.py tests/unit/test_base_agent.py  # Run specific test"
        )
        sys.exit(1)

    test_type = sys.argv[1]
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    no_coverage = "--no-coverage" in sys.argv

    if test_type == "lint":
        run_lint()
    elif test_type == "type":
        run_type_check()
    else:
        result = run_tests(test_type, verbose=verbose, coverage=not no_coverage)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
