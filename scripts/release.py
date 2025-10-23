#!/usr/bin/env python3
"""
Release script for exc-to-pdf project.

This script automates the release process:
1. Runs tests
2. Builds package
3. Creates git tag
4. Pushes to repository
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> bool:
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    """Main release process."""
    if len(sys.argv) != 2:
        print("Usage: python release.py <version>")
        sys.exit(1)

    version = sys.argv[1]
    print(f"Releasing version {version}")

    # Verify version format
    if not version.startswith('v'):
        print("Version should start with 'v' (e.g., v1.0.0)")
        sys.exit(1)

    # 1. Run tests
    print("\n1. Running tests...")
    if not run_command(['pytest', '--cov=exc_to_pdf', '--cov-fail-under=90']):
        print("Tests failed!")
        sys.exit(1)

    # 2. Build package
    print("\n2. Building package...")
    if not run_command(['python', '-m', 'build']):
        print("Build failed!")
        sys.exit(1)

    # 3. Add all files to git
    print("\n3. Adding files to git...")
    if not run_command(['git', 'add', '.']):
        print("Git add failed!")
        sys.exit(1)

    # 4. Commit changes
    print("\n4. Committing changes...")
    if not run_command(['git', 'commit', '-m', f'Release {version}']):
        print("Git commit failed!")
        sys.exit(1)

    # 5. Create tag
    print(f"\n5. Creating tag {version}...")
    if not run_command(['git', 'tag', version]):
        print("Git tag failed!")
        sys.exit(1)

    # 6. Push to main branch
    print("\n6. Pushing to main branch...")
    if not run_command(['git', 'push', 'origin', 'main']):
        print("Git push failed!")
        sys.exit(1)

    # 7. Push tag
    print(f"\n7. Pushing tag {version}...")
    if not run_command(['git', 'push', 'origin', version]):
        print("Git tag push failed!")
        sys.exit(1)

    print(f"\n✅ Release {version} completed successfully!")
    print("The CI/CD pipeline will handle publishing to PyPI.")


if __name__ == "__main__":
    main()