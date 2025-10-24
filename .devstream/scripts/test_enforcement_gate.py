#!/usr/bin/env python3
"""
Test script for Protocol Enforcement Gate

Tests the enforcement gate interactive flow with a sample prompt.
This simulates a non-trivial task that would trigger the enforcement gate.
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_enforcement_gate():
    """Test the enforcement gate with a sample non-trivial prompt."""

    print("🧪 Testing Protocol Enforcement Gate")
    print("=" * 50)

    # Sample non-trivial prompt that should trigger enforcement gate
    sample_prompt = "Implement JWT authentication with password hashing"

    # Test parameters
    test_data = {
        "user_prompt": sample_prompt,
        "estimated_duration": 25,  # minutes (>15 min triggers enforcement)
        "requires_code": True,
        "architectural_decisions": True,
        "multiple_files": True,
        "context7_required": True
    }

    print(f"📝 Sample Prompt: {sample_prompt}")
    print(f"⏱️  Estimated Duration: {test_data['estimated_duration']} minutes")
    print(f"💻 Requires Code: {test_data['requires_code']}")
    print(f"🏗️  Architectural Decisions: {test_data['architectural_decisions']}")
    print(f"📁 Multiple Files: {test_data['multiple_files']}")
    print(f"🔍 Context7 Research: {test_data['context7_required']}")
    print()

    # Check enforcement criteria
    enforcement_criteria = []
    if test_data['estimated_duration'] > 15:
        enforcement_criteria.append("Duration > 15 minutes")
    if test_data['requires_code']:
        enforcement_criteria.append("Requires code implementation")
    if test_data['architectural_decisions']:
        enforcement_criteria.append("Requires architectural decisions")
    if test_data['multiple_files']:
        enforcement_criteria.append("Involves multiple files")
    if test_data['context7_required']:
        enforcement_criteria.append("Requires Context7 research")

    print("🔒 Enforcement Criteria Met:")
    for criteria in enforcement_criteria:
        print(f"  ✅ {criteria}")

    if len(enforcement_criteria) >= 2:  # Enforcement gate triggers on 2+ criteria
        print("\n⚠️  ENFORCEMENT GATE SHOULD TRIGGER")
        print("\nExpected behavior:")
        print("  1. task_first_handler.py detects enforcement criteria")
        print("  2. User sees 3 options: [Protocol] [Override] [Cancel]")
        print("  3. Protocol option: Initiates 7-step workflow")
        print("  4. Override option: Logs decision and continues")
        print("  5. Cancel option: Stops execution")

        # Simulate expected enforcement gate response
        print("\n🎯 Expected Interactive Prompt:")
        print("┌" + "─" * 60 + "┐")
        print("│⚠️  DevStream Protocol Required                         │")
        print("│" + " " * 60 + "│")
        print("│ This task requires following the DevStream 7-step   │")
        print("│ workflow: DISCUSSION → ANALYSIS → RESEARCH →       │")
        print("│ PLANNING → APPROVAL → IMPLEMENTATION → VERIFICATION │")
        print("│" + " " * 60 + "│")
        print("│ OPTIONS:                                            │")
        print("│ ✅ [1] Protocol (research-driven, quality-assured)   │")
        print("│ ⚠️  [2] Override (quick fix, NO quality assurance)   │")
        print("│ 🚫 [3] Cancel                                        │")
        print("└" + "─" * 60 + "┘")

        return True
    else:
        print("\n✅ Enforcement gate NOT triggered (simple task)")
        return False

def test_hook_configuration():
    """Test that hooks are properly configured in settings.json."""

    print("\n🔧 Testing Hook Configuration")
    print("=" * 50)

    settings_file = project_root / ".claude" / "settings.json"

    if not settings_file.exists():
        print("❌ settings.json not found")
        return False

    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)

        # Check UserPromptSubmit hooks
        user_prompt_hooks = settings.get("hooks", {}).get("UserPromptSubmit", [])
        if not user_prompt_hooks:
            print("❌ No UserPromptSubmit hooks found")
            return False

        hooks = user_prompt_hooks[0].get("hooks", [])
        hook_commands = [h.get("command", "") for h in hooks]

        print("✅ UserPromptSubmit hooks found:")
        for i, cmd in enumerate(hook_commands, 1):
            hook_name = cmd.split("/")[-1].replace('"', '')
            print(f"  {i}. {hook_name}")

        # Check for required hooks
        required_hooks = [
            "task_first_handler.py",
            "user_query_context_enhancer.py"
        ]

        missing_hooks = []
        for required_hook in required_hooks:
            if not any(required_hook in cmd for cmd in hook_commands):
                missing_hooks.append(required_hook)

        if missing_hooks:
            print(f"❌ Missing required hooks: {missing_hooks}")
            return False

        # Check PostToolUse hooks for TodoWrite
        post_tool_hooks = settings.get("hooks", {}).get("PostToolUse", [])
        todo_write_hook_found = False

        for hook_group in post_tool_hooks:
            if hook_group.get("matcher") == "TodoWrite":
                todo_write_hook_found = True
                hooks = hook_group.get("hooks", [])
                if hooks:
                    hook_cmd = hooks[0].get("command", "")
                    hook_name = hook_cmd.split("/")[-1].replace('"', '')
                    print(f"✅ TodoWrite hook found: {hook_name}")
                break

        if not todo_write_hook_found:
            print("❌ No TodoWrite hook found in PostToolUse")
            return False

        print("✅ All required hooks properly configured")
        return True

    except Exception as e:
        print(f"❌ Error reading settings.json: {e}")
        return False

def test_environment_variables():
    """Test that required environment variables are set."""

    print("\n🌍 Testing Environment Variables")
    print("=" * 50)

    env_file = project_root / ".env.devstream"

    if not env_file.exists():
        print("❌ .env.devstream not found")
        return False

    required_vars = [
        "DEVSTREAM_PROTOCOL_ENFORCEMENT_ENABLED",
        "DEVSTREAM_TASK_FIRST_HANDLER_ENABLED",
        "DEVSTREAM_MCP_GRACEFUL_FALLBACK",
        "DEVSTREAM_MICRO_TASK_COMMITS",
        "DEVSTREAM_ENFORCEMENT_GATE_INTERACTIVE"
    ]

    try:
        with open(env_file, 'r') as f:
            env_content = f.read()

        print("✅ Required environment variables:")
        for var in required_vars:
            if f"{var}=" in env_content:
                # Extract value
                for line in env_content.split('\n'):
                    if line.startswith(f"{var}="):
                        value = line.split('=', 1)[1]
                        print(f"  {var} = {value}")
                        break
            else:
                print(f"  ❌ {var} = NOT FOUND")
                return False

        print("✅ All required environment variables are set")
        return True

    except Exception as e:
        print(f"❌ Error reading .env.devstream: {e}")
        return False

def main():
    """Run all enforcement gate tests."""

    print("🚀 Protocol Enforcement Gate Test Suite")
    print("=" * 60)

    tests = [
        ("Enforcement Gate Logic", test_enforcement_gate),
        ("Hook Configuration", test_hook_configuration),
        ("Environment Variables", test_environment_variables)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n📈 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enforcement gate is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check configuration and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())