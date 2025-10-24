#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
# ]
# ///

"""
Test SQL Injection Protection for PreToolUse Hook
Tests the _sanitize_query_elements function with various attack vectors.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).parent))

# Import the function to test
from pre_tool_use import _sanitize_query_elements


def test_sql_injection_protection():
    """
    Test SQL injection protection with various attack vectors.
    """
    print("🔒 Testing SQL Injection Protection\n")

    # Test cases: (input_elements, expected_behavior, description)
    test_cases = [
        # Safe inputs (should pass through)
        (["filename", "class_name", "function_name"], True, "Safe code elements"),
        (["MyClass", "my_method", "import_fastapi"], True, "Mixed case safe elements"),
        (["test-file.py", "user_data", "config.json"], True, "Safe elements with hyphens/dots"),

        # SQL injection attempts (should be blocked)
        (["'; DROP TABLE users; --"], False, "SQL injection with DROP TABLE"),
        (["' OR 1=1 --"], False, "Classic OR 1=1 injection"),
        (["' OR 'x'='x"], False, "String comparison injection"),
        (["UNION SELECT * FROM passwords"], False, "UNION injection attempt"),
        (["admin'/*", "*/--"], False, "Block comment injection"),
        (["WAITFOR DELAY '00:00:05'"], False, "Time-based injection"),
        (["sleep(5)"], False, "MySQL time-based injection"),
        (["xp_cmdshell 'dir'"], False, "Extended stored procedure injection"),
        (["information_schema.tables"], False, "Information schema access"),

        # Mixed safe + dangerous (should filter dangerous)
        (["MyClass", "'; DROP TABLE", "safe_method"], True, "Mixed safe and dangerous elements"),

        # Encoding-based attacks (should be blocked)
        (["0x414243"], False, "Hex encoding injection"),
        (["CHAR(65)"], False, "CHAR function injection"),

        # Edge cases
        (["", "", ""], False, "Empty elements"),
        (["a"], False, "Too short element"),
        ([" " * 200], False, "Long whitespace element"),

        # Unicode/advanced attacks
        (["' OR 1=1#"], False, "MySQL comment injection"),
        (["' OR 'a'='a'/*"], False, "Block comment in injection"),
    ]

    passed = 0
    failed = 0

    print("Running SQL injection protection test cases:\n")

    for i, (input_elements, should_pass, description) in enumerate(test_cases, 1):
        print(f"Test {i:2d}: {description}")
        print(f"   Input: {input_elements}")

        try:
            result = _sanitize_query_elements(input_elements)

            if should_pass:
                if result and len(result) > 0:
                    print(f"   ✅ PASS: Elements preserved safely")
                    print(f"   Output: {result}")
                    passed += 1
                else:
                    print(f"   ❌ FAIL: Expected safe elements to pass, but got empty result")
                    failed += 1
            else:
                if not result or len(result) == 0:
                    print(f"   ✅ PASS: Injection blocked")
                    passed += 1
                else:
                    print(f"   ❌ FAIL: Expected injection to be blocked, but got: {result}")
                    failed += 1

        except Exception as e:
            print(f"   ❌ ERROR: Exception during test - {str(e)}")
            failed += 1

        print()

    # Summary
    total = passed + failed
    print("=" * 50)
    print(f"🧪 SQL Injection Protection Test Results:")
    print(f"   Total tests: {total}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   Success rate: {(passed/total*100):.1f}%")

    if failed == 0:
        print("\n🎉 All tests passed! SQL injection protection is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review the protection implementation.")

    return failed == 0


if __name__ == "__main__":
    success = test_sql_injection_protection()
    sys.exit(0 if success else 1)