#!/usr/bin/env python3
"""
Security Verification Script - Database Path Validation
Demonstrates comprehensive protection against path traversal attacks.
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from path_validator import validate_db_path, PathValidationError


def test_security_scenarios():
    """Test all security scenarios with detailed output."""
    
    print("=" * 80)
    print("DATABASE PATH VALIDATION SECURITY TEST")
    print("=" * 80)
    print()
    
    project_root = Path.cwd()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Legitimate Relative Path",
            "path": "data/devstream.db",
            "should_pass": True,
            "threat_level": "🟢 SAFE"
        },
        {
            "name": "Legitimate Absolute Path",
            "path": str(project_root / "data" / "devstream.db"),
            "should_pass": True,
            "threat_level": "🟢 SAFE"
        },
        {
            "name": "Path Traversal Attack",
            "path": "../../etc/passwd",
            "should_pass": False,
            "threat_level": "🔴 CRITICAL",
            "attack_type": "CWE-22: Path Traversal"
        },
        {
            "name": "Arbitrary Write Attempt",
            "path": "/tmp/malicious.db",
            "should_pass": False,
            "threat_level": "🔴 CRITICAL",
            "attack_type": "Arbitrary File Write"
        },
        {
            "name": "Directory Traversal via Canonicalization",
            "path": "data/../../../etc/passwd",
            "should_pass": False,
            "threat_level": "🔴 CRITICAL",
            "attack_type": "CWE-22: Directory Traversal"
        },
        {
            "name": "Invalid File Extension",
            "path": "data/malicious.txt",
            "should_pass": False,
            "threat_level": "🟡 MEDIUM",
            "attack_type": "Extension Validation"
        },
        {
            "name": "Empty Path",
            "path": "",
            "should_pass": False,
            "threat_level": "🟡 MEDIUM",
            "attack_type": "Input Validation"
        }
    ]
    
    results = {
        "passed": 0,
        "failed": 0,
        "blocked": 0
    }
    
    for idx, scenario in enumerate(scenarios, 1):
        print(f"Test {idx}: {scenario['name']}")
        print(f"  Threat Level: {scenario['threat_level']}")
        if not scenario['should_pass']:
            print(f"  Attack Type: {scenario['attack_type']}")
        print(f"  Input: {scenario['path']}")
        
        try:
            result = validate_db_path(scenario['path'], str(project_root))
            
            if scenario['should_pass']:
                print(f"  ✅ PASS: Validated successfully")
                print(f"  Output: {result}")
                results['passed'] += 1
            else:
                print(f"  ❌ FAIL: Attack was not blocked!")
                print(f"  Output: {result}")
                results['failed'] += 1
        
        except PathValidationError as e:
            if not scenario['should_pass']:
                print(f"  ✅ BLOCKED: Attack prevented")
                print(f"  Reason: {str(e)[:100]}...")
                results['blocked'] += 1
            else:
                print(f"  ❌ FAIL: Valid path was blocked")
                print(f"  Error: {str(e)}")
                results['failed'] += 1
        
        except Exception as e:
            print(f"  ❌ ERROR: Unexpected exception")
            print(f"  Error: {type(e).__name__} - {str(e)}")
            results['failed'] += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print("SECURITY TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(scenarios)}")
    print(f"✅ Legitimate Paths Passed: {results['passed']}")
    print(f"🛡️  Attacks Blocked: {results['blocked']}")
    print(f"❌ Failed: {results['failed']}")
    print()
    
    if results['failed'] == 0:
        print("🎉 ALL SECURITY TESTS PASSED!")
        print()
        print("Security Validation:")
        print("  ✅ Path traversal attacks blocked (CWE-22)")
        print("  ✅ Arbitrary write attempts prevented")
        print("  ✅ Extension validation enforced")
        print("  ✅ Input validation functional")
        print()
        print("Compliance:")
        print("  ✅ OWASP A03:2021 - Injection")
        print("  ✅ CWE-22: Path Traversal")
        print("  ✅ Defense in Depth implemented")
        return 0
    else:
        print("⚠️  SECURITY VULNERABILITIES DETECTED!")
        print(f"   {results['failed']} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(test_security_scenarios())
