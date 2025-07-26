# SQLite compatibility patch for Chroma
# Import this file BEFORE importing any Chroma-related modules
# Based on Stack Overflow solution for SQLite version compatibility

import sys

def apply_sqlite_patch():
    """Apply SQLite compatibility patch using pysqlite3-binary"""
    try:
        # Force import of pysqlite3
        __import__('pysqlite3')
        # Replace the built-in sqlite3 with pysqlite3
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("✅ Successfully patched sqlite3 to use pysqlite3")
        return True
    except ImportError:
        print("⚠️ pysqlite3-binary not available, using system SQLite")
        print("💡 Install with: pip install pysqlite3-binary")
        return False

# Apply the patch automatically when imported
apply_sqlite_patch() 