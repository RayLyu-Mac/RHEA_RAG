# SQLite compatibility patch for Chroma
# Import this file BEFORE importing any Chroma-related modules

import os
import sys

def apply_sqlite_patch():
    """Apply SQLite compatibility patch"""
    try:
        import pysqlite3
        # Replace the built-in sqlite3 with pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ Using pysqlite3 for SQLite compatibility")
        return True
    except ImportError:
        print("⚠️ pysqlite3 not available, using system SQLite")
        print("💡 Install with: pip install pysqlite3-binary")
        return False

# Apply the patch automatically when imported
apply_sqlite_patch() 