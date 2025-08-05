"""
Configuration for pytest tests.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import inf_bp_texts
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
