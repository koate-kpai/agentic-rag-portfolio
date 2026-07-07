# conftest.py — Path configuration for app1-webscout tests
#
# pytest discovers conftest.py automatically in each test directory tree.
# We add the app's root directory to sys.path so that tests can import
# their modules naturally as "from src.agent import WebScoutAgent".
#
# ARCHITECTURAL DECISION: Per-app conftest isolates sys.path
# -----------------------------------------------------------
# Both apps have a "src" package, which means only one can be in sys.path
# at a time (Python resolves the first match).  Each per-app conftest
# adds only its own app to sys.path.  Because pytest collects tests per
# directory, and conftest.py is scoped to its directory tree, the correct
# path is always set when tests from that app are being collected.
#
# IMPORTANT: Do NOT run both test suites in the same pytest invocation.
# Run them separately:
#   pytest app1-webscout/tests/
#   pytest app2-documind/tests/

import sys
from pathlib import Path

APP_ROOT = Path(__file__).parent.parent
app_path = str(APP_ROOT.resolve())
if app_path not in sys.path:
    sys.path.insert(0, app_path)
