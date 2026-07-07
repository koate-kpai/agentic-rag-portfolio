# __init__.py — Explicit package marker for app1-webscout.src
#
# ARCHITECTURAL DECISION: Explicit __init__.py over implicit namespace packages
# -----------------------------------------------------------------------------
# Python 3.3+ supports implicit namespace packages (packages without __init__.py)
# when directories don't collide across sys.path.  However, several widely-used
# tools still require or behave better with explicit markers:
#   - mypy < 1.0  failed to resolve imports in namespace packages in some configs.
#   - IDEs (PyCharm, VS Code Pylance) sometimes skip namespace directories during
#     "Find Usages" and refactoring.
#   - pytest's `--import-mode=importlib` (the default since pytest 7) handles
#     both, but `--import-mode=prepend` (still common) can silently import the
#     wrong module without an __init__.py guard.
#
# The cost is one empty file per package directory — negligible overhead for
# guaranteed tooling compatibility.

