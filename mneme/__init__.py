"""Mneme — offline visual memory backend.

Contracts: docs/spec.md (external HTTP API), docs/backend.md (implementation),
docs/sidecar.md (inference wire protocol).
"""

__version__ = "1.4.0"

EMBED_TIMEOUT_MS = 5000
"""docs/sidecar.md: Embed always uses 5000ms, never --sidecar-timeout-ms."""
