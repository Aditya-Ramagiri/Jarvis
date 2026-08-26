"""Adrien - a persistent, wake-word-triggered personal voice assistant.

The package is deliberately layered so the parts that need no hardware and no
network (`config`, `core.keypool`, `tools.registry`, `tools.permissions`,
`server.protocol`) import cleanly on any machine, while the audio/ML stack is
pulled in lazily by the modules that actually need it.
"""

__version__ = "0.1.0"
