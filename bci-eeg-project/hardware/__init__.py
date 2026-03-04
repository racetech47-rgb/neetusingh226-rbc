"""
hardware/__init__.py
---------------------
Hardware interface package for real EEG headset support.

Supported devices:
  - Muse (via Lab Streaming Layer / pylsl)
  - OpenBCI Cyton/Ganglion (via brainflow)

Both modules provide graceful fallback to simulated data when no physical
device is available.
"""
