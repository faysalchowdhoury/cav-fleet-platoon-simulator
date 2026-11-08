"""cav-fleet-platoon-simulator.

Vectorised single-lane mixed-autonomy simulator: human drivers follow a Wiedemann 99
re-implementation (CC0-CC9), automated vehicles run the PATH ACC / CACC controllers with V2V
packet loss and fallback, and experiments measure queue-discharge capacity, string stability
and TTC safety across penetration rates.
"""

__version__ = "1.0.0"
