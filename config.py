# config.py
"""
Global Configuration Parameters for the LoRa MIMO Simulation.
Keeping these centralized makes it easy to test different scenarios.
"""

# LoRa Physical Parameters
BW = 125e3          # Bandwidth in Hz (125 kHz is standard for LoRa)
FS = 125e3          # Sampling Frequency. In baseband simulation, FS = BW is optimal.
DEFAULT_SF = 8      # Default Spreading Factor (can range from 7 to 12)

# MIMO Parameters (We will use these later)
NUM_TX_ANTENNAS = 2
NUM_RX_ANTENNAS = 2

# Simulation Parameters
PACKET_LENGTH = 20  # Number of symbols per simulated packet