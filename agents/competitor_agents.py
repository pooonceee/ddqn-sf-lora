import numpy as np

class Rugini2026Agent:
    def choose_action(self, state):
        snr = state[-1] # The SNR is the last element of the state
        
        # Base SF selection based on link quality
        if snr > 5:   base_sf = 7
        elif snr > 0: base_sf = 8
        else:         base_sf = 9
        
        # NEW LOGIC: Assign a unique SF to EVERY antenna
        # This ensures a 25/25/25/25 distribution across the 4 antennas.
        return {
            "sf_ant1": base_sf,
            "sf_ant2": min(base_sf + 1, 12),
            "sf_ant3": min(base_sf + 2, 12),
            "sf_ant4": min(base_sf + 3, 12)
        }

class Rugini2024Agent:
    def choose_action(self, state):
        snr = state[-1]
        
        # The paper prioritizes high data rate via multiplexing
        if snr > 2:    s = 7 
        elif snr > -5: s = 8
        else:          s = 9
        
        # CORRECT IMPLEMENTATION: All antennas use the SAME SF
        return {
            "sf_ant1": s, "sf_ant2": s, "sf_ant3": s, "sf_ant4": s
        }

class Ma2021Agent:
    """
    Strategy: STBC-MIMO (Maximum Reliability)
    Based on Ma et al. (2021).
    
    Fixed: All antennas use the EXACT SAME SF.
    This is required for the Alamouti matrix math to work.
    """
    def choose_action(self, state):
        snr = state[-1]
        
        # Diversity gain from STBC allows us to use lower SFs than 
        # standard LoRa at the same SNR.
        if snr > 0:    sf = 7
        elif snr > -5: sf = 8
        else:          sf = 10
        
        return {
            "sf_ant1": sf,
            "sf_ant2": sf,
            "sf_ant3": sf,
            "sf_ant4": sf
        }