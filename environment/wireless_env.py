# environment/wireless_env.py
import numpy as np

# Change to 4x4
NUM_RX_ANTENNAS = 4
NUM_TX_ANTENNAS = 4

def generate_rayleigh_channel_matrix(rx, tx):
    return (np.random.randn(rx, tx) + 1j * np.random.randn(rx, tx)) / np.sqrt(2)

class LoRaEnv:
    def __init__(self):
        self.H = None
        self.current_snr = None

    def reset(self):
        self.H = generate_rayleigh_channel_matrix(NUM_RX_ANTENNAS, NUM_TX_ANTENNAS)
        # Agriculture IoT Scenario: Normal (Gaussian) Distribution
        self.current_snr = np.random.normal(loc=8.0, scale=3.0) 
        return self._get_flattened_state()

    def set_state(self, state_array):
        """Helper to force the environment into a specific state for fair evaluation"""
        h_real = state_array[:16].reshape(NUM_RX_ANTENNAS, NUM_TX_ANTENNAS)
        h_imag = state_array[16:32].reshape(NUM_RX_ANTENNAS, NUM_TX_ANTENNAS)
        self.H = h_real + 1j * h_imag
        self.current_snr = state_array[-1]

    def _get_flattened_state(self):
        h_flat_real = self.H.real.flatten()
        h_flat_imag = self.H.imag.flatten()
        norm_snr = self.current_snr
        
        # Calculate condition number to represent channel spatial difficulty
        cond_val = np.linalg.cond(self.H)
        norm_cond = min(cond_val / 50.0, 1.0) 
        
        # SNR margin relative to SF7 threshold
        snr_margin = self.current_snr - (-2.5)
        
        state = np.concatenate([
            h_flat_real, 
            h_flat_imag, 
            [norm_snr],
            [norm_cond],   # New Feature: Channel difficulty
            [snr_margin]   # New Feature: Headroom for speed
        ])
        return state # Total size: 35

    def step(self, action_dict, is_stbc=False):
        sfs = [action_dict[f"sf_ant{i}"] for i in range(1, 5)]
        mode = action_dict.get("mode", "standard")
        unique_sfs = len(set(sfs))
        cond_val = np.linalg.cond(self.H)
        
        per = 0
        
        # 1. PHYSICS: INTER-ANTENNA INTERFERENCE (IAI)
        # 1. PHYSICS: INTER-ANTENNA INTERFERENCE (IAI)
        threshold = 35 if is_stbc else 15

        if mode == "constrained_symbol_ml":
            # The paper uses Near-ML detection, which is ultra-robust.
            # We increase the threshold to 100 to show it almost never crashes due to interference.
            threshold = 100 

        if unique_sfs < 2 and cond_val > threshold:
            per = 1.0
                
        # 2. SENSITIVITY: Standard LoRa limits
        snr_map = {7: -2.5, 8: -5, 9: -7.5, 10: -10, 11: -12.5, 12: -15}
        # Standard hardware doesn't support < SF7. We floor it just in case.
        min_sf_requested = max(min(sfs), 7)
        if self.current_snr < snr_map[min_sf_requested]:
            per = 1.0

        if per == 0:
            # 3. FAIR THROUGHPUT CALCULATION
            # Basic throughput heuristic
            raw_reward = sum([(13 - s)**2 for s in sfs])
            
            # MULTIPLIERS FOR FAIRNESS
            multiplier = 1.0
            
            # STBC Penalty: Takes 2 time slots to send symbols (Rate 1/2)
            if is_stbc:
                multiplier *= 0.5
            
            # Coherent Tax: Rugini 24/26 require pilot symbols for channel estimation.
            # In 4x4 MIMO, this overhead is roughly 15-20% of airtime.
            if mode in ["multi_sf_multiplexing", "constrained_symbol_ml"]:
                multiplier *= 0.82 
            
            # Complexity/Energy Penalty: Rugini schemes require high-power DSP.
            # We subtract a small constant for the energy/complexity cost.
            complexity_penalty = 0.5 if mode != "standard" else 0.0
            
            reward = (raw_reward * multiplier) - complexity_penalty
        else:
            reward = 0.0

        info = {"per": per, "snr": self.current_snr, "cond": cond_val}
        return self._get_flattened_state(), reward, info