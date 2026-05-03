# agents/baseline_lora.py
class BaselineADRAgent:
    def choose_action(self, state):
        # State[-1] is the SNR
        snr = state[-1]
        
        # Standard LoRa ADR ignores the MIMO matrix completely
        if snr > 5:   sf = 7
        elif snr > 0: sf = 8
        elif snr > -5:sf = 9
        elif snr > -10:sf= 10
        elif snr > -15:sf= 11
        else:         sf = 12
            
        return {"sf_ant1": sf, "sf_ant2": sf, "sf_ant3": sf, "sf_ant4": sf}