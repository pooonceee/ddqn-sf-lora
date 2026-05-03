# test_channel.py
import numpy as np
from config import BW, FS, DEFAULT_SF
from phy_layer.lora_css import modulate_symbol, demodulate_symbol
from phy_layer.channel_model import add_awgn

def test_waterfall_curve():
    print(f"--- Testing LoRa Robustness to Noise (SF={DEFAULT_SF}) ---")
    
    # We will test SNRs from +5 dB down to -25 dB
    snr_range = np.arange(5, -26, -2)
    num_symbols_per_snr = 100  
    
    max_symbol = 2**DEFAULT_SF
    
    for snr in snr_range:
        errors = 0
        
        for _ in range(num_symbols_per_snr):
            # 1. Generate random data
            tx_symbol = np.random.randint(0, max_symbol)
            
            # 2. Modulate
            tx_waveform = modulate_symbol(tx_symbol, DEFAULT_SF, BW, FS)
            
            # 3. Pass through the noisy channel (Single Antenna, just AWGN for now)
            rx_waveform = add_awgn(tx_waveform, snr)
            
            # 4. Demodulate
            rx_symbol = demodulate_symbol(rx_waveform, DEFAULT_SF, BW, FS)
            
            if tx_symbol != rx_symbol:
                errors += 1
                
        error_rate = errors / num_symbols_per_snr
        print(f"SNR: {snr:3} dB | Symbol Error Rate (SER): {error_rate*100:6.2f}%")
        
        # Stop testing if it's completely failing
        if error_rate > 0.9:
            print(f"\nLimit Reached! SF{DEFAULT_SF} completely fails around {snr} dB.")
            break

if __name__ == "__main__":
    test_waterfall_curve()