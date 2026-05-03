# test_mimo.py
import numpy as np
from config import BW, FS, DEFAULT_SF, NUM_TX_ANTENNAS, NUM_RX_ANTENNAS
from phy_layer.lora_css import modulate_symbol, demodulate_symbol
from phy_layer.channel_model import generate_rayleigh_channel_matrix, add_awgn
from phy_layer.mimo_coding import apply_mimo_channel, zero_forcing_equalizer

def test_spatial_multiplexing():
    print(f"--- Testing 2x2 MIMO Spatial Multiplexing (SF={DEFAULT_SF}) ---")
    
    max_symbol = 2**DEFAULT_SF
    
    # 1. Generate TWO different random symbols for our TWO antennas
    tx_symbols = [np.random.randint(0, max_symbol) for _ in range(NUM_TX_ANTENNAS)]
    print(f"Transmitted Symbols: Antenna 1 -> {tx_symbols[0]}, Antenna 2 -> {tx_symbols[1]}")
    
    # 2. Modulate both symbols into waveforms
    waveform_1 = modulate_symbol(tx_symbols[0], DEFAULT_SF, BW, FS)
    waveform_2 = modulate_symbol(tx_symbols[1], DEFAULT_SF, BW, FS)
    
    # Stack them into a matrix of shape (2, num_samples)
    tx_waveforms = np.vstack((waveform_1, waveform_2))
    
    # 3. Generate a 2x2 Rayleigh Fading Channel Matrix (H)
    H = generate_rayleigh_channel_matrix(NUM_RX_ANTENNAS, NUM_TX_ANTENNAS)
    
    # 4. Pass the signals through the channel (This mashes them together)
    clean_rx_waveforms = apply_mimo_channel(tx_waveforms, H)
    
    # 5. Add a bit of noise (Let's use a safe 10 dB SNR for now)
    noisy_rx_waveforms = add_awgn(clean_rx_waveforms, snr_db=10)
    
    # 6. RECEIVER: Equalize the signals to untangle them
    estimated_tx_waveforms = zero_forcing_equalizer(noisy_rx_waveforms, H)
    
    # 7. Demodulate the two separated waveforms
    rx_symbol_1 = demodulate_symbol(estimated_tx_waveforms[0], DEFAULT_SF, BW, FS)
    rx_symbol_2 = demodulate_symbol(estimated_tx_waveforms[1], DEFAULT_SF, BW, FS)
    
    print(f"Received Symbols:    Antenna 1 -> {rx_symbol_1}, Antenna 2 -> {rx_symbol_2}")
    
    if tx_symbols[0] == rx_symbol_1 and tx_symbols[1] == rx_symbol_2:
        print("\nSuccess! The Zero-Forcing Equalizer perfectly untangled the MIMO signals.")
    else:
        print("\nFailure! Signals collided and corrupted.")

if __name__ == "__main__":
    test_spatial_multiplexing()