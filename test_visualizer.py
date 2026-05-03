# test_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from config import BW, FS, DEFAULT_SF
from phy_layer.lora_css import modulate_symbol, demodulate_symbol, generate_base_chirp

def test_and_visualize():
    print(f"--- Testing LoRa PHY Layer (SF={DEFAULT_SF}) ---")
    
    # 1. Pick a random symbol to transmit
    max_symbol = 2**DEFAULT_SF
    tx_symbol = np.random.randint(0, max_symbol)
    print(f"Transmitted Symbol: {tx_symbol}")
    
    # 2. Modulate the symbol into a waveform
    tx_waveform = modulate_symbol(tx_symbol, DEFAULT_SF, BW, FS)
    
    # 3. (Optional for visualization) Append a base down-chirp so we can see both
    downchirp = generate_base_chirp(DEFAULT_SF, BW, FS, up=False)
    combined_signal = np.concatenate((tx_waveform, downchirp))
    
    # 4. Demodulate the waveform back into a symbol
    rx_symbol = demodulate_symbol(tx_waveform, DEFAULT_SF, BW, FS)
    print(f"Received Symbol:    {rx_symbol}")
    
    if tx_symbol == rx_symbol:
        print("Success! Modulation and Demodulation math is perfectly aligned.")
    else:
        print("Error in DSP logic.")

    # 5. Visualize it using a Spectrogram
    # A spectrogram shows Time on the X-axis and Frequency on the Y-axis.
    f, t, Sxx = signal.spectrogram(combined_signal.real, fs=FS, nperseg=64, noverlap=60)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f - (BW/2), 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.title(f"LoRa CSS Spectrogram (SF={DEFAULT_SF})\nLeft: Modulated Data (Symbol {tx_symbol}) | Right: Base Down-Chirp")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_and_visualize()