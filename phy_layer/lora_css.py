# phy_layer/lora_css.py
import numpy as np

def generate_base_chirp(sf, bw, fs, up=True):
    """
    Generates a base baseband LoRa chirp (either up-chirp or down-chirp).
    
    Smart Comment: We don't simulate a 900 MHz carrier wave. That takes too much compute.
    Instead, we simulate the "baseband equivalent" using complex numbers. 
    The frequency sweeps from -BW/2 to +BW/2 over the symbol duration.
    """
    M = 2**sf                  # Number of samples per symbol
    T = M / bw                 # Symbol duration in seconds
    t = np.arange(0, M) / fs   # Discrete time vector
    
    # Instantaneous frequency sweeps from -BW/2 to BW/2
    # f(t) = (BW / T) * t - (BW / 2)
    # Phase is the integral of frequency over time.
    phase = 2 * np.pi * ((bw / (2 * T)) * t**2 - (bw / 2) * t)
    
    if up:
        # Up-chirp: frequency increases
        return np.exp(1j * phase)
    else:
        # Down-chirp: frequency decreases (conjugate of up-chirp)
        return np.exp(-1j * phase)

def modulate_symbol(symbol, sf, bw, fs):
    """
    Modulates an integer symbol into a LoRa CSS waveform.
    """
    M = 2**sf
    assert 0 <= symbol < M, f"Symbol must be between 0 and {M-1}"
    
    base_upchirp = generate_base_chirp(sf, bw, fs, up=True)
    
    # FIX: Use -symbol to shift the frequency in the positive direction
    modulated_chirp = np.roll(base_upchirp, int(-symbol))
    
    return modulated_chirp

def demodulate_symbol(received_signal, sf, bw, fs):
    """
    Demodulates a received LoRa waveform back into an integer symbol.
    
    Smart Comment: How do we extract the data? We multiply the received signal 
    by a base DOWN-chirp (this is called "de-chirping"). This turns the sweeping 
    signal into a constant frequency tone. We then take the FFT, and the peak 
    frequency bin directly corresponds to our transmitted symbol!
    """
    base_downchirp = generate_base_chirp(sf, bw, fs, up=False)
    
    # 1. De-chirp the signal
    dechirped = received_signal * base_downchirp
    
    # 2. Perform Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(dechirped)
    
    # 3. Find the index of the maximum peak (argmax)
    # The magnitude is used because the channel might alter the phase
    decoded_symbol = np.argmax(np.abs(fft_result))
    
    return decoded_symbol