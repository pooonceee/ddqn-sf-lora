# phy_layer/channel_model.py
import numpy as np

def add_awgn(signal, snr_db):
    """
    Adds Additive White Gaussian Noise (AWGN) to a baseband signal.
    
    Smart Comment: SNR (Signal-to-Noise Ratio) is in decibels (dB). 
    A negative SNR means the noise is actually LOUDER than the signal. 
    One of LoRa's superpowers is its ability to decode signals even when SNR is negative!
    """
    # 1. Measure the average power of the transmitted signal
    sig_power = np.mean(np.abs(signal)**2)
    
    # 2. Convert SNR from decibels (dB) to a linear scale
    snr_linear = 10**(snr_db / 10)
    
    # 3. Calculate the required noise power
    noise_power = sig_power / snr_linear
    
    # 4. Generate Complex Gaussian Noise (White Noise)
    # We divide by 2 because the power is split between the real and imaginary parts
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    
    return signal + noise

def generate_rayleigh_channel_matrix(num_rx, num_tx):
    """
    Generates a MIMO Flat Rayleigh fading channel matrix (H).
    
    Smart Comment: 'Flat' means we assume the channel environment doesn't change 
    *during* the transmission of a single symbol. For LoRa's narrow bandwidth (125kHz), 
    this is a very accurate assumption. 
    Returns an (num_rx x num_tx) matrix of complex numbers.
    """
    # Rayleigh fading is mathematically represented by a complex Gaussian variable.
    # The variance is 1 (0.5 for real, 0.5 for imag) so the channel doesn't magically create energy.
    H = np.sqrt(0.5) * (np.random.randn(num_rx, num_tx) + 1j * np.random.randn(num_rx, num_tx))
    
    return H