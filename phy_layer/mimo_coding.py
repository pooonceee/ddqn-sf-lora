# phy_layer/mimo_coding.py
import numpy as np

def apply_mimo_channel(tx_waveforms, H):
    """
    Passes transmitted waveforms through the MIMO channel.
    
    tx_waveforms: Numpy array of shape (num_tx_antennas, num_samples)
    H: Complex channel matrix of shape (num_rx_antennas, num_tx_antennas)
    
    Smart Comment: We use the @ operator for matrix multiplication. 
    This perfectly mixes the signals from all Tx antennas into the Rx antennas 
    based on the fading weights in H.
    """
    rx_waveforms = H @ tx_waveforms
    return rx_waveforms

def zero_forcing_equalizer(rx_waveforms, H):
    """
    Untangles the received MIMO signals using Zero-Forcing.
    
    Smart Comment: Assuming the receiver knows the channel (Perfect CSI), 
    we can separate the overlapping signals by multiplying the received 
    waveforms by the pseudo-inverse of H.
    """
    # 1. Calculate the pseudo-inverse of the channel matrix H
    # Mathematically: H_pinv = (H^H * H)^-1 * H^H
    H_pinv = np.linalg.pinv(H)
    
    # 2. Apply the equalizer to the mashed-up received waveforms
    estimated_tx_waveforms = H_pinv @ rx_waveforms
    
    return estimated_tx_waveforms