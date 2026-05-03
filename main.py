# main.py
from environment.wireless_env import LoRaEnv
from agents.baseline_lora import BaselineADRAgent

def run_simulation():
    print("--- Starting LoRa Baseline ADR Simulation ---")
    
    env = LoRaEnv()
    agent = BaselineADRAgent()
    
    # Get the starting conditions
    state = env.reset()
    
    total_reward = 0
    
    # Simulate sending 20 packets through changing weather/distance
    for step in range(1, 21):
        # 1. Agent looks at the state (array) and picks SFs
        action = agent.choose_action(state)
        
        # 2. Environment runs the MIMO physics
        next_state, reward, info = env.step(action)
        
        # 3. Log the results
        snr = state[-1] # SNR is the last element of the state array
        sf1 = action['sf_ant1']
        sf2 = action['sf_ant2']
        per = info['per']
        
        print(f"Packet {step:2}: SNR = {snr:6.2f} dB | Agent chose SF{sf1}/SF{sf2} | Error Rate: {per*100:5.1f}% | Reward: {reward:4}")
        
        # Move to the next time step
        state = next_state
        total_reward += reward
        
    print(f"\nSimulation Complete. Baseline Agent Total Score: {total_reward}")

if __name__ == "__main__":
    run_simulation()