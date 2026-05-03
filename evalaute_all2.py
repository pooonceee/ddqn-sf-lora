import numpy as np
import matplotlib.pyplot as plt
from environment.wireless_env import LoRaEnv
from agents.dqn_agent import DQNAgent
from agents.competitor_agents import Rugini2026Agent, Ma2021Agent, Rugini2024Agent

def run_comparison():
    env = LoRaEnv()
    num_packets = 1000  # Evaluation length
    
    # Initialize Agents
    agents = {
        "Proposed DQN (AI)": DQNAgent(),
        "Static SF (Rugini '26)": Rugini2026Agent(),
        "STBC-MIMO (Ma '21)": Ma2021Agent(),
        "Coherent (Rugini '24)": Rugini2024Agent()
    }
    
    # Load AI Brain
    try:
        agents["Proposed DQN (AI)"].load("lora_mimo_4x4.pth")
        agents["Proposed DQN (AI)"].epsilon = 0
    except:
        print("Warning: Model file not found. Running with untrained AI.")
    
    results = {name: {"rewards": [], "errors": 0} for name in agents.keys()}
    
    # Generate fixed states for fair comparison
    test_states = [env.reset() for _ in range(num_packets)]

    print(f"--- Starting Evaluation over {num_packets} packets ---")
    for name, agent in agents.items():
        print(f"Testing {name}...")
        for state in test_states:
            env.set_state(state)
            
            # Identify physics flags
            is_stbc = (name == "STBC-MIMO (Ma '21)")
            
            if name == "Proposed DQN (AI)":
                action_idx = agent.choose_action(state)
                action = agent.get_action_format(action_idx)
            else:
                action = agent.choose_action(state)
                
            _, reward, info = env.step(action, is_stbc=is_stbc)
            results[name]["rewards"].append(reward)
            if info["per"] > 0: 
                results[name]["errors"] += 1

    # --- PRINT FINAL STATISTICS ---
    print("\n" + "="*60)
    print(f"{'Scheme':<25} | {'Avg. Reward':<12} | {'PER (%)':<10}")
    print("-" * 60)
    
    for name, data in results.items():
        avg_reward = np.mean(data["rewards"])
        per = (data["errors"] / num_packets) * 100
        print(f"{name:<25} | {avg_reward:<12.2f} | {per:<10.2f}%")
    print("="*60 + "\n")

    # --- PLOTTING ---
    plt.figure(figsize=(14, 7))
    
    for name, data in results.items():
        # Plotting raw reward values for every packet
        plt.plot(data["rewards"], label=f"{name}", alpha=0.6, linewidth=1)
    
    plt.title("Instantaneous Throughput Reward per Packet")
    plt.xlabel("Packet Sequence")
    plt.ylabel("Reward (Raw Throughput)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()