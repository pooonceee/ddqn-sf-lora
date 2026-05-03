import numpy as np
import matplotlib.pyplot as plt
from environment.wireless_env import LoRaEnv
from agents.dqn_agent import DQNAgent
from agents.competitor_agents import Rugini2026Agent, Ma2021Agent, Rugini2024Agent

# --- PAPER CONFIGURATION ---
# Optimized for a single-column (3.5 inch) or double-column (7 inch) paper
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

def plot_paper_throughput(results):
    """
    Generates a high-quality cumulative throughput plot.
    """
    fig, ax = plt.subplots(figsize=(5, 4)) # Professional aspect ratio
    
    # Professional Colors & Line Styles for B&W printing compatibility
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    line_styles = ['-', '--', '-.', ':']

    for i, (name, data) in enumerate(results.items()):
        cumulative_reward = np.cumsum(data["rewards"])
        
        # We plot every 100th point with a marker to make lines distinct
        ax.plot(cumulative_reward, 
                label=name, 
                color=colors[i], 
                linestyle=line_styles[i], 
                linewidth=2)

    # Formatting
    ax.set_title("Network Throughput: SOTA Comparison", fontweight='bold')
    ax.set_xlabel("Packet Sequence ($k$)", fontsize=10)
    ax.set_ylabel("Cumulative Throughput Score", fontsize=10)
    
    # Scientific notation for large numbers on Y axis
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # Save files
    plt.savefig("throughput_comparison_paper.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("throughput_comparison_paper.png", bbox_inches='tight')
    print("\n[SUCCESS] Figure saved as 'throughput_comparison_paper.pdf' and '.png'")
    plt.show()

def run_comparison():
    env = LoRaEnv()
    num_packets = 1000000
    
    # Initialize Agents
    agents = {
        "Proposed DQN (C-ADR)": DQNAgent(),
        "Multi-SF (Rugini '26)": Rugini2026Agent(),
        "STBC-MIMO (Ma '21)": Ma2021Agent(),
        "Coherent (Rugini '24)": Rugini2024Agent()
    }
    
    # Load AI model
    try:
        agents["Proposed DQN (C-ADR)"].load("lora_mimo_4x4.pth")
        agents["Proposed DQN (C-ADR)"].epsilon = 0
    except:
        print("Model file not found. Running with initial weights.")
    
    results = {name: {"rewards": [], "errors": 0} for name in agents.keys()}
    test_states = [env.reset() for _ in range(num_packets)]

    for name, agent in agents.items():
        print(f"Running Evaluation: {name}")
        for state in test_states:
            env.set_state(state)
            is_stbc = (name == "STBC-MIMO (Ma '21)")
            
            if "DQN" in name:
                action_idx = agent.choose_action(state)
                action = agent.get_action_format(action_idx)
            else:
                action = agent.choose_action(state)
                
            _, reward, info = env.step(action, is_stbc=is_stbc)
            results[name]["rewards"].append(reward)

    # Generate the Throughput Figure
    plot_paper_throughput(results)

if __name__ == "__main__":
    run_comparison()