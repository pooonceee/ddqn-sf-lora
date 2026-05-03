import numpy as np
import matplotlib.pyplot as plt
from environment.wireless_env import LoRaEnv
from agents.dqn_agent import DQNAgent
from agents.competitor_agents import Rugini2026Agent, Ma2021Agent, Rugini2024Agent

# --- PAPER CONFIGURATION ---
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

def run_grouped_sf_analysis():
    env = LoRaEnv()
    num_packets = 1000 
    
    # Initialize Agents with EXACT names from the throughput graph
    agents = {
        "Proposed DQN (C-ADR)": DQNAgent(),
        "Multi-SF (Rugini '26)": Rugini2026Agent(),
        "STBC-MIMO (Ma '21)": Ma2021Agent(),
        "Coherent (Rugini '24)": Rugini2024Agent()
    }
    
    # Load AI Brain
    try:
        agents["Proposed DQN (C-ADR)"].load("lora_mimo_4x4.pth")
        agents["Proposed DQN (C-ADR)"].epsilon = 0
    except:
        print("Warning: Model file not found.")

    # Storage for counts
    sf_labels = [7, 8, 9, 10, 11, 12]
    sf_data = {name: [] for name in agents.keys()}
    
    # Run evaluation
    test_states = [env.reset() for _ in range(num_packets)]
    for name, agent in agents.items():
        print(f"Analyzing {name}...")
        for state in test_states:
            env.set_state(state)
            if "DQN" in name:
                action_idx = agent.choose_action(state)
                action = agent.get_action_format(action_idx)
            else:
                action = agent.choose_action(state)
            
            sfs = [action[f"sf_ant{i}"] for i in range(1, 5)]
            sf_data[name].extend(sfs)

    # --- GROUPED BAR PLOTTING ---
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x = np.arange(len(sf_labels))  # The label locations
    width = 0.2  # The width of the individual bars
    
    # Professional color palette
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    
    for i, (name, sfs) in enumerate(sf_data.items()):
        counts = [sfs.count(sf) for sf in sf_labels]
        percentages = [c / len(sfs) * 100 for c in counts]
        
        # Offset each agent's bars so they group together
        offset = (i - 1.5) * width
        ax.bar(x + offset, percentages, width, label=name, color=colors[i], 
               edgecolor='black', linewidth=0.5, alpha=0.85)

    # Styling
    ax.set_ylabel('Frequency of Use (%)', fontweight='bold')
    ax.set_xlabel('Spreading Factor (SF)', fontweight='bold')
    ax.set_title('Comparative SF Selection Policy Distribution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sf_labels)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True, ncol=2) # 2-column legend for space
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # SAVE FILES
    plt.savefig("sf_distribution_grouped_paper.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("sf_distribution_grouped_paper.png", bbox_inches='tight')
    print("\n[SUCCESS] Figure saved as 'sf_distribution_grouped_paper.pdf' and '.png'")
    plt.show()

if __name__ == "__main__":
    run_grouped_sf_analysis()