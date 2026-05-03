# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from environment.wireless_env import LoRaEnv
from agents.baseline_lora import BaselineADRAgent
from agents.dqn_agent import DQNAgent

def evaluate_models():
    print("--- Running Final Head-to-Head 4x4 Evaluation ---")
    
    env = LoRaEnv()
    baseline_agent = BaselineADRAgent()
    
    # Load the trained AI
    ai_agent = DQNAgent()
    ai_agent.load("lora_mimo_4x4.pth") # LOAD THE BRAIN!
    ai_agent.epsilon = 0.0  # Zero randomness. Pure AI decision making.
    
    num_test_packets = 50
    baseline_rewards, ai_rewards = [], []
    baseline_errors, ai_errors = 0, 0
    
    state = env.reset()
    
    for step in range(num_test_packets):
        current_state = np.copy(state)
        
        # BASELINE
        env.set_state(current_state)
        action_base = baseline_agent.choose_action(current_state)
        _, reward_base, info_base = env.step(action_base)
        baseline_rewards.append(reward_base)
        if info_base['per'] > 0: baseline_errors += 1
        
        # AI
        env.set_state(current_state) 
        action_ai_idx = ai_agent.choose_action(current_state)
        action_ai = ai_agent.get_action_format(action_ai_idx)
        next_state, reward_ai, info_ai = env.step(action_ai)
        ai_rewards.append(reward_ai)
        if info_ai['per'] > 0: ai_errors += 1
        
        # Move forward
        state = next_state
        
    print(f"\nResults over {num_test_packets} packets:")
    print(f"Baseline Errors: {baseline_errors} | Score: {sum(baseline_rewards):.1f}")
    print(f"AI Errors:       {ai_errors} | Score: {sum(ai_rewards):.1f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(baseline_rewards), label='Baseline Standard ADR', linestyle='--', color='red')
    plt.plot(np.cumsum(ai_rewards), label='AI (Trained DQN)', linewidth=2, color='blue')
    plt.title("Cumulative Throughput: AI vs ADR in 4x4 MIMO Channel")
    plt.xlabel("Packets Transmitted")
    plt.ylabel("Cumulative Throughput Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_models()