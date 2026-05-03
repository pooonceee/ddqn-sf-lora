# train_ai.py
from environment.wireless_env import LoRaEnv
from agents.dqn_agent import DQNAgent

def train():
    env = LoRaEnv()
    agent = DQNAgent() # Automatically uses new 33/1296 sizes
    
    episodes = 300 # Increased to allow learning the larger action space
    steps_per_episode = 20
    
    print("--- Commencing 4x4 Deep Q-Network Training ---")
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(steps_per_episode):
            action_idx = agent.choose_action(state)
            action_dict = agent.get_action_format(action_idx)
            next_state, reward, info = env.step(action_dict)
            agent.remember(state, action_idx, reward, next_state, done=False)
                        # Around line 21
            # Change this:
            # agent.replay()

            # To this:
            batch_size = 32 # Common standard for DQN
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            state = next_state
            total_reward += reward
            
        print(f"Episode {e+1:3}/{episodes} | Total Reward: {total_reward:7.1f} | Epsilon: {agent.epsilon*100:5.1f}%")

    # SAVE THE BRAIN!
    agent.save("lora_mimo_4x4.pth")
    print("Training Complete. Model weights saved to lora_mimo_4x4.pth")

if __name__ == "__main__":
    train()