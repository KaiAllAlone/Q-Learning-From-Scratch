import gymnasium as gym  
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sn
env = gym.make('CliffWalking-v0')
env.reset()
learning_rate=0.1
discount_factor=0.95
epsilon=1.0
epsilon_decay=0.9995
epsilon_min=0.01
reward_list=[]
#observation space contains 48 values
#action space contains 4 values
#observation consists of only a single value
q_table=np.random.uniform(low=0,high=1,size=(48,4))
for episode in range(10000):
    step=0
    render=False
    state=env.reset()[0]
    done=False
    total_reward=0
    while not done and step<50:
        if random.uniform(0, 1)<epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        new_state,reward,done,_,_ = env.step(action)
        total_reward+=reward
        if not done:
            best_future_q=np.max(q_table[new_state])
            current_q=q_table[state][action]
            new_q=(1-learning_rate)*current_q+learning_rate*((reward+discount_factor*best_future_q))
            q_table[state][action]=new_q
        elif new_state==37:
            q_table[state][action]=0
            done=True
            print(f'Reached goal in episode {episode} at step {step}')
        step+=1
        state=new_state
    reward_list.append(total_reward)
    epsilon=max(epsilon_min,epsilon*epsilon_decay)
env.close()
sn.lineplot(reward_list)
plt.title('Reward per episode')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()
eval_env=gym.make('CliffWalking-v0',render_mode='human')
state = eval_env.reset()[0]
done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _, _ = eval_env.step(action)
    eval_env.render()
eval_env.close()