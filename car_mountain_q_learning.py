import gymnasium as gym  
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sn
def get_discrete_state(state,window_size):
    state_index=(state-env.observation_space.low)/window_size
    return tuple(state_index.astype(int))

env = gym.make('MountainCar-v0')
env.reset()
learning_rate=0.1
discount_factor=0.95
epsilon=1.0
epsilon_decay=0.9995
epsilon_min=0.01
reward_list=[]
table_size=(20,20)
window_size=(env.observation_space.high-env.observation_space.low)/table_size
interval=2000
total_reward=0
q_table=np.random.uniform(low=0,high=1,size=(table_size[0],table_size[1],env.action_space.n))
for episode in range(60000):
    step=0
    render=False
    state=env.reset()[0]
    discrete_state=get_discrete_state(state, window_size)
    done=False
    total_reward=0
    while not done and step<120:
        if random.uniform(0, 1)<epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[discrete_state])
        new_state,reward,done,_,_ = env.step(action)
        total_reward+=reward
        new_discrete_state=get_discrete_state(new_state,window_size)
        if not done:
            best_future_q=np.max(q_table[new_discrete_state])
            current_q=q_table[discrete_state][action]
            new_q=(1-learning_rate)*current_q+learning_rate*((reward+discount_factor*best_future_q))
            q_table[discrete_state][action]=new_q
        elif new_state[0] >=env.unwrapped.goal_position:
            q_table[discrete_state][action]=0
            done=True
            print(f'Reached goal in episode {episode} at step {step}')
        step+=1
        discrete_state=new_discrete_state
    reward_list.append(total_reward)
    epsilon=max(epsilon_min,epsilon*epsilon_decay)
env.close()
sn.lineplot(reward_list)
plt.title('Reward per episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
eval_env=gym.make('MountainCar-v0',render_mode='human')
state = eval_env.reset()[0]
discrete_state = get_discrete_state(state, window_size)
done = False
while not done:
    action = np.argmax(q_table[discrete_state])
    new_state, reward, done, _, _ = eval_env.step(action)
    eval_env.render()  # now UI shows
    discrete_state = get_discrete_state(new_state, window_size)
eval_env.close()