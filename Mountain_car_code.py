import gym
import numpy as np
env = gym.make("MountainCar-v0")
#env.reset()

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / (DISCRETE_OS_SIZE)

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#print(discrete_os_win_size)

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n])) #THIS WILL CREATE A Q TABLE OF EVERY POSSIBLE COMBINATION OF POS AND VEL IN 3 D FOR 3 ACTIONS

#print(q_table.shape)

def get_discrete(state):
  discrete_state = (state - env.observation_space.low) / discrete_os_win_size
  return tuple(discrete_state.astype(np.int))

#print(discrete_state)          #THIS WILL TELL US THE COORDINATTES OF INITIAL STATE IN Q TABLE
#print(np.argmax(q_table[discrete_state])) #THIS WILL TELL US THE Q VALUES FOR THREE ACTIONS AT THAT SPECIFIC COORDINATE STATE AND NP.ARGMAX WILL FIND MAXIMUM VALUE BETWEEN THESE THREE Q VALUES



for episodes in range(EPISODES):

  discrete_state = get_discrete(env.reset())
  done = False

  if episodes % SHOW_EVERY == 0:
    print(episodes)
    render = True
  else:
    render = False

  
  while not done:
    if np.random.random() > epsilon:
      action = np.argmax(q_table[discrete_state])
    else:
      action = np.random.randint(0, env.action_space.n)

    new_state,reward,done, _ = env.step(action)
    new_discrete_state = get_discrete(new_state)
    if render:
      env.render()
    if not done:
      max_future_q = np.max(q_table[new_discrete_state]) 
      current_q = q_table[discrete_state + (action, )] #THIS WILL YIELD THE Q VALUE OF THAT SPECIFIC ACTION AT GIVEN STATE
      new_q = (1 - LEARNING_RATE ) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
      q_table[discrete_state + (action, )] = new_q
    elif new_state[0] >= env.goal_position:
      print(f"We made it on episode {episodes}!")
      q_table[discrete_state + (action, )] = 0

    discrete_state = new_discrete_state

  if END_EPSILON_DECAYING >= episodes > START_EPSILON_DECAYING:
    epsilon -= epsilon_decay_value


    


env.close()