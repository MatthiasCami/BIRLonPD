import numpy as np
import copy


class PDEnv:
    def __init__(self, rewards=None, loop_states=None, trans_probs=None):
        self.n_states = 16
        self.n_actions = 2
        # The "*" just means that it's variable in length
        self.states = np.array([*range(self.n_states)], dtype=int)
        #self.states = np.array([range(self.n_states)], dtype=int)
        self.loop_states = loop_states
        self.trans_probs = trans_probs #if trans_probs else self._get_trans_probs()
        self.state = None
        self._rewards = rewards

    # Property (and getter) for rewards object
    @property
    def rewards(self):
        return self._rewards

    # Setter for the rewards object
    @rewards.setter
    def rewards(self, rewards):
        if isinstance(rewards, list):
            # For every state there has to be a reward defined
            assert len(rewards) == self.n_states, 'Invalid rewards specified'
            rewards = np.array(rewards)
        assert rewards.shape == (self.n_states,), 'Invalid rewards specified'
        self._rewards = rewards

    def update_trans_probs(self, trans_probs_new):
        self.trans_probs = trans_probs_new

    def step(self, a, expert):
        assert 0 <= a < self.n_actions, '{} is invalid action index. ' \
                                        'Action must be in the range of [0, {}]'.format(a, self.n_actions)
        cur_state = copy.deepcopy(self.state)
        print("ACTION TAKEN IS " + str(a))
        print("CURRENT STATE IS " + str(cur_state))
        choice = np.random.choice(self.states, p = self.trans_probs[self.state, a])
        print("CHOICE IS " + str(choice))
        self.state = choice #+ expert.opponent_action(cur_state)
        reward_cur = self._get_reward()
        return self.state, reward_cur

    def next_step(self, action_self, action_opponent):
        assert 0 <= action_self < self.n_actions, '{} is invalid action index. ' \
                                        'Action must be in the range of [0, {}]'.format(action_self, self.n_actions)
        cur_state = copy.deepcopy(self.state)
        #choice = np.random.choice(self.states, p = self.trans_probs[self.state, a])

        self.state = self.update_state(cur_state, action_self, action_opponent)
        reward_cur = self._get_reward()

        return self.state, reward_cur


    #TO CALCULATE NEW STATE BASED ON CURRENT STATE, OWN ACTION AND OPPONENT ACTION (TODO : MOVE THIS FUNCTION TO CORRECT FILE)
    def update_state(self, current_state, action_self, action_opponent):
        # CURRENTLY BASED ON 2-player states with memory 16, thats also why this needs to be moved to make it general with more players
        value_opponent = current_state >> 2
        value_opponent = value_opponent << 1
        #print(action_opponent)
        new_value_opponent = (value_opponent + action_opponent) % 4
        new_value_opponent = new_value_opponent << 2

        value_self = current_state % 4
        value_self = value_self << 1
        new_value_self = (value_self + action_self) % 4

        new_state = (new_value_opponent + new_value_self)
        return new_state

    # Reset to a random state
    def reset(self):
        self.state = np.random.randint(self.n_states)
        return self.state

    # Return rewards for a given state
    def _get_reward(self, state=None):
        assert self.rewards is not None, 'rewards is not specified'
        state = self.state if state is None else state
        return self.rewards[state]


    def _get_trans_probs(self):
        a0 = 0
        a1 = 1
        # The 3 arguments are for the following (assume (a,b,c) here):
        # a is for the amount of "matrices" (n_states, so this is for current state we are in, so we select the trans matrix for this state)
        # b is amount of rows for each matrix (so action taken)
        # c is amount of columns (so chance of going to this state for given row/action taken)
        trans_probs = np.empty(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.float32)
        # Only 1 "s" (or 0), so that's why we need "*" again since we don't know if the length will be 0 or 1 of the argument (will result in 0 integer if there is no s)
        a1_next_state = int(*[s for s in self.states if s not in self.loop_states])
        for state in self.states:
            trans_probs[a1_next_state, a0, state] = 0 if state == a1_next_state else 1/3
        for state, a0_next_state in zip(self.loop_states, self.loop_states[1:] + [self.loop_states[0]]):
            trans_probs[state, 0] = np.eye(self.n_states)[a0_next_state]
        trans_probs[:, a1] = np.eye(self.n_states, dtype=np.float32)[a1_next_state]
        return trans_probs


if __name__ == '__main__':
    ## The expert in this environments loops s1, s2, s3
    trans_probs = np.empty(shape=(4, 2, 4), dtype=np.float32)
    loop_states = [1, 3, 2]
    a1_next_state = [s for s in range(trans_probs.shape[0]) if s not in loop_states][0]
    # print(a1_next_state)
    # Take every row of every "submatrix" (so basically all probabilities for a given action, here action 1)
    # Np.eye(x) gives x*x matrix with 0's everywhere except on diagonal where there are 1's
    # This specific here results in [1,0,0,0] to every "action 1" row
    # This means on action 1 we will always return to state 0
    trans_probs[:, 1] = np.eye(4)[a1_next_state]
    for state in range(trans_probs.shape[0]):
        trans_probs[a1_next_state, 0, state] = 0 if state == a1_next_state else 1/3
    for state, a0_next_state in zip(loop_states, loop_states[1:] + [loop_states[0]]):
        trans_probs[state, 0] = np.eye(4)[a0_next_state]

    env = PDEnv(rewards=[0, 0, 0, 1], loop_states=loop_states)
    obs = env.reset()
    for _ in range(100):
        a = np.random.randint(env.n_actions)
        obs, reward = env.step(a)
        print('obs: {}, action: {}, reward: {}'.format(obs, a, reward))