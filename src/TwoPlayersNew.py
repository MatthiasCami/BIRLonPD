import numpy as np
from env import PDEnv
from utils import helper
from utils import Excel_Data_Handler


class TwoPlayerAgent:
    def __init__(self):
        # ACTUAL memory is 1 less than this (need to remember 1 extra to get rewards sometimes
        self.memory = 2
        # If you look at the states as "memory of your own actions" vs "memory of opponent actions" (per memory you have CC + CD + DC + DD)
        self.n_states = (4 ** 2)
        # For a Prisoners' dilemma you obviously only have 2 actions
        self.n_actions = 2
        self._trans_probs = self.calculate_trans_probs_real()

    def calculate_rewards(self):
        raise NotImplementedError

    def action(self, s):
        raise NotImplementedError

    # Get the opponent action (in right form to easily work with and generalize/read)
    def opponent_action(self, s):
        # shifts out decisions of expert player
        own_choices = (s >> 2)
        # Adds a 0 to make place for the new action (of the opponent, which will be 0 or 1)
        add_shift_zero = (own_choices << 1)
        # Get action from opponent

        # OPTION 1: RANDOM CHOICE
        opponent_choice = Random.action(s)

        # OPTION 2: ALTERNATE (PASS ALTERNATE AS FIRST ARGUMENT AGAIN TO MAKE IT LIKE IT WOULD BE A STATIC CALL)
        #opponent_choice = Alternate.action(Alternate, s)

        #print("OPPONENT CHOSE ACTION: " + helper.binary_to_choice(opponent_choice))
        # Add the action from the opponent (0 or 1)
        new_value = add_shift_zero + opponent_choice
        # modulo 4 to delete the choice that we will not remember anymore
        new_history = new_value % 4
        # shift 2 0's back again to add the choice of the agent/expert itself (which will be added to the return value where this function is called)
        real_value = new_history << 2

        return real_value

    @property
    def trans_probs(self):
        return self._trans_probs

    # Should probably be getting some checks and asserts
    @trans_probs.setter
    def trans_probs(self, trans_probs):
        self._trans_probs = trans_probs


    def calculate_trans_probs(self):
        trans_probs_cur = np.zeros(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.float32)
        # This only gives own choice, opponent choice for new next state has to be added

        # 0 means cooperate and 1 means defect
        for i in range(self.n_states):
            # If previous action was 0:
            if (i % 2) == 0:
                # You chose action 0 (cooperate) meaning total value will become 0
                trans_probs_cur[i, 0, 0] = 1
                # You chose action 1 (defect) meaning total value will become 1
                trans_probs_cur[i, 1, 1] = 1

            else:
                # You chose action 0 (cooperate) meaning total value will become 2 (your 1 shifts left + 0)
                trans_probs_cur[i, 0, 2] = 1
                # You chose action 1 (defect) meaning total value will become 3 (your 1 shifts left + 1)
                trans_probs_cur[i, 1, 3] = 1
        return trans_probs_cur

    def calculate_trans_probs_real(self, prob_0 = 0.5, prob_1 = 0.5):
        trans_probs_cur = np.zeros(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.float32)

        for i in range(self.n_states):
            x = '{0:04b}'.format(i)[3]
            # shifts out decisions of expert player
            own_choices = (i >> 2)
            # Adds a 0 to make place for the new action (of the opponent, which will be 0 or 1)
            add_shift_zero = (own_choices << 1)

            new_value_0 = add_shift_zero + 0
            new_value_1 = add_shift_zero + 1
            # modulo 4 to delete the choice that we will not remember anymore
            new_history_0 = new_value_0 % 4
            new_history_1 = new_value_1 % 4
            # shift 2 0's back again to add the choice of the agent/expert itself (which will be added to the return value where this function is called)
            real_value_0 = new_history_0 << 2
            real_value_1 = new_history_1 << 2

            #Probability opponent choses 0
            Prob_opp_0 = prob_0
            trans_probs_cur[i, 0, (real_value_0 + int(x)*2) ] = Prob_opp_0
            trans_probs_cur[i, 1, (real_value_0 + int(x) * 2 + 1)] = Prob_opp_0

            #Probability opponent choses 1
            Prob_opp_1 = prob_1
            trans_probs_cur[i, 0, (real_value_1 + int(x) * 2)] = Prob_opp_1
            trans_probs_cur[i, 1, (real_value_1 + int(x) * 2 + 1)] = Prob_opp_1
        #print(trans_probs_cur)
        return(trans_probs_cur)





# Just a random expert (used as opponent for early testing, not that important), only action (so random choice) is performed
class Random(TwoPlayerAgent):

    def calculate_rewards(self):
        pass

    # Static so it can be called with Random.action(s)
    @staticmethod
    def action(s):
        choice = np.random.randint(2)
        return choice


class SmartGuy(TwoPlayerAgent):
    def __init__(self):
        TwoPlayerAgent.__init__(self)
        self._rewards = self.calculate_rewards()

    def action(self, current_state):
        #binary_number = bin(current_state)
        binary_number = '{0:04b}'.format(current_state)
        #print(binary_number)
        if (binary_number[1] == binary_number[3]):
            return 0
        else:
            return 1

    # Property (and getter) for rewards object
    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
        self._rewards = rewards

    def calculate_rewards(self):
        rewards_cur = np.zeros(self.n_states)
        for i in range(self.n_states):
            if (i % 4 == 2):
                rewards_cur[i] = 1
            elif (i % 4 == 1):
                rewards_cur[i] = 1
        return rewards_cur

#REWARDS SHOULD NOT MATTER ANYMORE SINCE THERE ARE SET AT BEGINNING OF CALCULATION RANDOMLY ANYWAYS (SO CHECK IF YOU CAN DELETE THEM WITHOUT CONSEQUENCES)
class RealDataAgent(TwoPlayerAgent):
    def __init__(self, file, group, treatment):
        TwoPlayerAgent.__init__(self)
        self.treatment = treatment
        self.groupnumber = group
        self.filenumber = file
        self._rewards = self.calculate_rewards()
        self.data, self.real_actions = Excel_Data_Handler.get_experiment(self.groupnumber,self.filenumber, self.treatment)
        self.actions_opponent = self.extract_actions()
        self.actions_self = self.extract_actions_real()
        self.actions_opponent_real
        self.counter_self = 0
        self.counter_opponent = 0

    def extract_actions(self):
        keys = list(self.data.keys())
        opponent_actions = self.data[keys[1]]
        return opponent_actions

    def extract_actions_real(self):
        keys = list(self.real_actions.keys())
        own_actions = self.real_actions[keys[0]]
        self.actions_opponent_real = self.real_actions[keys[1]]
        return own_actions

    def action(self, current_state):
        value = self.actions_self.iloc[self.counter_self]
        self.counter_self = self.counter_self + 1
        return value

    def get_actions_real(self):
        return self.actions_self

    def action_opponent(self):
        #print(self.action_self)
        value = self.actions_opponent.iloc[self.counter_opponent]
        self.counter_opponent = self.counter_opponent + 1
        return value

    # Property (and getter) for rewards object
    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
        self._rewards = rewards

    def get_opponent_actions(self):
        return self.actions_opponent

    def calculate_rewards(self):
        rewards_cur = np.zeros(self.n_states)
        for i in range(self.n_states):
            if (i % 4 == 2):
                rewards_cur[i] = 1
            elif (i % 4 == 1):
                rewards_cur[i] = 1
        return rewards_cur

# Create expert that follows a basic alternate pattern (regardless of opponent decisions)
class Alternate(TwoPlayerAgent):
    def __init__(self):
        TwoPlayerAgent.__init__(self)
        self._rewards = self.calculate_rewards()

    def action(self, current_state):
        # If previous action was 0, now go for 1 and vice versa
        if (current_state % 2 == 0):
            return 1
        else:
            return 0

    # Property (and getter) for rewards object
    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
        self._rewards = rewards

    def calculate_rewards(self):
        rewards_cur = np.zeros(self.n_states)
        for i in range(self.n_states):
            if (i % 4 == 2):
                rewards_cur[i] = 1
            elif (i % 4 == 1):
                rewards_cur[i] = 1
        return rewards_cur

