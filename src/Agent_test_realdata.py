import numpy as np
from algos import learn, policy
from env import LoopEnv, PDEnv
from utils import sample_demos, prob_dists, expert_data
#from src import twoplayers
from src import TwoPlayersNew
import argparse
import copy
import csv
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from multiprocessing import Process
from sklearn import preprocessing


def get_args():
    parser = argparse.ArgumentParser(description='Bayesian Inverse Reinforcement Learning')
    parser.add_argument('--policy', '-p', choices=('eps', 'bol'))
    parser.add_argument('--alpha', '-a', default=10, type=float, help='1/temperature of boltzmann distribution, '
                                                                      'larger value makes policy close to the greedy')
    parser.add_argument('--env_id', default=0, type=int)
    parser.add_argument('--r_max', default=1, type=float)
    parser.add_argument('--gamma', default=0.75, type=float)
    parser.add_argument('--n_iter', default=3000, type=int)
    parser.add_argument('--burn_in', default=100, type=int)
    parser.add_argument('--dist', default='opposing', type=str, choices=['uniform', 'gaussian', 'beta', 'gamma', 'ising', 'opposing'])
    return parser.parse_args()


def bayesian_irl(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, burn_in, sample_freq):
    assert burn_in <= n_iter
    # "burn_in::sample_freq" means starting at index "burn_in", with steps of length "sample_freq"
    sampled_rewards = np.array(list(policy_walk(**locals()))[burn_in::sample_freq])
    #map_policy_walk(**locals())
    #return 0

    return sampled_rewards


# The Policy Walk algorithm from the paper (adapted Grid Walk)
def policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, **kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    # which is basically random reward picking, should check though why this has to be negative, could it not be made just positive?
    env.rewards = sample_random_rewards(env.n_states, step_size, r_max)
    #print(env.rewards)
    # step 2
    # calculate random policy
    pi = learn.policy_iteration(env, gamma)
    # step 3
    # tqdm just results in the progress bar
    for _ in tqdm(range(n_iter)):
        env_tilda = copy.deepcopy(env)
        # change reward at random index (with value between -step_size to + step_size) (step 3(a))
        env_tilda.rewards = mcmc_reward_step(env.rewards, step_size, r_max)
        #recompute Q_pi (step 3(b))
        q_pi_r_tilda = learn.compute_q_for_pi(env, pi, gamma)
        # step 3(c)
        # policy not optimal anymore for new R_tilda => calculate new best policy for this R_tilda, if you change to R_tilda, your policy will change to this new optimal policy too)
        if is_not_optimal(q_pi_r_tilda, pi):
            pi_tilda = learn.policy_iteration(env_tilda, gamma, pi)
            #probability of changing R (with accompanying new best policy)
            if np.random.random() < compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
                env, pi = env_tilda, pi_tilda
        # policy is still optimal for the new R_tilda, so if we change R to R_tilda, policy can stay the same
        else:
            #probability of changing R
            if np.random.random() < compute_ratio(demos, env_tilda, pi, env, pi, prior, alpha, gamma):
                env = env_tilda
        #print(pi)
        #print(env.rewards)
        yield env.rewards


def map_policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, **kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    # which is basically random reward picking, should check though why this has to be negative, could it not be made just positive?
    env.rewards = sample_random_rewards(env.n_states, step_size, r_max)
    #print(env.rewards)
    # step 2
    # calculate random policy
    pi = learn.policy_iteration(env, gamma)
    # step 3
    # tqdm just results in the progress bar
    current_highest_post = -50
    best_rew = env.rewards
    for _ in tqdm(range(n_iter)):
        env_tilda = copy.deepcopy(env)
        # change reward at random index (with value between -step_size to + step_size) (step 3(a))
        env_tilda.rewards = mcmc_reward_step(env.rewards, step_size, r_max)
        #recompute Q_pi (step 3(b))
        q_pi_r_tilda = learn.compute_q_for_pi(env, pi, gamma)
        # step 3(c)
        # policy not optimal anymore for new R_tilda => calculate new best policy for this R_tilda, if you change to R_tilda, your policy will change to this new optimal policy too)
        if is_not_optimal(q_pi_r_tilda, pi):
            #print("HALLO")
            pi_tilda = learn.policy_iteration(env_tilda, gamma, pi)
            #probability of changing R (with accompanying new best policy)
            #env, pi = env_tilda, pi_tilda
            # policy is still optimal for the new R_tilda, so if we change R to R_tilda, policy can stay the same
            #print(pi)
            check_post = compute_posterior(demos, env_tilda, pi_tilda, prior, alpha, gamma )
            #print(check_post)
            if (check_post > current_highest_post):
                #print("FOUND HIGHER ONE")
                current_highest_post = check_post
                best_rew = env.rewards


        env.rewards = env_tilda.rewards

    env.rewards = best_rew



def is_not_optimal(q_values, pi):
    n_states, n_actions = q_values.shape
    for s in range(n_states):
        for a in range(n_actions):
            if q_values[s, pi[s]] < q_values[s, a]:
                return True
    return False


def compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
    ln_p_tilda = compute_posterior(demos, env_tilda, pi_tilda, prior, alpha, gamma)
    ln_p = compute_posterior(demos, env, pi, prior, alpha, gamma)
    ratio = np.exp(ln_p_tilda - ln_p)
    return ratio


def compute_posterior(demos, env, pi, prior, alpha, gamma):
    q = learn.compute_q_for_pi(env, pi, gamma)
    # q[s,a] gives the q value for state s and action a taken
    # q[s] gives array of all q values for given state (for all possible actions)
    '''for s,a  in demos:
        print(q[s])
        print(q[s, a])
        print(np.log(np.sum(np.exp(alpha * q[s]))))
        print((np.sum(np.exp(alpha * q[s]))))
        print(((np.exp(alpha * q[s]))))
        print((((alpha * q[s]))))
        break'''
    ln_p = np.sum([alpha * q[s, a] - np.log(np.sum(np.exp(alpha * q[s]))) for s, a in demos]) + np.log(prior(env.rewards))
    ln_p_test = np.sum([alpha * q[s, a] - (np.sum((alpha * q[s]))) for s, a in demos]) + np.log(prior(env.rewards))
    #print("POSTERIOR IS " + str(ln_p))
    #print("POSTERIOR_TEST IS " + str(ln_p_test))
    return ln_p


# change something about the rewards (at least 1 rewards has to be changed)
def mcmc_reward_step(rewards, step_size, r_max):
    new_rewards = copy.deepcopy(rewards)
    index = np.random.randint(len(rewards))
    step = np.random.choice([-step_size, step_size])
    new_rewards[index] += step
    new_rewards = np.clip(a=new_rewards, a_min=-r_max, a_max=r_max)
    if np.all(new_rewards == rewards):
        new_rewards[index] -= step
    #assert that at least 1 reward changed somewhere
    assert np.any(rewards != new_rewards), 'rewards do not change: {}, {}'.format(new_rewards, rewards)

    #NEWLY TESTED TO GET ONLY POSITIVE REWARDS!
    #return scale_array(new_rewards)
    return new_rewards

def scale_array(a):
    max_array = np.max(a)
    min_array = np.min(a)

    return (a-min_array)/(max_array - min_array)


def sample_random_rewards(n_states, step_size, r_max):
    """
    sample random rewards form gridpoint(R^{n_states}/step_size).
    :param n_states:
    :param step_size:
    :param r_max:
    :return: sampled rewards
    """
    rewards = np.random.uniform(low=-r_max, high=r_max, size=n_states)
    # move these random rewards toward a gridpoint
    # add r_max to make mod to be always positive
    # add step_size for easier clipping
    rewards = rewards + r_max + step_size
    for i, reward in enumerate(rewards):
        mod = reward % step_size
        rewards[i] = reward - mod
    # subtracts added values from rewards
    rewards = rewards - (r_max + step_size)
    return rewards

# for given distribution and r_max
def prepare_prior(dist, r_max):
    # prob_dists is file in "utils" folder, just makes sure first letter is upper case, adds "Dist" to it and prior becomes a prior function according to implementation
    prior = getattr(prob_dists, dist[0].upper() + dist[1:] + 'Dist')
    if dist == 'uniform':
        # in this case for example you just get uniform for -xmax to xmax
        return prior(xmax=r_max)
    elif dist == 'gaussian':
        return prior()
    elif dist in {'beta', 'gamma'}:
        return prior(loc=-r_max, scale=1/(2 * r_max))
    elif dist == 'ising':
        return prior(coupling = 3, magnetization = 1)
    elif dist == 'opposing':
        return prior()
    else:
        raise NotImplementedError('{} is not implemented.'.format(dist))


#based on current past X observations, calculate policy (and action we would predict at current point based on this)
def calculate_current_q_values(obs, env):
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(env, obs, step_size=0.05, n_iter=args.n_iter, r_max=args.r_max, prior=prior,
                                   alpha=args.alpha, gamma=args.gamma, burn_in=args.burn_in, sample_freq=1)

    #print(sampled_rewards)
    est_rewards = np.mean(sampled_rewards, axis=0)
    env.rewards = est_rewards

    #ADDED FOR MAP INSTEAD OF POLICYWALK
    #np.argmax(ln_p = np.sum([args.alpha * q[s, a] - np.log(np.sum(np.exp(alpha * q[s]))) for s, a in obs]) + np.log(prior(env.rewards)))


    learner_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)

    return learner_q_values

def calc_and_compare_predicted_action(next, q_values, real_action):
    cur_state = next[0]
    #next_action = next[1]
    next_action = real_action

    predicted_action = np.argmax(q_values[cur_state])
    current_q_values = q_values[cur_state]
    print("CURRENT STATE IS " + str(cur_state))
    print("Q_VALUES FOR CUR STATE IS " + str(current_q_values))

    print("NEXT REAL ACTION IS " + str(next_action))
    print("PREDICTED ACTION IS " + str(predicted_action))

    # Added this as return value to start counting wrongs, seperate between 0 or 1 wrong
    # Returning -1 means that real action was 0 while prediction was 1
    # Returning +1 means that real action was 1 while prediction was 0
    # Returning 0 means that the prediction was correct
    return predicted_action, current_q_values, (next_action - predicted_action)

def write_simulation_to_csv(outputfile, data):
    header = ['GroupNumber', 'Period', 'memory length', 'Real action', 'Predicted taken', 'Q0', 'Q1', 'Q-value diff', 'Occurence count', 'Occurence actions', 'Rewards']

    with open(outputfile, 'w', encoding = 'UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def take_highest_immidiate_reward_choice(state, rewards):
    x = '{0:04b}'.format(state)[3]
    # shifts out decisions of expert player
    own_choices = (state >> 2)
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

    choices = []
    if (rewards[real_value_0 + 2 * int(x)] > rewards[real_value_0 + 2*int(x) + 1]):
        choices.append(0)
    else:
        choices.append(1)

    if (rewards[real_value_1 + 2 * int(x)] > rewards[real_value_1 + 2*int(x) + 1]):
        choices.append(0)
    else:
        choices.append(1)
    return choices




def run_simulation(filenumber, groupnumber, treatment, memory_length):
    #TODO: CHECK THIS BELOW
    #IMPORTANT: RANDOM SEED HERE OR NOT??? CURRENTLY YES TO HAVE REPEATABLE RESULTS
    np.random.seed(5)

    #Prepare expert and environment
    Expert = TwoPlayersNew.RealDataAgent(filenumber, groupnumber, treatment)
    env = PDEnv(rewards = Expert.rewards, trans_probs = Expert.trans_probs)

    # sample expert demonstrations
    # LEAVE THIS OUT HERE FOR NOW
    # expert_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    # demos is array of tuples [x,y], where:
    # x is state when choice y was made (so if state is 6 and expert now goes for 0, it will state [6,0]

    demos = np.array(list(expert_data(env, Expert)))
    opponent_actions =Expert.get_opponent_actions().values
    real_actions = Expert.get_actions_real().values

    # To keep track of what kind of errors:
    # Zero wrong means real action was zero while a 1 was predicted (and one wrong vice versa)
    zero_wrong = 0
    one_wrong = 0
    right = 0

    #data = ['GroupNumber', 'Period', 'memory length', 'Real action', 'Predicted taken', 'Q0', 'Q1', 'Q-value diff']
    data = []

    # If you wanna use every single period, rounds_memory and range should add up to 100
    rounds_memory = memory_length
    print("Memory is currently " + str(rounds_memory))
    range_length = 100 - rounds_memory
    for i in range(range_length):
        opp_avg = np.mean(opponent_actions[i + rounds_memory - 1])
        #env.update_trans_probs(Expert.calculate_trans_probs_real(0.8 - (opp_avg*0.6),0.2 + (opp_avg*0.6)))
        env.update_trans_probs(Expert.calculate_trans_probs_real(1 - opp_avg, opp_avg ))
        cur_period = i + 1 + rounds_memory
        print("---------------------------------------")
        print("CURRENT PREDICTION IS FOR PERIOD " + str(cur_period))

        part_demos = demos[i:i + rounds_memory]
        #part_demos = demos[0:i+rounds_memory]

        q_values = calculate_current_q_values(part_demos, env)

        '''
        q_values_new = calculate_current_q_values(part_demos, env)
        #print(q_values_new)
        differences = 0
        for x in range(len(q_values)):
            if (np.argmax(q_values[x]) != np.argmax(q_values_new[x])):
                differences += 1

        print(q_values)
        print(q_values_new)
        print("ARG DIFFERENCES NUMBER IS " + str(differences))
        '''

        next_real_data = demos[i + rounds_memory]
        predicted_action, current_q_values, value = calc_and_compare_predicted_action(next_real_data, q_values, real_actions[i + rounds_memory])

        count = 0
        occurence_actions = []
        for x in range(len(part_demos)):
            if (part_demos[x][0] == next_real_data[0]):
                count = count + 1
                occurence_actions.append(part_demos[x][1])

        print(take_highest_immidiate_reward_choice(next_real_data[0], env.rewards))
        print("NUMBER OF OCCURENCES WAS " + str(count))

        if (value == -1):
            zero_wrong += 1
        elif (value == 1):
            one_wrong += 1
        else:
            right += 1

        print("RIGHT: " + str(right) + "   WRONG: " + str(zero_wrong + one_wrong) + "   ( " + str(
            zero_wrong) + " zeros not predicted and " + str(one_wrong) + " ones not predicted correctly)")
        Q_0 = current_q_values[0]
        Q_1 = current_q_values[1]
        new_data_row = [groupnumber, cur_period, rounds_memory, real_actions[i + rounds_memory], predicted_action, Q_0, Q_1, abs(Q_0 - Q_1), count, occurence_actions, env.rewards]
        data.append(new_data_row)
    print("---------------------------------------")
    file_to_use = "Simulation" + str(filenumber) + "_" + str(groupnumber) + "_" + str(memory_length)  + ".csv"
    write_simulation_to_csv(file_to_use,data)

def run_simulation_full_memory(filenumber, groupnumber, treatment, memory_length):
    #TODO: CHECK THIS BELOW
    #IMPORTANT: RANDOM SEED HERE OR NOT??? CURRENTLY YES TO HAVE REPEATABLE RESULTS
    np.random.seed(10)

    #Prepare expert and environment
    Expert = TwoPlayersNew.RealDataAgent(filenumber, groupnumber, treatment)
    env = PDEnv(rewards = Expert.rewards, trans_probs = Expert.trans_probs)

    # sample expert demonstrations
    # LEAVE THIS OUT HERE FOR NOW
    # expert_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    # demos is array of tuples [x,y], where:
    # x is state when choice y was made (so if state is 6 and expert now goes for 0, it will state [6,0]

    demos = np.array(list(expert_data(env, Expert)))
    opponent_actions =Expert.get_opponent_actions().values
    real_actions = Expert.get_actions_real().values

    # To keep track of what kind of errors:
    # Zero wrong means real action was zero while a 1 was predicted (and one wrong vice versa)
    zero_wrong = 0
    one_wrong = 0
    right = 0

    #data = ['GroupNumber', 'Period', 'memory length', 'Real action', 'Predicted taken', 'Q0', 'Q1', 'Q-value diff']
    data = []

    # If you wanna use every single period, rounds_memory and range should add up to 100
    rounds_memory = memory_length
    print("Memory is currently " + str(rounds_memory))
    range_length = 95
    for i in range(range_length):
        opp_avg = np.mean(opponent_actions[i + rounds_memory - 1])
        env.update_trans_probs(Expert.calculate_trans_probs_real(0.8 - (opp_avg*0.6),0.2 + (opp_avg*0.6)))
        cur_period = i + 1 + rounds_memory
        print("---------------------------------------")
        print("CURRENT PREDICTION IS FOR PERIOD " + str(cur_period))

        part_demos = demos[0:i + rounds_memory]
        #print(str(part_demos))



        next_real_data = demos[i + rounds_memory]
        count = 0

        #FOR ONLY SAME OCCURENCES (TESTING OUT)
        part_demos_occured = []


        for x in range(len(part_demos)):
            if (part_demos[x][0] == next_real_data[0]):
                count = count + 1

                #ALSO FOR OCCURENCE TESTING
                part_demos_occured.append(part_demos[x])
        print(part_demos_occured)

        q_values = None
        #ALSO FOR OCCURENCE TESTING:
        '''
        if (count > 0):
            q_values = calculate_current_q_values(part_demos_occured, env)
        else:
            q_values = calculate_current_q_values(part_demos, env)
        '''
        q_values = calculate_current_q_values(part_demos, env)
        #print(env.rewards)
        predicted_action, current_q_values, value = calc_and_compare_predicted_action(next_real_data, q_values, real_actions[i + rounds_memory])

        print(take_highest_immidiate_reward_choice(next_real_data[0], env.rewards))
        print("NUMBER OF OCCURENCES WAS " + str(count))

        if (value == -1):
            zero_wrong += 1
        elif (value == 1):
            one_wrong += 1
        else:
            right += 1

        print("RIGHT: " + str(right) + "   WRONG: " + str(zero_wrong + one_wrong) + "   ( " + str(
            zero_wrong) + " zeros not predicted and " + str(one_wrong) + " ones not predicted correctly)")
        Q_0 = current_q_values[0]
        Q_1 = current_q_values[1]
        new_data_row = [groupnumber, cur_period, rounds_memory, real_actions[i + rounds_memory], predicted_action, Q_0, Q_1, abs(Q_0 - Q_1)]
        data.append(new_data_row)
    print("---------------------------------------")
    file_to_use = "fullmemory" + str(filenumber) + "_" + str(groupnumber) + "_" + str(memory_length)  + ".csv"
    write_simulation_to_csv(file_to_use,data)


def main(args):
    groups = [5,6,7,8,9,10]
    for x in groups:
        run_simulation(6,x,'Changed', 5)
    '''for x in groups:
        run_simulation(6,x,'Changed', 30)'''
    '''for x in problem_groups:
        run_simulation(6, x, 'Control', 10)
    for x in problem_groups:
        run_simulation(6,x,'Control', 15)
    for x in problem_groups:
        run_simulation(6,x,'Control', 20)
    for x in problem_groups:
        run_simulation(6,x,'Control', 25)'''
    '''for x in problem_groups: 
        run_simulation(6,x,'Control', 30)'''
    '''for x in range(1,16):
        run_simulation(6,x,'Control', 20)
    for x in range(1,16):
        run_simulation(6,x,'Control', 15)
    for x in range(1,16):
        run_simulation(6,x,'Control', 25)'''

    #run_simulation(6,3,'Control', 30)
    #run_simulation(6, 3, 'Control', 30)
    #run_simulation(6, 4, 'Control', 5)
    #run_simulation(6, 4, 'Control', 10)
    #run_simulation(6, 4, 'Control', 30)


    #print(demos)

    # run birl
    # args.dist is the distribution for the priors (given in arguments) and args.r_max is also given in arguments
    # just gives certain "prior function" behavior (in normal case uniform from -r_max to + r_max)
    '''
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(env, demos, step_size=0.05, n_iter=args.n_iter, r_max=args.r_max, prior=prior,
                                   alpha=args.alpha, gamma=args.gamma, burn_in=args.burn_in, sample_freq=1)
    x = sampled_rewards

    # plot rewards
    fig, ax = plt.subplots(1, env.n_states, sharey='all')
    for i, axes in enumerate(ax.flatten()):
        axes.hist(sampled_rewards[:, i], range=(-args.r_max, args.r_max))
    fig.suptitle('Loop Environment {}'.format(args.env_id), )

    #path = '/' + os.path.join(*(os.path.abspath(__file__).split('/')[:-2]), 'results',
    #                          'samples_env{}.png'.format(args.env_id))
    path = "C:\\Users\\Matthias\\Documents\\Thesis\\Bayesian\\bayesian_irl-master\\results"
    plt.savefig(path)

    est_rewards = np.mean(sampled_rewards, axis=0)
    #print('True rewards: ', env_args['rewards'])
    #print('Estimated rewards: ', est_rewards)

    # compute optimal q values for estimated rewards
    env.rewards = est_rewards
    learner_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    #for print_value in ('expert_q_values', 'learner_q_values'):
    #    print(print_value + '\n', locals()[print_value])
    #print('Is a0 optimal action for all states: ', np.all(learner_q_values[:, 0] > learner_q_values[:, 1]))
    print(learner_q_values[:])
'''

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
