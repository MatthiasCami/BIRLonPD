import numpy as np
from algos import learn, policy
from env import LoopEnv, PDEnv
from utils import sample_demos, prob_dists, expert_demos
from src import twoplayers
import argparse
import copy
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Bayesian Inverse Reinforcement Learning')
    parser.add_argument('--policy', '-p', choices=('eps', 'bol'))
    parser.add_argument('--alpha', '-a', default=10, type=float, help='1/temperature of boltzmann distribution, '
                                                                      'larger value makes policy close to the greedy')
    parser.add_argument('--env_id', default=0, type=int)
    parser.add_argument('--r_max', default=1, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--n_iter', default=1000, type=int)
    parser.add_argument('--burn_in', default=100, type=int)
    parser.add_argument('--dist', default='uniform', type=str, choices=['uniform', 'gaussian', 'beta', 'gamma'])
    return parser.parse_args()


def bayesian_irl(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, burn_in, sample_freq):
    assert burn_in <= n_iter
    # "burn_in::sample_freq" means starting at index "burn_in", with steps of length "sample_freq"
    sampled_rewards = np.array(list(policy_walk(**locals()))[burn_in::sample_freq])
    return sampled_rewards


# The Policy Walk algorithm from the paper (adapted Grid Walk)
def policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, **kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    # which is basically random reward picking, should check though why this has to be negative, could it not be made just positive?
    env.rewards = sample_random_rewards(env.n_states, step_size, r_max)
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
        yield env.rewards


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
    ln_p = np.sum([alpha * q[s, a] - np.log(np.sum(np.exp(alpha * q[s]))) for s, a in demos]) + np.log(prior(env.rewards))
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
    return new_rewards


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
    else:
        raise NotImplementedError('{} is not implemented.'.format(dist))

def main(args):
    np.random.seed(20)

    # prepare environments
    #Expert = twoplayers.Alternate()
    Expert = twoplayers.Copyer()
    env = PDEnv(rewards = Expert.rewards, trans_probs = Expert.trans_probs)

    # sample expert demonstrations
    expert_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    # demos is array of tuples [x,y], where:
    # x is state when choice y was made (so if state is 6 and expert now goes for 0, it will state [6,0]

    demos = np.array(list(expert_demos(env, Expert)))
    print(demos)


    # run birl
    # args.dist is the distribution for the priors (given in arguments) and args.r_max is also given in arguments
    # just gives certain "prior function" behavior (in normal case uniform from -r_max to + r_max)
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(env, demos, step_size=0.05, n_iter=args.n_iter, r_max=args.r_max, prior=prior,
                                   alpha=args.alpha, gamma=args.gamma, burn_in=args.burn_in, sample_freq=1)

    # plot rewards

    est_rewards = np.mean(sampled_rewards, axis=0)
    #print('True rewards: ', env_args['rewards'])
    #print('Estimated rewards: ', est_rewards)

    # compute optimal q values for estimated rewards
    env.rewards = est_rewards
    learner_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    #print('Is a0 optimal action for all states: ', np.all(learner_q_values[:, 0] > learner_q_values[:, 1]))
    print(learner_q_values)
    real_learner_q = learner_q_values

    # run birl
    # args.dist is the distribution for the priors (given in arguments) and args.r_max is also given in arguments
    # just gives certain "prior function" behavior (in normal case uniform from -r_max to + r_max)
    acc_array = []
    for i in range(50):
        Expert2 = twoplayers.Copyer()
        env2 = PDEnv(rewards=Expert2.rewards, trans_probs=Expert2.trans_probs)
        prior = prepare_prior(args.dist, args.r_max)
        #print(demos[0:i+1])
        sampled_rewards = bayesian_irl(env2, demos[0:i+1], step_size=0.05, n_iter=args.n_iter, r_max=args.r_max, prior=prior,
                                       alpha=args.alpha, gamma=args.gamma, burn_in=args.burn_in, sample_freq=1)

        # plot rewards

        est_rewards = np.mean(sampled_rewards, axis=0)
        # print('True rewards: ', env_args['rewards'])
        # print('Estimated rewards: ', est_rewards)

        # compute optimal q values for estimated rewards
        env.rewards = est_rewards
        learner_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
        # print('Is a0 optimal action for all states: ', np.all(learner_q_values[:, 0] > learner_q_values[:, 1]))
        temp = learner_q_values
        print(temp)

        wrong = 0
        right = 0
        for k in range(len(temp)):
            if (np.argmax(real_learner_q[k]) == np.argmax(temp[k])):
                right +=1
            else:
                wrong +=1
        print(i)
        print("ACCURACY IS " + str((right)/(wrong + right)) + "   " + str(right) + "   " + str(wrong))
        acc_array.append((right)/(wrong + right))
    print(acc_array)

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
