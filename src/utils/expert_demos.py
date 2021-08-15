from utils import helper
def expert_demos(env, expert, n_episode=10, epi_length=10):
    obs = env.reset()
    for _ in range(n_episode):
        for i in range(epi_length):
            action = expert.action(obs)
            #print("EXPERT CHOSE ACTION: " + helper.binary_to_choice(action))
            yield [obs, action]
            obs, _ = env.step(action, expert)
        obs = env.reset()
