def sample_demos(env, policy, n_episode=20, epi_length=10):
    obs = env.reset()
    print("BEGIN")
    for _ in range(n_episode):
        print("EPISODE NUMBER")
        print(_)
        for i in range(epi_length):
            print("EPISODE INDEX")
            print(i)
            action = policy(obs)
            print("ACTION TAKEN")
            print(action)
            yield [obs, action]
            obs, _ = env.step(action)
        obs = env.reset()
