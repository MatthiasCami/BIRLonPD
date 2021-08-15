def expert_data(env, expert):
    obs = env.reset()
    for x in range(100):
        action = expert.action(obs)
        #print("ACTION CHOSEN IS " + str(action) + "     AT PERIOD" + str(x+1))
        #print("EXPERT CHOSE ACTION: " + helper.binary_to_choice(action))
        yield [obs, action]
        obs, _ = env.next_step(action, expert.action_opponent())

