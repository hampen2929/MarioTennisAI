

def reward_normalize(score_difference):
    if score_difference > 0:
        reward = 1
    elif score_difference < 0:
        reward = -1
    else:
        reward = 0

    return reward

