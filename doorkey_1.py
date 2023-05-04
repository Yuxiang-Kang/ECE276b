import numpy as np
from utils import *
import minigrid

# from example import example_use_of_gym_env

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

NL = 0  # no key, door locked
KL = 1  # get key, door still locked
KO = 2  # get key, door open
NO = 3  # no key, but door open
INF = 99999
# define directions
DIR = [0, 1, 2, 3]  # up, left, down, right,
DIR_vec = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]),
           np.array([1, 0])]  # up, left, down, right,
# random env key pos
key_pos = [0, 1, 2]
key_pos_vec = [np.array([1, 1]), np.array([2, 3]), np.array([1, 6])]

# random env goal pos
goal_pos = [0, 1, 2]
goal_pos_vec = [np.array([5, 1]), np.array([6, 3]), np.array([5, 6])]


def doorkey_problem(env, info):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """
    # Define state
    map_w = info["width"]
    map_h = info["height"]

    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq


def partA():
    env_path = "./envs/known_envs/doorkey-8x8-normal.env" # enter the considered environment file name
    env, info = load_env(env_path)  # load an environment
    state_space, goal_state_num, state_array = generate_state_space(env, info)
    print(info)
    # get start state number
    for i in range(len(state_space)):
        if np.array_equal(state_space[i]["cood"], info['init_agent_pos']) \
                and np.array_equal(DIR_vec[state_space[i]["orientation"]], info['init_agent_dir']) \
                and state_space[i]["keydoor"] == 0:
            print('get start point!')
            start_state_num = i
            # print(state_space[i])
            # print(i)
            break

    cost_matrix = generate_cost_matrix(env, info, state_space, state_array)
    T = len(state_space) - 1

    # 1) initialize value function and policy
    V = np.ones((len(state_space), T)) * INF
    for i in goal_state_num:
        V[i, :] = 0
    policy = []

    # 2）iterate backwards from end
    t = T - 2
    while t > -2:
        Q = np.zeros((len(state_space), len(state_space)))
        # print(V[:, t + 1])
        Q = cost_matrix + V[:, t + 1]  # Q(i,j) from V[j,t+1]
        # print(Q)
        V[:, t] = np.amin(Q, axis=1)  # j of minimum Q for each i, get V[i,t]
        policy.append(np.argmin(Q, axis=1))  # record best j for each i at t
        if np.array_equal(V[:, t], V[:, t + 1]):
            print('early termination!')
            break  # early termination
        t = t - 1
    print(np.asarray(policy))
    # print(np.asarray(policy).shape)
    # print(state_space[348])

    # 3) pick policy for given start state
    policy_arr = np.asarray(policy)
    path_state = []
    path_state.append(start_state_num)
    cur_state_num = start_state_num
    i = policy_arr.shape[0] - 1
    while i > -1:
        j = cur_state_num
        cur_state_num = policy_arr[i][cur_state_num]
        # print(cur_state_num)
        i = i - 1

        if np.array_equal(state_space[cur_state_num]["cood"], state_space[j]["cood"]) \
                and np.array_equal(state_space[cur_state_num]["orientation"], state_space[j]["orientation"]) \
                and state_space[cur_state_num]["keydoor"] == state_space[j]["keydoor"]:
            # print('stay!')
            continue
        path_state.append(cur_state_num)
        # print(cur_state_num)

    # print(path_state)
    print(len(path_state))
    print(path_state)
    for i in path_state:
        print(state_space[i])
    print('<======================>')
    action_seq = path2actions(path_state, state_space)
    print(action_seq)
    # print(policy_arr)
    # print(policy_arr.shape)
    # print(len(policy_arr))
    print('<======================>')
    draw_gif_from_seq(action_seq, env)  # draw a GIF & save


def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)
    print('env path: ', env_path)
    print(info)
    state_space, goal_state_num, state_array = generate_state_space_rd_1(env, info)
    # get start state
    if info['door_open'][0]:
        state_keydoor_1 = 3
    else:
        state_keydoor_1 = 0
    if info['door_open'][1]:
        state_keydoor_2 = 3
    else:
        state_keydoor_2 = 0
    for i in goal_pos:
        if np.array_equal(info['goal_pos'], goal_pos_vec[i]):
            state_goal_pos = i
    for i in key_pos:
        if np.array_equal(info['key_pos'], key_pos_vec[i]):
            state_key_pos = i
    for i in range(len(state_space)):
        if np.array_equal(state_space[i]["cood"], info['init_agent_pos']) \
                and np.array_equal(DIR_vec[state_space[i]["orientation"]], info['init_agent_dir']) \
                and state_space[i]["keydoor1"] == state_keydoor_1 \
                and state_space[i]["keydoor2"] == state_keydoor_2 \
                and state_space[i]["goal_pos"] == state_goal_pos \
                and state_space[i]["key_pos"] == state_key_pos:
            start_state_num = i
            print('Found the start point.')
            print(state_space[i])
            # print(i)
            break
    cost_matrix = generate_cost_matrix_rd_1(env, info, state_space, state_array)
    print("cost matrix generated.")
    T = len(state_space) - 1
    # 1) initialize value function and policy
    V = np.ones((len(state_space), T)) * INF
    for i in goal_state_num:
        V[i, :] = 0
    policy = []
    # 2）iterate backwards from end
    t = T - 2
    i = 0
    while t > -2:
        Q = np.zeros((len(state_space), len(state_space)))
        # print(V[:, t + 1])
        Q = cost_matrix + V[:, t + 1]  # Q(i,j) from V[j,t+1]
        # print(Q)
        V[:, t] = np.amin(Q, axis=1)  # j of minimum Q for each i, get V[i,t]
        policy.append(np.argmin(Q, axis=1))  # record best j for each i at t
        if np.array_equal(V[:, t], V[:, t + 1]):
            print('Policy reaches an early termination!')
            break  # early termination
        print('iteration:', i)
        i = i + 1
        t = t - 1
    # print(np.asarray(policy))
    # print(np.asarray(policy).shape)
    # 3) pick policy for given start state
    policy_arr = np.asarray(policy)
    path_state = []
    path_state.append(start_state_num)
    cur_state_num = start_state_num
    i = policy_arr.shape[0] - 1
    while i > -1:
        j = cur_state_num

        cur_state_num = policy_arr[i][cur_state_num]
        # print(cur_state_num)
        print(state_space[cur_state_num])
        i = i - 1

        if np.array_equal(state_space[cur_state_num]["cood"], state_space[j]["cood"]) \
                and np.array_equal(state_space[cur_state_num]["orientation"], state_space[j]["orientation"]) \
                and state_space[cur_state_num]["keydoor1"] == state_space[j]["keydoor1"] \
                and state_space[cur_state_num]["keydoor2"] == state_space[j]["keydoor2"] \
                and state_space[cur_state_num]["key_pos"] == state_space[j]["key_pos"] \
                and state_space[cur_state_num]["goal_pos"] == state_space[j]["goal_pos"]:
            # print('stay!')
            continue
        path_state.append(cur_state_num)
        # print(cur_state_num)
    print("Creating actions:")
    action_seq = path2actions_rd_1(path_state, state_space)
    print('<======================>')
    print(action_seq)
    draw_gif_from_seq(action_seq, env)  # draw a GIF & save


if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    # partB()
