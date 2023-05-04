import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from minigrid.core.world_object import Goal, Key, Door

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door
control_space = [MF, TL, TR, PK, UD]

NL = 0  # no key, door locked
KL = 1  # get key, door still locked
KO = 2  # get key, door open
NO = 3  # no key, door open
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


def generate_state_space(env, info):
    map_w = info["width"]
    map_h = info["height"]
    goal_coor = np.asarray(info["goal_pos"])
    state_space = []
    goal_state_num = []
    state_array = np.zeros((map_w, map_h, 4, 3)).astype('int')
    # Generate state space
    for i in range(map_w):
        for j in range(map_h):
            for k in DIR:
                for l in range(3):
                    state = {}
                    state['cood'] = np.array([i, j])
                    state['orientation'] = k
                    state['keydoor'] = l
                    if env.grid.get(i, j) == None:
                        state_array[i, j, k, l] = len(state_space)
                        state_space.append(state)
                    elif env.grid.get(i, j).type != 'wall':
                        state_array[i, j, k, l] = len(state_space)
                        state_space.append(state)
                    if np.array_equal(state['cood'], goal_coor):
                        goal_state_num.append(len(state_space) - 1)

    return state_space, goal_state_num, state_array


def generate_cost_matrix(env, info, state_space, state_array):
    door_cood = info['door_pos']
    key_cood = info['key_pos']
    num_state = len(state_space)
    cost_matrix = np.ones((num_state, num_state)) * INF  # State = [x, y, dir, door,keys]
    for i in range(num_state):
        cost_matrix[i, i] = 0
        if np.array_equal(state_space[i]['cood'], door_cood) and \
                state_space[i]['keydoor'] != 2:
            continue
        for act in control_space:
            if act == MF:
                next_cood = state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]

                # print(state_space[i]['cood'])
                if env.grid.get(next_cood[0], next_cood[1]) is not None and \
                        env.grid.get(next_cood[0], next_cood[1]).type == 'wall':
                    continue
                if np.array_equal(state_space[i]['cood'], door_cood) and \
                        state_space[i]['keydoor'] != 2:
                    continue
                next_state_num = state_array[next_cood[0], next_cood[1], state_space[i]['orientation'],
                state_space[i]['keydoor']]
                cost_matrix[i, next_state_num] = 1
            elif act == TL:
                # print(state_space[i]['orientation'])
                next_orientation = (state_space[i]['orientation'] + 1) % 4
                next_state_num = state_array[
                    state_space[i]['cood'][0], state_space[i]['cood'][1], next_orientation, state_space[i]['keydoor']]
                cost_matrix[i, next_state_num] = 1
                # print('=====')
            elif act == TR:
                next_orientation = (state_space[i]['orientation'] - 1) % 4
                # print(state_space[i]['orientation'])
                # print(next_orientation)
                next_state_num = state_array[
                    state_space[i]['cood'][0], state_space[i]['cood'][1], next_orientation, state_space[i]['keydoor']]
                cost_matrix[i, next_state_num] = 1
            elif act == PK:
                if np.array_equal((state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]), key_cood) \
                        and state_space[i]['keydoor'] == NL:
                    # print("get!")
                    # print(state_space[i])
                    next_state_num = state_array[
                        state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], KL]
                    # print(state_space[next_state_num])
                    cost_matrix[i, next_state_num] = 1
            elif act == UD:
                if np.array_equal((state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]), door_cood) \
                        and state_space[i]['keydoor'] == KL:
                    # print("get!")
                    # print(state_space[i])
                    next_state_num = state_array[
                        state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], KO]
                    # print(state_space[next_state_num])
                    cost_matrix[i, next_state_num] = 1
    return cost_matrix


def path2actions(path, state_space):
    actions = []
    for i in range(len(path) - 1):
        # print(i)
        # print(i + 1)
        if state_space[path[i]]['keydoor'] != state_space[path[i + 1]]['keydoor']:
            # print('key')
            if state_space[path[i + 1]]['keydoor'] == 2:
                print('unlock')
                actions.append(UD)
            elif state_space[path[i + 1]]['keydoor'] == 1:
                actions.append(PK)
                print('pick up key')
            else:
                raise Exception("Unknown action")
        elif state_space[path[i]]['orientation'] != state_space[path[i + 1]]['orientation']:
            # print('turn!')
            if (state_space[path[i]]['orientation'] + 1) % 4 == state_space[path[i + 1]]['orientation']:
                actions.append(TL)
                print('turn left!')
            elif (state_space[path[i]]['orientation'] - 1) % 4 == state_space[path[i + 1]]['orientation']:
                actions.append(TR)
                print('turn right!')
            else:
                raise Exception("Unknown action")
        elif not np.array_equal(state_space[path[i]]['cood'], state_space[path[i + 1]]['cood']):
            print('forward!')
            actions.append(MF)
        else:
            raise Exception("Unknown action")
    return actions


def generate_state_space_rd(env, info):
    map_w = info["width"]
    map_h = info["height"]
    goal_coor = np.asarray(info["goal_pos"])
    state_space = []
    goal_state_num = []
    state_array = np.zeros((map_w, map_h, 4, 4, 4)).astype('int')  # x, y, dir, keydoor1, keydoor2
    # Generate state space
    for i in range(map_w):
        for j in range(map_h):
            for k in DIR:
                for m in range(4):
                    for n in range(4):
                        state = {}
                        state['cood'] = np.array([i, j])
                        state['orientation'] = k
                        state['keydoor1'] = m
                        state['keydoor2'] = n
                        if env.grid.get(i, j) == None:
                            state_array[i, j, k, m, n] = len(state_space)
                            state_space.append(state)
                        elif env.grid.get(i, j).type != 'wall':
                            state_array[i, j, k, m, n] = len(state_space)
                            state_space.append(state)
                        if np.array_equal(state['cood'], goal_coor):
                            goal_state_num.append(len(state_space) - 1)

    return state_space, goal_state_num, state_array


def generate_state_space_rd_1(env, info):
    map_w = 8
    map_h = 8
    # goal_coor = np.asarray(info["goal_pos"])
    state_space = []
    goal_state_num = []
    state_array = np.zeros((map_w, map_h, 4, 4, 4, 3, 3)).astype(
        'int')  # x, y, dir, keydoor1, keydoor2, key_pos, goal_pos
    # Generate state space
    for i in range(map_w):
        for j in range(map_h):
            for k in DIR:
                for m in range(4):
                    for n in range(4):
                        for p in range(3):
                            for q in range(3):
                                state = {}
                                state['cood'] = np.array([i, j])
                                state['orientation'] = k
                                state['keydoor1'] = m
                                state['keydoor2'] = n
                                state['key_pos'] = p
                                state['goal_pos'] = q
                                if env.grid.get(i, j) == None:
                                    state_array[i, j, k, m, n, p, q] = len(state_space)
                                    state_space.append(state)
                                elif env.grid.get(i, j).type != 'wall':
                                    state_array[i, j, k, m, n, p, q] = len(state_space)
                                    state_space.append(state)
                                if np.array_equal(state['cood'], goal_pos_vec[0]) and state['goal_pos'] == 0:
                                    goal_state_num.append(len(state_space) - 1)
                                if np.array_equal(state['cood'], goal_pos_vec[1]) and state['goal_pos'] == 1:
                                    goal_state_num.append(len(state_space) - 1)
                                if np.array_equal(state['cood'], goal_pos_vec[2]) and state['goal_pos'] == 2:
                                    goal_state_num.append(len(state_space) - 1)

    return state_space, goal_state_num, state_array


def generate_cost_matrix_rd(env, info, state_space, state_array):
    map_w = info["width"]
    map_h = info["height"]
    door_cood_1 = info['door_pos'][0]
    door_cood_2 = info['door_pos'][1]
    key_cood = info['key_pos']
    if info['door_open'][0]:
        state_keydoor_1 = 3
    else:
        state_keydoor_1 = 0
    if info['door_open'][1]:
        state_keydoor_2 = 3
    else:
        state_keydoor_2 = 0
    num_state = len(state_space)
    cost_matrix = np.ones((num_state, num_state)) * INF  # State = [x, y, dir, door,keys]
    for i in range(num_state):
        cost_matrix[i, i] = 0
        '''
        if (np.array_equal(state_space[i]['cood'], door_cood_1) and \
                state_keydoor_1 < 2) or (np.array_equal(state_space[i]['cood'], door_cood_2) and \
                state_keydoor_2 < 2):
            continue
        
        if np.array_equal(state_space[i]['cood'], door_cood_2) and \
                state_keydoor_2 < 2:
            continue
        '''
        for act in control_space:
            if act == MF:
                next_cood = state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]
                if next_cood[0] > (map_w - 1) or next_cood[0] < 0:
                    continue
                if next_cood[1] > (map_h - 1) or next_cood[1] < 0:
                    continue
                # print(state_space[i]['cood'])
                if env.grid.get(next_cood[0], next_cood[1]) is not None and \
                        env.grid.get(next_cood[0], next_cood[1]).type == 'wall':
                    continue
                if np.array_equal(next_cood, door_cood_1) and \
                        state_space[i]['keydoor1'] < 2:
                    continue
                if np.array_equal(next_cood, door_cood_2) and \
                        state_space[i]['keydoor2'] < 2:
                    continue
                next_state_num = state_array[
                    next_cood[0], next_cood[1], state_space[i]['orientation'], state_space[i]['keydoor1'],
                    state_space[i]['keydoor2']]
                cost_matrix[i, next_state_num] = 1
                '''
                if np.array_equal(next_cood, door_cood_2):
                    print('through door 2!')
                    print(state_space[i])
                    print(state_space[next_state_num])
                '''
            elif act == TL:
                # print(state_space[i]['orientation'])
                next_orientation = (state_space[i]['orientation'] + 1) % 4
                next_state_num = state_array[
                    state_space[i]['cood'][0], state_space[i]['cood'][1], next_orientation, state_space[i]['keydoor1'],
                    state_space[i]['keydoor2']]
                cost_matrix[i, next_state_num] = 1
                # print('=====')
            elif act == TR:
                next_orientation = (state_space[i]['orientation'] - 1) % 4
                # print(state_space[i]['orientation'])
                # print(next_orientation)
                next_state_num = state_array[
                    state_space[i]['cood'][0], state_space[i]['cood'][1], next_orientation, state_space[i]['keydoor1'],
                    state_space[i]['keydoor2']]
                cost_matrix[i, next_state_num] = 1
            elif act == PK:
                if np.array_equal((state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]), key_cood):
                    if state_space[i]['keydoor1'] == 3 and state_space[i]['keydoor2'] == 3:
                        next_state_num = state_array[
                            state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], 2, 2]

                        cost_matrix[i, next_state_num] = 1
                    if state_space[i]['keydoor1'] == 3 and state_space[i]['keydoor2'] == 0:
                        next_state_num = state_array[
                            state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], 2, 1]
                        cost_matrix[i, next_state_num] = 1
                    if state_space[i]['keydoor1'] == 0 and state_space[i]['keydoor2'] == 3:
                        next_state_num = state_array[
                            state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], 1, 2]
                        cost_matrix[i, next_state_num] = 1
                    if state_space[i]['keydoor1'] == 0 and state_space[i]['keydoor2'] == 0:
                        next_state_num = state_array[
                            state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], 1, 1]
                        cost_matrix[i, next_state_num] = 1
            elif act == UD:
                if np.array_equal((state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]), door_cood_1) \
                        and state_space[i]['keydoor1'] == KL:
                    # print('Unlock')
                    # print(state_space[i])
                    next_state_num = state_array[
                        state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'],
                        2, state_space[i]['keydoor2']]
                    cost_matrix[i, next_state_num] = 1
                    # print(state_space[next_state_num])
                if np.array_equal((state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]), door_cood_2) \
                        and state_space[i]['keydoor2'] == KL:
                    # print('Unlock')
                    # print(state_space[i])
                    next_state_num = state_array[
                        state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'],
                        state_space[i]['keydoor1'], 2]
                    cost_matrix[i, next_state_num] = 1
                    # print(state_space[next_state_num])
    return cost_matrix


def generate_cost_matrix_rd_1(env, info, state_space, state_array):
    map_w = 8
    map_h = 8
    door_cood_1 = info['door_pos'][0]
    door_cood_2 = info['door_pos'][1]
    num_state = len(state_space)
    cost_matrix = np.ones((num_state, num_state)) * INF  # State = [x, y, dir, keydoor1, keydoor2, key_pos, goal_pos]
    for i in range(num_state):
        cost_matrix[i, i] = 0
        for act in control_space:
            if act == MF:
                next_cood = state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]
                if next_cood[0] > (map_w - 1) or next_cood[0] < 0:
                    continue
                if next_cood[1] > (map_h - 1) or next_cood[1] < 0:
                    continue
                # print(state_space[i]['cood'])
                if env.grid.get(next_cood[0], next_cood[1]) is not None and \
                        env.grid.get(next_cood[0], next_cood[1]).type == 'wall':
                    continue
                if np.array_equal(next_cood, door_cood_1) and \
                        state_space[i]['keydoor1'] < 2:
                    continue
                if np.array_equal(next_cood, door_cood_2) and \
                        state_space[i]['keydoor2'] < 2:
                    continue
                next_state_num = state_array[
                    next_cood[0], next_cood[1], state_space[i]['orientation'], state_space[i]['keydoor1'],
                    state_space[i]['keydoor2'], state_space[i]['key_pos'], state_space[i]['goal_pos']]
                cost_matrix[i, next_state_num] = 1
            elif act == TL:
                # print(state_space[i]['orientation'])
                next_orientation = (state_space[i]['orientation'] + 1) % 4
                next_state_num = state_array[
                    state_space[i]['cood'][0], state_space[i]['cood'][1], next_orientation, state_space[i]['keydoor1'],
                    state_space[i]['keydoor2'], state_space[i]['key_pos'], state_space[i]['goal_pos']]
                cost_matrix[i, next_state_num] = 1
                # print('=====')
            elif act == TR:
                next_orientation = (state_space[i]['orientation'] - 1) % 4
                # print(state_space[i]['orientation'])
                # print(next_orientation)
                next_state_num = state_array[
                    state_space[i]['cood'][0], state_space[i]['cood'][1], next_orientation, state_space[i]['keydoor1'],
                    state_space[i]['keydoor2'], state_space[i]['key_pos'], state_space[i]['goal_pos']]
                cost_matrix[i, next_state_num] = 1
            elif act == PK:
                if np.array_equal((state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]),
                                  key_pos_vec[state_space[i]['key_pos']]):
                    if state_space[i]['keydoor1'] == 3 and state_space[i]['keydoor2'] == 3:
                        next_state_num = state_array[
                            state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], 2, 2,
                            state_space[i]['key_pos'], state_space[i]['goal_pos']]

                        cost_matrix[i, next_state_num] = 1
                    if state_space[i]['keydoor1'] == 3 and state_space[i]['keydoor2'] == 0:
                        next_state_num = state_array[
                            state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], 2, 1,
                            state_space[i]['key_pos'], state_space[i]['goal_pos']]
                        cost_matrix[i, next_state_num] = 1
                    if state_space[i]['keydoor1'] == 0 and state_space[i]['keydoor2'] == 3:
                        next_state_num = state_array[
                            state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], 1, 2,
                            state_space[i]['key_pos'], state_space[i]['goal_pos']]
                        cost_matrix[i, next_state_num] = 1
                    if state_space[i]['keydoor1'] == 0 and state_space[i]['keydoor2'] == 0:
                        next_state_num = state_array[
                            state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'], 1, 1,
                            state_space[i]['key_pos'], state_space[i]['goal_pos']]
                        cost_matrix[i, next_state_num] = 1
            elif act == UD:
                if np.array_equal((state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]), door_cood_1) \
                        and state_space[i]['keydoor1'] == KL:
                    # print('Unlock')
                    # print(state_space[i])
                    next_state_num = state_array[
                        state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'],
                        2, state_space[i]['keydoor2'], state_space[i]['key_pos'], state_space[i]['goal_pos']]
                    cost_matrix[i, next_state_num] = 1
                    # print(state_space[next_state_num])
                if np.array_equal((state_space[i]['cood'] + DIR_vec[state_space[i]['orientation']]), door_cood_2) \
                        and state_space[i]['keydoor2'] == KL:
                    # print('Unlock')
                    # print(state_space[i])
                    next_state_num = state_array[
                        state_space[i]['cood'][0], state_space[i]['cood'][1], state_space[i]['orientation'],
                        state_space[i]['keydoor1'], 2, state_space[i]['key_pos'], state_space[i]['goal_pos']]
                    cost_matrix[i, next_state_num] = 1
                    # print(state_space[next_state_num])

    return cost_matrix


def path2actions_rd(path, state_space):
    actions = []
    for i in range(len(path) - 1):
        # print(i)
        # print(i + 1)
        if state_space[path[i]]['keydoor1'] != state_space[path[i + 1]]['keydoor1'] \
                or state_space[path[i]]['keydoor2'] != state_space[path[i + 1]]['keydoor2']:
            # print('key')
            if state_space[path[i + 1]]['keydoor1'] == 1 and state_space[path[i]]['keydoor1'] == 0:
                print('pick up key')
                actions.append(PK)
            elif state_space[path[i + 1]]['keydoor2'] == 1 and state_space[path[i]]['keydoor2'] == 0:
                actions.append(PK)
                print('pick up key')
            else:
                actions.append(UD)
                print('unlock the door')
        elif state_space[path[i]]['orientation'] != state_space[path[i + 1]]['orientation']:
            # print('turn!')
            if (state_space[path[i]]['orientation'] + 1) % 4 == state_space[path[i + 1]]['orientation']:
                actions.append(TL)
                print('turn left')
            elif (state_space[path[i]]['orientation'] - 1) % 4 == state_space[path[i + 1]]['orientation']:
                actions.append(TR)
                print('turn right')
            else:
                raise Exception("Unknown action")
        elif not np.array_equal(state_space[path[i]]['cood'], state_space[path[i + 1]]['cood']):
            print('forward')
            actions.append(MF)
        else:
            raise Exception("Unknown action")
    return actions


def path2actions_rd_1(path, state_space):
    actions = []
    for i in range(len(path) - 1):
        # print(i)
        # print(i + 1)
        if state_space[path[i]]['keydoor1'] != state_space[path[i + 1]]['keydoor1'] \
                or state_space[path[i]]['keydoor2'] != state_space[path[i + 1]]['keydoor2']:
            # print('key')
            if state_space[path[i + 1]]['keydoor1'] == 1 and state_space[path[i]]['keydoor1'] == 0:
                print('pick up key')
                actions.append(PK)
            elif state_space[path[i + 1]]['keydoor2'] == 1 and state_space[path[i]]['keydoor2'] == 0:
                actions.append(PK)
                print('pick up key')
            else:
                actions.append(UD)
                print('unlock the door')
        elif state_space[path[i]]['orientation'] != state_space[path[i + 1]]['orientation']:
            # print('turn!')
            if (state_space[path[i]]['orientation'] + 1) % 4 == state_space[path[i + 1]]['orientation']:
                actions.append(TL)
                print('turn left')
            elif (state_space[path[i]]['orientation'] - 1) % 4 == state_space[path[i + 1]]['orientation']:
                actions.append(TR)
                print('turn right')
            else:
                raise Exception("Unknown action")
        elif not np.array_equal(state_space[path[i]]['cood'], state_space[path[i + 1]]['cood']):
            print('forward')
            actions.append(MF)
        else:
            raise Exception("Unknown action")
    return actions


def step_cost(state_current, state_next, action, env):
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************

    '''if action == MF:
        if env.agent_pos'''

    return 0  # the cost of action


def step(env, action):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        0: env.actions.forward,
        1: env.actions.left,
        2: env.actions.right,
        3: env.actions.pickup,
        4: env.actions.toggle,
    }

    (obs, reward, terminated, truncated, info) = env.step(actions[action])
    return step_cost(0, 0, action, env), terminated


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    with open(path, "rb") as f:
        env = pickle.load(f)

    info = {"height": env.height, "width": env.width, "init_agent_pos": env.agent_pos, "init_agent_dir": env.dir_vec}

    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info


def load_random_env(env_folder):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
    # print(env_list)
    env_path = random.choice(env_list)
    # env_path = "./envs/random_envs/DoorKey-8x8-35.env"
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.height,
        "width": env.width,
        "init_agent_pos": env.agent_pos,
        "init_agent_dir": env.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info, env_path


def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq, env, path="./gif/doorkey.gif"):
    # with imageio.get_writer(path, mode="I", duration=0.8) as writer:
    img = env.render()
    frames = []
    frames.append(img)
    for act in seq:
        step(env, act)
        img = env.render()
        frames.append(img)
        # print(env.agent_pos)
        # writer.append_data(img)
    imageio.mimsave(path, frames, duration=1)
    print(f"GIF is written to {path}")
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    return


if __name__ == '__main__':
    env_path = "./envs/known_envs/doorkey-6x6-normal.env"
    env, info = load_env(env_path)  # load an environment

    state_space, goal_state_num, state_array = generate_state_space(env, info)
    cost_matix = generate_cost_matrix(env, info, state_space, state_array)
    # print(cost_matix)
    # motion_model(state_space[1], 0, info['door_pos'], info['key_pos'])
