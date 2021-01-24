import os
import numpy as np
import operator
from GridWorld import GridWorld
from Q_Agent import Q_Agent
import argparse
import matplotlib.pyplot as plt
import utils
import json

from hyperopt import hp, tpe, fmin, Trials, rand
# we import tpe algorithm 
# fmin function which helps us minimize the equation
# hp which creates the search space

import pandas as pd
save = False
save_path = ''


def objectiveFunction(args):

    learning_rate, min_epsilon, max_epsilon, epsilon_decay, discount_factor = args

    num_of_episodes=500
    max_steps=1000

    environment = GridWorld()

    agentQ = Q_Agent(environment, epsilon=max_epsilon, learning_rate=learning_rate, discount_factor=discount_factor)

    train(environment, agentQ, episodes=num_of_episodes, max_steps_per_episode=max_steps, min_epsilon=min_epsilon, max_epsilon=max_epsilon, epsilon_decay=epsilon_decay)
    mean_reward = test(environment, agentQ, episodes=1000)

    value_map = np.zeros((environment.height, environment.width))
    for x in range(environment.height):
        for y in range(environment.width):
          q_values_of_state = agentQ.q_table[(x,y)]
          maxValue = max(q_values_of_state.values())
          value_map[x,y] = maxValue

    if save == True:
        utils.plotValueFunction(value_map,os.path.join(save_path,'heatmap.jpg'))
    
    return -(mean_reward)

def train(environment, agent, episodes=500, max_steps_per_episode=1000, min_epsilon=0.05, max_epsilon=1, epsilon_decay=0.01):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = []
    eps_history = []

    for episode in range(episodes):
        rewards = []
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True:
            old_state = environment.current_location
            action = agent.choose_action(environment.actions, True) 
            reward = environment.make_step(action)
            new_state = environment.current_location
            
            # update Q-values
            agent.learn(old_state, reward, new_state, action)
                
            cumulative_reward += reward
            step += 1
            rewards.append(reward)
            # start next trial upon reaching terminal state
            if environment.check_state() == 'TERMINAL': 
                environment.__init__()
                game_over = True     

        eps_history.append(agent.epsilon)

        # Cutting down on exploration by reducing the epsilon 
        agent.epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-epsilon_decay*episode)

        reward_per_episode.append(cumulative_reward)

    if save == True:
        utils.plotCumulativeReward2(episodes, reward_per_episode, os.path.join(save_path,'cumulativeReward.jpg'), eps_history)

def test(environment, agent, episodes=500, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = []

    for episode in range(episodes): 
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True:
            old_state = environment.current_location
            action = agent.choose_action(environment.actions, False) 
            reward = environment.make_step(action)
            new_state = environment.current_location
                
            cumulative_reward += reward
            step += 1
            
            if environment.check_state() == 'TERMINAL':
                environment.__init__()
                game_over = True     
                
        reward_per_episode.append(cumulative_reward)
        
    return sum(reward_per_episode)/len(reward_per_episode)

def argumentParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epsilon', default=0.05, type=float, help='Probability of chossing random action')
    parser.add_argument('--alpha', default=0.1, type=float, help='Learning Rate')
    parser.add_argument('--gamma', default=0.95, type=float, help='Discounting Factor')

    return parser

def limitDelta(hyperParameterValue,minVal,maxVal, expectedDelta):
    delta = min(expectedDelta, maxVal - hyperParameterValue)
    delta = max(minVal, delta)
    return delta


def run(space,algoType,algoName, modeName,maxEvals):
    trials = Trials()

    # calling the hyperopt function
    # Instead of minimizing an objective function we want to maximize it. To to this we need only return the negative of the function.
    best = fmin(objectiveFunction,space,algo=algoType,max_evals=maxEvals, trials=trials)
    # fmin function’s first argument is the objective function
    # second argument is the hyperopt space
    # third the algorithm that is to be used for optimization
    # maximum number of evaluations
    # fmin returns a dictionary giving best hyper-parameter

    
    results = pd.DataFrame({'loss': [x['loss'] for x in trials.results],
                    'learning_rate_iteration': trials.idxs_vals[0]['learning_rate'],
                    'learning_rate': trials.idxs_vals[1]['learning_rate'],
                    'min_epsilon_iteration': trials.idxs_vals[0]['min_epsilon'],
                    'min_epsilon': trials.idxs_vals[1]['min_epsilon'],
                    'max_epsilon_iteration': trials.idxs_vals[0]['max_epsilon'],
                    'max_epsilon': trials.idxs_vals[1]['max_epsilon'],
                    'epsilon_decay_iteration': trials.idxs_vals[0]['epsilon_decay'],
                    'epsilon_decay': trials.idxs_vals[1]['epsilon_decay'],
                    'discount_factor_iteration': trials.idxs_vals[0]['discount_factor'],
                    'discount_factor': trials.idxs_vals[1]['discount_factor']})
    

    histoSavePath=  os.path.join('graphs', algoName, modeName,'histogram')
    distriSavePath=  os.path.join('graphs', algoName, modeName,'distribution')

    # plot hyperparameters graphs
    utils.plotDistributionHistogram(results['learning_rate'],'learning_rate','Count', f'Histogram of {algoName} Values',os.path.join(histoSavePath,'learning_rate.jpg'))
    utils.plotDistributionHistogram(results['min_epsilon'],'min_epsilon','Count', f'Histogram of {algoName} Values',os.path.join(histoSavePath,'min_epsilon.jpg'))
    utils.plotDistributionHistogram(results['max_epsilon'],'max_epsilon','Count', f'Histogram of {algoName} Values',os.path.join(histoSavePath,'max_epsilon.jpg'))
    utils.plotDistributionHistogram(results['epsilon_decay'],'epsilon_decay','Count', f'Histogram of {algoName} Values',os.path.join(histoSavePath,'epsilon_decay.jpg'))
    utils.plotDistributionHistogram(results['discount_factor'],'discount_factor','Count', f'Histogram of {algoName} Values',os.path.join(histoSavePath,'discount_factor.jpg'))

    # plot Distribution values graphs
    utils.plotDistributionGraph(results['learning_rate_iteration'], results['learning_rate'],'Iteration','learning_rate',f'{algoName} Sequence of Values',os.path.join(distriSavePath,'learning_rate.jpg'))
    utils.plotDistributionGraph(results['min_epsilon_iteration'], results['min_epsilon'],'Iteration','min_epsilon',f'{algoName} Sequence of Values',os.path.join(distriSavePath,'min_epsilon.jpg'))
    utils.plotDistributionGraph(results['max_epsilon_iteration'], results['max_epsilon'],'Iteration','max_epsilon',f'{algoName} Sequence of Values',os.path.join(distriSavePath,'max_epsilon.jpg'))
    utils.plotDistributionGraph(results['epsilon_decay_iteration'], results['epsilon_decay'],'Iteration','epsilon_decay',f'{algoName} Sequence of Values',os.path.join(distriSavePath,'epsilon_decay.jpg'))
    utils.plotDistributionGraph(results['discount_factor_iteration'], results['discount_factor'],'Iteration','discount_factor',f'{algoName} Sequence of Values',os.path.join(distriSavePath,'discount_factor.jpg'))

    return(best)

def main():

    global save
    global save_path

    evalsNum = 1000

    tpe_algo = tpe.suggest
    rand_algo = rand.suggest

    # first domain run: uniform
    # defining the search space
    space = [hp.uniform('learning_rate',0.1,0.5),
            hp.uniform('min_epsilon',0.05,0.2),
            hp.uniform('max_epsilon',0.5,1),
            hp.uniform('epsilon_decay',0.01,0.1),
            hp.uniform('discount_factor',0.85,0.95)]

    print('start: tpe uniform')

    tpe_best = run(space,tpe_algo,'tpe','uniform',evalsNum)

    # lets run 1 time with the best values:
    bestValsArr = [tpe_best['learning_rate'], tpe_best['min_epsilon'], tpe_best['max_epsilon'], tpe_best['epsilon_decay'], tpe_best['discount_factor']]
    save = True
    save_path = os.path.join('graphs', 'tpe', 'uniform')
    rewardVal = objectiveFunction(bestValsArr)

    # save best values to file
    tpe_best['reward'] = -(rewardVal)
    with open(os.path.join(save_path,'bestValues.json'),'w') as json_file:
        json.dump(tpe_best, json_file)

    save = False
    save_path = ''

    print('done: tpe uniform, reward: ' + str(rewardVal))
###########################################################################

    print('start: tpe normal')
    # tpe + normal
    tpe_space_normal = [hp.normal('learning_rate', tpe_best['learning_rate'], limitDelta(tpe_best['learning_rate'],0.05,0.99,0.01)),
                        hp.normal('min_epsilon', tpe_best['min_epsilon'], limitDelta(tpe_best['min_epsilon'],0.05,0.99,0.01)),
                        hp.normal('max_epsilon', tpe_best['max_epsilon'], limitDelta(tpe_best['max_epsilon'],0.05,0.99,0.01)),
                        hp.normal('epsilon_decay', tpe_best['epsilon_decay'], limitDelta(tpe_best['epsilon_decay'],0.05,0.99,0.01)),
                        hp.normal('discount_factor', tpe_best['discount_factor'], limitDelta(tpe_best['discount_factor'],0.05,0.99,0.01))]

    tpe_best_normal = run(tpe_space_normal,tpe_algo,'tpe','normal',evalsNum)

    # lets run 1 time with the best values:
    bestValsArr = [tpe_best_normal['learning_rate'], tpe_best_normal['min_epsilon'], tpe_best_normal['max_epsilon'], tpe_best_normal['epsilon_decay'], tpe_best_normal['discount_factor']]
    save = True
    save_path = os.path.join('graphs', 'tpe', 'normal')
    rewardVal = objectiveFunction(bestValsArr)

    # save best values to file
    tpe_best_normal['reward'] = -(rewardVal)
    with open(os.path.join(save_path,'bestValues.json'),'w') as json_file:
        json.dump(tpe_best_normal, json_file)

    save = False
    save_path = ''

    print('done: tpe normal, reward: ' + str(rewardVal))
###########################################################################


    print('start: random uniform')

    random_best = run(space,rand_algo,'random','uniform',evalsNum)

    # lets run 1 time with the best values:
    bestValsArr = [random_best['learning_rate'], random_best['min_epsilon'], random_best['max_epsilon'], random_best['epsilon_decay'], random_best['discount_factor']]
    save = True
    save_path = os.path.join('graphs', 'random', 'uniform')
    rewardVal = objectiveFunction(bestValsArr)

    # save best values to file
    random_best['reward'] = -(rewardVal)
    with open(os.path.join(save_path,'bestValues.json'),'w') as json_file:
        json.dump(random_best, json_file)

    save = False
    save_path = ''

    print('done: random uniform, reward: ' + str(rewardVal))
###########################################################################

    print('start: random normal')
    # random + normal
    rand_space_normal = [hp.normal('learning_rate', random_best['learning_rate'], limitDelta(random_best['learning_rate'],0.05,0.99,0.01)),
                            hp.normal('min_epsilon', random_best['min_epsilon'], limitDelta(random_best['min_epsilon'],0.05,0.99,0.01)),
                            hp.normal('max_epsilon', random_best['max_epsilon'], limitDelta(random_best['max_epsilon'],0.05,0.99,0.01)),
                            hp.normal('epsilon_decay', random_best['epsilon_decay'], limitDelta(random_best['epsilon_decay'],0.05,0.99,0.01)),
                            hp.normal('discount_factor', random_best['discount_factor'], limitDelta(random_best['discount_factor'],0.05,0.99,0.01))]

    rand_best_normal = run(tpe_space_normal,rand_algo,'random','normal',evalsNum)

    # lets run 1 time with the best values:
    bestValsArr = [rand_best_normal['learning_rate'], rand_best_normal['min_epsilon'], rand_best_normal['max_epsilon'], rand_best_normal['epsilon_decay'], rand_best_normal['discount_factor']]
    save = True
    save_path = os.path.join('graphs', 'random', 'normal')
    rewardVal = objectiveFunction(bestValsArr)

    # save best values to file
    rand_best_normal['reward'] = -(rewardVal)
    with open(os.path.join(save_path,'bestValues.json'),'w') as json_file:
        json.dump(rand_best_normal, json_file)

    save = False
    save_path = ''

    print('done: random normal, reward: ' + str(rewardVal))
###########################################################################

    print('done all!')



if __name__ == '__main__':

    main()

    
