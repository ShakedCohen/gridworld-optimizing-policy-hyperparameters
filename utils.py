
import numpy as np
import matplotlib.pyplot as plt
import os

def plotCumulativeReward(train_episodes, training_rewards,path):
    x = range(train_episodes)
    plt.plot(x, training_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Training total reward')
    plt.title('Total rewards over all episodes in training') 
    plt.savefig(path)
    plt.clf()


def plotCumulativeReward2(train_episodes, training_rewards,path,epsilons):
    x = range(train_episodes)

    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C1")
    ax.set_ylabel("Epsilon", color="C1")
    ax.tick_params(axis='x', colors="C1")
    ax.tick_params(axis='y', colors="C1")
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()

    ax2.plot(x, training_rewards, color="C0")
    ax2.set_xlabel('Episode', color="C0")
    ax2.set_ylabel('Training total reward', color="C0")
    

    plt.title('Total rewards over all episodes in training') 
    plt.savefig(path)
    plt.clf()

def plotValueFunction(value_map,path):
    c = plt.imshow(value_map, cmap='hot', interpolation='nearest')
    plt.colorbar(c)
    plt.title('Value Function', fontweight ="bold")
    plt.savefig(path)
    plt.clf()

def plotDistributionGraph(x_axis,y_axis,x_title,y_title,main_title, path):
    plt.figure(figsize = (10, 8))
    plt.plot(x_axis , y_axis,  'bo', alpha = 0.5)
    plt.xlabel(x_title, size = 22); plt.ylabel(y_title, size = 22); plt.title(main_title, size = 24)
    plt.savefig(path)
    plt.clf()


def plotDistributionHistogram(hyperParameterValues,x_title,y_title, main_title, path):
    plt.figure(figsize = (8, 6))
    plt.hist(hyperParameterValues, bins = 50, edgecolor = 'k')
    plt.title(main_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    asdf = os.getcwd()
    directory_contents = os.listdir(asdf)
    plt.savefig(path)
    plt.clf()
