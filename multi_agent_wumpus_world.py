# Imports
import cv2
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from operator import add
import random

# This ensures that all the data isn't loaded into the GPU memory at once.
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Disables eager execution.
tf.compat.v1.disable_eager_execution()


# Defining the Multi-Agent Wumpus World Environment.
class MultiAgentWumpusWorldEnvironment(gym.Env):
    """This class implements the Wumpus World environment."""

    def __init__(self, environment_type):
        """This method initializes the environment.

        :param environment_type: - (It can take two values: 1. 'deterministic' 2. 'stochastic' indicating the type of
                                    environment.)"""

        self.environment_type = environment_type  # This indicates whether the environment is of the type deterministic
                                                  # or stochastic.
        self.observation_space = spaces.Discrete(16)  # This defines that there are 16 states in the environment.
        self.action_space = spaces.Discrete(6)  # This defines that there are 6 discrete actions that the agent can
                                                # perform.
        self.number_of_agents = 2  # This defines the number of agents in the environment.
        self.number_of_arrows = 10  # This defines the number of arrows each agent has.
        self.arrows = [self.number_of_arrows for _ in range(self.number_of_agents)]  # List indicating the number of
                                                                                     # arrows per agent.
        self.agent_positions = [[0, 0], [3, 1]]  # Initializing the agents in fixed positions.
        self.breeze_pos = [[1, 0], [1, 2], [2, 1], [2, 3], [3, 0], [3, 2]]  # This defines the positions of breeze
                                                                            # in the environment.
        self.gold_pos = [3, 3]  # This defines the position of gold in the environment.
        self.gold_quantity = 1  # This defines the quantity of gold.
        self.pit_pos = [[2, 0], [2, 2]]  # This defines the positions of pit in the environment.
        self.stench_pos = [[0, 1], [0, 3], [1, 2]]  # This defines the positions of stench in the environment.
        self.wumpus_pos = [0, 2]  # This defines the position of the Wumpus in the environment.
        self.wumpus_alive = True  # Boolean indicating whether the Wumpus is alive or dead.
        self.timesteps = 0  # This defines the steps the agent has taken during an episode.
        self.max_timesteps = 10  # This defines the maximum steps the agent can take during an episode.
        # This defines the distance of the agents to the Gold.
        self.gold_distance = [self.compute_distance(self.agent_positions[i], self.gold_pos)
                              for i in range(self.number_of_agents)]
        # This defines the distance of the agents to the Wumpus.
        self.wumpus_distance = [self.compute_distance(self.agent_positions[i], self.wumpus_pos)
                                for i in range(self.number_of_agents)]

    def reset(self, random_start=False):
        """This method resets the agent position and returns the state as the observation.

        :param random_start: - (Boolean indicating whether the agent will start in a random or fixed position.)

        :returns observation: - (Integers from 0 to 15 defining the agent's position in the environment."""

        # Creating the mapping from the possible states the agent can start in to the co-ordinates.
        coordinates_state_mapping = {12: [0, 3], 13: [1, 3], 14: [2, 3],
                                     9: [1, 2], 11: [3, 2],
                                     4: [0, 1], 5: [1, 1], 6: [2, 1], 7: [3, 1],
                                     0: [0, 0], 1: [1, 0], 3: [3, 0]}

        if not random_start:
            self.agent_positions = [[0, 0], [3, 1]]
            observation = [0, 7]
            # The state mapping for position [0, 0] is '0', and for position [3, 1] is '7'.

        else:
            # Randomly selecting the agent's position.
            observation = [random.choice([0, 1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14]) for
                           _ in range(self.number_of_agents)]

            self.agent_positions = [coordinates_state_mapping[observation[i]] for i in range(self.number_of_agents)]

        self.arrows = [self.number_of_arrows for _ in range(self.number_of_agents)]  # Resetting the number of arrows.
        self.wumpus_alive = True  # Resetting the Wumpus to be alive.
        self.gold_quantity = 1  # Resetting the Gold quantity to be 1.
        self.timesteps = 0  # Resetting the number of steps taken by the agent.
        # Resetting the distance of the agents to the Gold.
        self.gold_distance = [self.compute_distance(self.agent_positions[i], self.gold_pos)
                              for i in range(self.number_of_agents)]
        # Resetting the distance of the agents to the Wumpus.
        self.wumpus_distance = [self.compute_distance(self.agent_positions[i], self.wumpus_pos)
                                for i in range(self.number_of_agents)]

        return observation

    def step(self, actions):
        """This function implements what happens when the agent takes a particular action. It changes the agent's
        position (While not allowing it to go out of the environment space.), maps the environment co-ordinates to a
        state, defines the rewards for the various states, and determines when the episode ends.

        :param actions: - (Vector of Integers in the range 0 to 5 inclusive.)

        :returns observation: - (Vector of Integers from 0 to 15 defining the agent's position in the environment.)
                 rewards: - (Vector of Integers values that are used to measure the performance of the agents.)
                 done: - (Boolean describing whether or not the episode has ended.)
                 info: - (A dictionary that can be used to provide additional implementation information.)"""

        if self.environment_type == 'deterministic':
            for i in range(self.number_of_agents):
                # Describing the outcomes of the various possible actions.
                if actions[i] == 0:  # This action causes the agent to go right.
                    self.agent_positions[i][0] += 1
                if actions[i] == 1:  # This action causes the agent to go left.
                    self.agent_positions[i][0] -= 1
                if actions[i] == 2:  # This action causes the agent to go up.
                    self.agent_positions[i][1] += 1
                if actions[i] == 3:  # This action causes the agent to go down.
                    self.agent_positions[i][1] -= 1
                if actions[i] == 4:  # This action causes the agent to not move.
                    self.agent_positions[i] = self.agent_positions[i]
                if actions[i] == 5:  # This action causes the agent to shoot an arrow.
                    if self.arrows[i] > 0:
                        self.arrows[i] -= 1

        if self.environment_type == 'stochastic':
            for i in range(self.number_of_agents):
                # Describing the outcomes of the various possible actions.
                if actions[i] == 0:  # This action causes the agent to go right.
                    probability = random.uniform(0, 1)
                    if probability > 0.1:
                        self.agent_positions[i][0] += 1
                if actions[i] == 1:  # This action causes the agent to go left.
                    probability = random.uniform(0, 1)
                    if probability > 0.1:
                        self.agent_positions[i][0] -= 1
                if actions[i] == 2:  # This action causes the agent to go up.
                    probability = random.uniform(0, 1)
                    if probability > 0.1:
                        self.agent_positions[i][1] += 1
                if actions[i] == 3:  # This action causes the agent to go down.
                    probability = random.uniform(0, 1)
                    if probability > 0.1:
                        self.agent_positions[i][1] -= 1
                if actions[i] == 4:  # This action causes the agent to not move.
                    probability = random.uniform(0, 1)
                    if probability > 0.1:
                        self.agent_positions[i] = self.agent_positions[i]
                if actions[i] == 5:  # This action causes the agent to shoot an arrow.
                    probability = random.uniform(0, 1)
                    if probability > 0.1:
                        if self.arrows[i] > 0:
                            self.arrows[i] -= 1

        # Ensuring that the agent doesn't go out of the environment.
        self.agent_positions = np.clip(self.agent_positions, a_min=0, a_max=3)

        # Computing the new distance of the agents to the Gold.
        new_gold_distance = [self.compute_distance(self.agent_positions[i], self.gold_pos)
                             for i in range(self.number_of_agents)]

        # Setting the rewards to 0.
        rewards = [0 for _ in range(self.number_of_agents)]

        # Giving the agents reward 10 for picking up the Gold.
        for i in range(self.number_of_agents):
            if (self.agent_positions[i] == self.gold_pos).all():
                if self.gold_quantity > 0:
                    self.gold_quantity -= 1
                    rewards[i] = 10

        # Giving the agents different rewards if their distance to the Gold increases, decreases or remains the same.
        if self.gold_quantity > 0:
            for i in range(self.number_of_agents):
                # If agent moves away from the Gold it gets reward -1.
                if new_gold_distance[i] > self.gold_distance[i]:
                    rewards[i] = -1
                    self.gold_distance[i] = new_gold_distance[i]

                # If agent moves closer to the Gold it gets reward 1.
                elif new_gold_distance[i] < self.gold_distance[i]:
                    rewards[i] = 1
                    self.gold_distance[i] = new_gold_distance[i]

                else:  # If agent's distance to the Gold doesn't change it gets reward 0.
                    rewards[i] = 0

        # Computing the new distance of the agents to the Wumpus.
        new_wumpus_distance = [self.compute_distance(self.agent_positions[i], self.wumpus_pos)
                               for i in range(self.number_of_agents)]

        # Giving the agents different rewards if their distance to the Wumpus increases, decreases or remains the same.
        # These reward will only come in when the Gold is collected.
        if self.gold_quantity == 0:
            # If agent moves away from the Wumpus it gets reward -1.
            for i in range(self.number_of_agents):
                if new_wumpus_distance[i] > self.wumpus_distance[i]:
                    rewards[i] = -1
                    self.wumpus_distance[i] = new_wumpus_distance[i]

                # If agent moves closer to the Wumpus it gets reward 1.
                elif new_wumpus_distance[i] < self.wumpus_distance[i]:
                    rewards[i] = 1
                    self.wumpus_distance[i] = new_wumpus_distance[i]

                else:  # If the agent's distance to the Wumpus doesn't change it gets reward 0.
                    rewards[i] = 0

        # Creating the mapping from the co-ordinates to the state.
        coordinates_state_mapping = {'[0 3]': 12, '[1 3]': 13, '[2 3]': 14, '[3 3]': 15,
                                     '[0 2]': 8, '[1 2]': 9, '[2 2]': 10, '[3 2]': 11,
                                     '[0 1]': 4, '[1 1]': 5, '[2 1]': 6, '[3 1]': 7,
                                     '[0 0]': 0, '[1 0]': 1, '[2 0]': 2, '[3 0]': 3}

        # Setting the observation to be the state occupied by the agent.
        observation = [coordinates_state_mapping[f'{self.agent_positions[i]}'] for i in range(self.number_of_agents)]

        self.timesteps += 1  # Increasing the total number of steps taken by the agent.

        # Setting the reward to -1 if the agent falls into the pit.
        for i in range(self.number_of_agents):
            for j in range(len(self.pit_pos)):
                if (self.agent_positions[i] == self.pit_pos[j]).all():
                    rewards[i] = -1

            # Setting the reward to -1 if the agent is killed by the Wumpus.
            if (self.agent_positions[i] == self.wumpus_pos).all():
                rewards[i] = -1

        # Giving the agents reward 10 for killing the Wumpus.
        if all((self.wumpus_distance[i] == 1 and actions[i] == 5 and self.arrows[i] > 0)
               for i in range(self.number_of_agents)):
            rewards = [10 for _ in range(self.number_of_agents)]
            self.wumpus_alive = False

        # The episode terminates when one agent picks the Gold and both agents kill the Wumpus together,
        # or at least one agent is killed by the Wumpus, falls into the pit, or takes more than 10 steps.
        if (self.gold_quantity == 0 and not self.wumpus_alive) or \
                any((self.agent_positions[i] == self.wumpus_pos).all() for i in range(self.number_of_agents)):
            done = True
        else:
            done = False
        for i in range(len(self.pit_pos)):
            if any((self.agent_positions[i] == self.pit_pos[i]).all() for i in range(self.number_of_agents)):
                done = True
        if self.timesteps == self.max_timesteps:
            done = True
        info = {}

        return observation, rewards, done, info

    @staticmethod
    def compute_distance(x, y):
        """This method computes the distance between the old and new.

        :param x: This is the first array representing the agent position.
        :param y: This is the second array representing the goal position.

        :returns distance: The Manhattan distance between the agent and the goal."""

        distance = np.abs(x[0] - y[0]) + np.abs(x[1] - y[1])
        return distance

    def render(self, mode='human', plot=False):
        """This method renders the environment.

        :param mode:
        :param plot: Boolean indicating whether we show a plot or not. If False, the method returns a resized NumPy
                     array representation of the environment to be used as the state. If True it plots the environment.

        :returns preprocessed_image: Grayscale NumPy array representation of the environment."""

        fig, ax = plt.subplots(figsize=(10, 10))  # Initializing the figure.
        ax.set_xlim(0, 4)  # Setting the limit on the x-axis.
        ax.set_ylim(0, 4)  # Setting the limit on the y-axis.

        def plot_image(plot_pos):
            """This is a helper function to render the environment. It checks which objects are in a particular
            position on the grid and renders the appropriate image.

            :param plot_pos: Co-ordinates of the grid position which needs to be rendered."""

            # Initially setting every object to not be plotted.
            plot_agent_1, plot_agent_2, plot_breeze, plot_gold, plot_pit, plot_stench, plot_wumpus = \
                False, False, False, False, False, False, False

            # Checking which objects need to be plotted by comparing their positions.
            if plot_pos[0] == self.agent_positions[0][0] and plot_pos[1] == self.agent_positions[0][1]:
                plot_agent_1 = True
            if plot_pos[0] == self.agent_positions[1][0] and plot_pos[1] == self.agent_positions[1][1]:
                plot_agent_2 = True
            for i in range(len(self.breeze_pos)):
                if plot_pos == self.breeze_pos[i]:
                    plot_breeze = True
            if self.gold_quantity > 0:  # Gold isn't plotted if it has already been picked by one of the agents.
                if plot_pos == self.gold_pos:
                    plot_gold = True
            for i in range(len(self.pit_pos)):
                if plot_pos == self.pit_pos[i]:
                    plot_pit = True
            for i in range(len(self.stench_pos)):
                if plot_pos == self.stench_pos[i]:
                    plot_stench = True
            if plot_pos == self.wumpus_pos:
                plot_wumpus = True

            # Plot for Agent 1.
            if plot_agent_1 and \
                    all(not item for item in
                        [plot_agent_2, plot_breeze, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent_1 = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1.png'), zoom=0.36),
                                         list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_1)

            # Plot for Agent 2.
            if plot_agent_2 and \
                    all(not item for item in
                        [plot_agent_1, plot_breeze, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent_2 = AnnotationBbox(OffsetImage(plt.imread('./images/agent_2.png'), zoom=0.36),
                                         list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_2)

            # Plot for Breeze.
            if plot_breeze and \
                    all(not item for item in
                        [plot_agent_1, plot_agent_2, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                breeze = AnnotationBbox(OffsetImage(plt.imread('./images/breeze.png'), zoom=0.36),
                                        list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(breeze)

            # Plot for Gold.
            if plot_gold and \
                    all(not item for item in
                        [plot_agent_1, plot_agent_2, plot_breeze, plot_pit, plot_stench, plot_wumpus]):
                gold = AnnotationBbox(OffsetImage(plt.imread('./images/gold.png'), zoom=0.36),
                                      list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(gold)

            # Plot for Pit.
            if plot_pit and \
                    all(not item for item in
                        [plot_agent_1, plot_agent_2, plot_breeze, plot_gold, plot_stench, plot_wumpus]):
                pit = AnnotationBbox(OffsetImage(plt.imread('./images/pit.png'), zoom=0.36),
                                     list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(pit)

            # Plot for Stench.
            if plot_stench and \
                    all(not item for item in
                        [plot_agent_1, plot_agent_2, plot_breeze, plot_gold, plot_pit, plot_wumpus]):
                stench = AnnotationBbox(OffsetImage(plt.imread('./images/stench.png'), zoom=0.36),
                                        list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(stench)

            # Plot for alive Wumpus.
            if self.wumpus_alive:
                if plot_wumpus and \
                        all(not item for item in
                            [plot_agent_1, plot_agent_2, plot_breeze, plot_gold, plot_pit, plot_stench]):
                    wumpus_alive = AnnotationBbox(OffsetImage(plt.imread('./images/wumpus.png'), zoom=0.36),
                                                  list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                    ax.add_artist(wumpus_alive)

            # Plot for dead Wumpus.
            if not self.wumpus_alive:
                if plot_wumpus and \
                        all(not item for item in
                            [plot_agent_1, plot_agent_2, plot_breeze, plot_gold, plot_pit, plot_stench]):
                    wumpus_dead = AnnotationBbox(OffsetImage(plt.imread('./images/wumpus_dead.png'), zoom=0.36),
                                                 list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                    ax.add_artist(wumpus_dead)

            # Plot for Agent 1 and Agent 2.
            if all(item for item in [plot_agent_1, plot_agent_2]) and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent_1_2 = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_2.png'), zoom=0.36),
                                           list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_1_2)

            # Plot for Agent 1 and Breeze.
            if all(item for item in [plot_agent_1, plot_breeze]) and \
                    all(not item for item in
                        [plot_agent_2, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent_1_breeze = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_breeze.png'), zoom=0.36),
                                                list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_1_breeze)

            # Plot for Agent 2 and Breeze.
            if all(item for item in [plot_agent_2, plot_breeze]) and \
                    all(not item for item in
                        [plot_agent_1, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent_2_breeze = AnnotationBbox(OffsetImage(plt.imread('./images/agent_2_breeze.png'), zoom=0.36),
                                                list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_2_breeze)

            # Plot for Agent 1, Agent 2, and Breeze.
            if all(item for item in [plot_agent_1, plot_agent_2, plot_breeze]) and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent_1_2_breeze = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_2_breeze.png'), zoom=0.36),
                                                  list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_1_2_breeze)

            # Plot for Agent 1 and Pit.
            if all(item for item in [plot_agent_1, plot_pit]) and \
                    all(not item for item in
                        [plot_agent_2, plot_breeze, plot_gold, plot_stench, plot_wumpus]):
                agent_1_pit = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_dead_pit.png'), zoom=0.36),
                                             list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_1_pit)

            # Plot for Agent 2 and Pit.
            if all(item for item in [plot_agent_2, plot_pit]) and \
                    all(not item for item in
                        [plot_agent_1, plot_breeze, plot_gold, plot_stench, plot_wumpus]):
                agent_2_pit = AnnotationBbox(OffsetImage(plt.imread('./images/agent_2_dead_pit.png'), zoom=0.36),
                                             list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_2_pit)

            # Plot for Agent 1, Agent 2, and Pit.
            if all(item for item in [plot_agent_1, plot_agent_2, plot_pit]) and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_stench, plot_wumpus]):
                agent_1_2_pit = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_2_dead_pit.png'), zoom=0.36),
                                               list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_1_2_pit)

            # Plot for Agent 1 and Stench.
            if all(item for item in [plot_agent_1, plot_stench]) and \
                    all(not item for item in
                        [plot_agent_2, plot_breeze, plot_gold, plot_pit, plot_wumpus]):
                agent_1_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_stench.png'), zoom=0.36),
                                                list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_1_stench)

            # Plot for Agent 2 and Stench.
            if all(item for item in [plot_agent_2, plot_stench]) and \
                    all(not item for item in
                        [plot_agent_1, plot_breeze, plot_gold, plot_pit, plot_wumpus]):
                agent_2_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_2_stench.png'), zoom=0.36),
                                                list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_2_stench)

            # Plot for Agent 1, Agent 2, and Stench.
            if all(item for item in [plot_agent_1, plot_agent_2, plot_stench]) and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_pit, plot_wumpus]):
                agent_1_2_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_2_stench.png'), zoom=0.36),
                                                  list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_1_2_stench)

            # Plot for Agent 1, Breeze, Stench.
            if all(item for item in [plot_agent_1, plot_breeze, plot_stench]) and \
                    all(not item for item in
                        [plot_agent_2, plot_gold, plot_pit, plot_wumpus]):
                agent_1_breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_breeze_stench.png'),
                                                                   zoom=0.36), list(map(add, plot_pos, [0.5, 0.5])),
                                                       frameon=False)
                ax.add_artist(agent_1_breeze_stench)

            # Plot for Agent 2, Breeze, Stench.
            if all(item for item in [plot_agent_2, plot_breeze, plot_stench]) and \
                    all(not item for item in
                        [plot_agent_1, plot_gold, plot_pit, plot_wumpus]):
                agent_2_breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_2_breeze_stench.png'),
                                                                   zoom=0.36), list(map(add, plot_pos, [0.5, 0.5])),
                                                       frameon=False)
                ax.add_artist(agent_2_breeze_stench)

            # Plot for Agent 1, Agent 2, Breeze, and Stench.
            if all(item for item in [plot_agent_1, plot_agent_2, plot_breeze, plot_stench]) and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_wumpus]):
                agent_1_2_breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_1_2_breeze_stench.png'),
                                                                     zoom=0.36), list(map(add, plot_pos, [0.5, 0.5])),
                                                         frameon=False)
                ax.add_artist(agent_1_2_breeze_stench)

            # Plot for Breeze and Gold.
            if all(item for item in [plot_breeze, plot_gold]) and \
                    all(not item for item in
                        [plot_agent_1, plot_agent_2, plot_pit, plot_stench, plot_wumpus]):
                breeze_gold = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_gold.png'), zoom=0.36),
                                             list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(breeze_gold)

            # Plot for Breeze and Stench.
            if all(item for item in [plot_breeze, plot_stench]) and \
                    all(not item for item in
                        [plot_agent_1, plot_agent_2, plot_gold, plot_pit, plot_wumpus]):
                breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_stench.png'), zoom=0.36),
                                               list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(breeze_stench)

            # Plot for Breeze, Stench, and Gold.
            if all(item for item in [plot_breeze, plot_gold, plot_stench]) and \
                    all(not item for item in
                        [plot_agent_1, plot_agent_2, plot_pit, plot_wumpus]):
                breeze_gold_stench = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_gold_stench.png'),
                                                    zoom=0.36), list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(breeze_gold_stench)

            # Plot for Stench and Gold.
            if all(item for item in [plot_stench, plot_gold]) and \
                    all(not item for item in
                        [plot_agent_1, plot_agent_2, plot_breeze, plot_pit, plot_wumpus]):
                stench_gold = AnnotationBbox(OffsetImage(plt.imread('./images/stench_gold.png'), zoom=0.36),
                                             list(map(add, plot_pos, [0.5, 0.5])), frameon=False)
                ax.add_artist(stench_gold)

        # Dictionary mapping the states to their co-ordinates.
        coordinates_state_mapping = {
            12: [0, 3], 13: [1, 3], 14: [2, 3], 15: [3, 3],
            8: [0, 2], 9: [1, 2], 10: [2, 2], 11: [3, 2],
            4: [0, 1], 5: [1, 1], 6: [2, 1], 7: [3, 1],
            0: [0, 0], 1: [1, 0], 2: [2, 0], 3: [3, 0]}

        # Rendering the images for all states.
        for state in range(self.observation_space.n):
            plot_image(coordinates_state_mapping[state])

        plt.xticks([0, 1, 2, 3])  # Specifying the ticks on the x-axis.
        plt.yticks([0, 1, 2, 3])  # Specifying the ticks on the y-axis.
        plt.grid()  # Setting the plot to be of the type 'grid'.

        if plot:  # Displaying the plot.
            plt.show()
        else:  # Returning the preprocessed image representation of the environment.
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :1]
            width = int(img.shape[1] * 84 / 1000)
            height = int(img.shape[0] * 84 / 1000)
            dim = (width, height)
            preprocessed_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            return preprocessed_image


class AdvantageWeightedRegression:
    """This class implements the AWR Agent."""

    def __init__(self, environment, alternate_network=False, offline_memory_size=10000, iterations=10):
        """This method initializes the AWR parameters, and calls the train, evaluate and render_actions methods.

        :param environment: This is the environment on which the agent will learn.
        :param alternate_network: Boolean indicating whether to use the second deeper network.
        :param offline_memory_size: Integer indicating the size of the offline replay memory.
        :param iterations: Integer indicating the number of episodes for which the agent will train."""

        self.environment = environment  # The environment which we need the agent to solve.
        self.alternate_network = alternate_network  # Boolean indicating whether to use the second deeper network.
        self.offline_replay_memory_size = offline_memory_size  # This specifies the length of the offline replay memory.
        # Creating as many replay memories as there are agents.
        self.offline_replay_memory = [[] for _ in range(self.environment.number_of_agents)]
        self.iterations = iterations  # Number of episodes for which the agent will train.
        self.discount_factor = 0.9  # Discount factor determines the value of the future rewards.
        self.beta = 0.5  # Hyper-parameter used to calculate the exponential advantage.
        # Creating the actor, critic and policy models.
        self.actor_model, self.critic_model, self.policy_model = [], [], []
        for _ in range(self.environment.number_of_agents):
            self.actor_model_, self.critic_model_, self.policy_model_ = self.neural_network()
            self.actor_model.append(self.actor_model_)
            self.critic_model.append(self.critic_model_)
            self.policy_model.append(self.policy_model_)

        self.cumulative_rewards_evaluation = []  # List containing the cumulative rewards per episode during evaluation.
        self.train()  # Calling the train method.
        self.evaluate()  # Calling the evaluate method.
        self.render_actions()  # Calling the render method.

    def neural_network(self):
        """This method builds the actor, critic and policy networks."""

        if not self.alternate_network:
            # Input 1 is the one-hot representation of the environment state.
            input_ = Input(shape=(self.environment.observation_space.n,))
            # Input 2 is the exponential advantage.
            exponential_advantage = Input(shape=[1])
            common1 = Dense(64, activation='relu')(input_)  # Common layer for the networks.
            common2 = Dense(128, activation='relu')(common1)
            probabilities = Dense(self.environment.action_space.n, activation='softmax')(common2)  # Actor output.
            values = Dense(1, activation='linear')(common2)  # Critic output.

        else:
            # Input 1 is the one-hot representation of the environment state.
            input_ = Input(shape=(self.environment.observation_space.n,))
            # Input 2 is the exponential advantage.
            exponential_advantage = Input(shape=[1])
            common1 = Dense(512, activation='relu')(input_)  # Common layer 1 for the networks.
            common2 = Dense(256, activation='relu')(common1)  # Common layer 2 for the networks.
            common3 = Dense(128, activation='relu')(common2)  # Common layer 3 for the networks.
            probabilities = Dense(self.environment.action_space.n, activation='softmax')(common3)  # Actor output.
            values = Dense(1, activation='linear')(common3)  # Critic output.

        def custom_loss(exponential_advantage_):
            """This method defines the custom loss wrapper function that will be used by the actor model."""

            def loss_fn(y_true, y_pred):
                # Clipping y_pred so that we don't end up taking the log of 0 or 1.
                clipped_y_pred = k.clip(y_pred, 1e-8, 1 - 1e-8)
                log_probability = y_true * k.log(clipped_y_pred)
                return k.sum(-log_probability * exponential_advantage_)
            return loss_fn

        # Instantiating the actor model.
        actor_model = Model(inputs=[input_, exponential_advantage], outputs=[probabilities])
        actor_model.compile(optimizer=Adam(), loss=custom_loss(exponential_advantage))

        # Instantiating the critic model.
        critic_model = Model(inputs=[input_], outputs=[values])
        critic_model.compile(optimizer=Adam(), loss=tf.keras.losses.Huber())

        # Instantiating the policy model.
        policy_model = Model(inputs=[input_], outputs=[probabilities])

        return actor_model, critic_model, policy_model

    def monte_carlo_returns(self, index):
        """This method calculates the Monte Carlo returns given a list of rewards.

        :param index: Integer indicating the agent for whom we will calculate the Monte Carlo returns."""

        rewards = [item[2] for item in self.offline_replay_memory[index]]
        monte_carlo_returns = []  # List containing the Monte-Carlo returns.
        monte_carlo_return = 0
        t = 0  # Exponent by which the discount factor is raised.

        for i in range(len(self.offline_replay_memory[index])):

            while not self.offline_replay_memory[index][i][4]:  # Execute until you encounter a terminal state.

                # Equation to calculate the Monte-Carlo return.
                monte_carlo_return += self.discount_factor ** t * rewards[i]
                i += 1  # Go to the next sample.
                t += 1  # Increasing the exponent by which the discount factor is raised.

                # Condition to check whether we have reached the end of the replay memory without the episode being
                # terminated, and if so break. (This can happen with the samples at the end of the replay memory as we
                # only store the samples till we reach the replay memory size and not till we exceed it with the episode
                # being terminated.)
                if i == len(self.offline_replay_memory[index]):

                    # If the episode hasn't terminated but you reach the end append the Monte-Carlo return to the list.
                    monte_carlo_returns.append(monte_carlo_return)

                    # Resetting the Monte-Carlo return value and the exponent to 0.
                    monte_carlo_return = 0
                    t = 0

                    break  # Break from the loop.

            # If for one of the samples towards the end we reach the end of the replay memory and it hasn't terminated,
            # we will go back to the beginning of the for loop to calculate the Monte-Carlo return for the future
            # samples if any for whom the episode hasn't terminated.
            if i == len(self.offline_replay_memory[index]):
                continue

            # Equation to calculate the Monte-Carlo return.
            monte_carlo_return += self.discount_factor ** t * rewards[i]

            # Appending the Monte-Carlo Return for cases where the episode terminates without reaching the end of the
            # replay memory.
            monte_carlo_returns.append(monte_carlo_return)

            # Resetting the Monte-Carlo return value and the exponent to 0.
            monte_carlo_return = 0
            t = 0

        # Normalizing the returns.
        monte_carlo_returns = np.array(monte_carlo_returns)
        monte_carlo_returns = (monte_carlo_returns - np.mean(monte_carlo_returns)) / (np.std(monte_carlo_returns)
                                                                                      + 1e-08)
        monte_carlo_returns = monte_carlo_returns.tolist()

        return monte_carlo_returns

    def replay(self, i):
        """This is the replay method, that is used to fit the actor and critic networks and synchronize the weights
            between the actor and policy networks.

        :param i: Integer indicating the agent for whom we will perform the replay. """

        states = [item[0] for item in self.offline_replay_memory[i]]
        states = np.asarray(states).reshape(-1, self.environment.observation_space.n)

        actions = [tf.keras.utils.to_categorical(item[1], self.environment.action_space.n).tolist()
                   for item in self.offline_replay_memory[i]]

        monte_carlo_returns = self.monte_carlo_returns(i)

        critic_values = self.critic_model[i].predict(states).flatten()

        exponential_advantages = [np.exp(1/self.beta * (monte_carlo_returns[x] - critic_values[x]))
                      for x in range(len(self.offline_replay_memory[i]))]

        # Fitting the actor model.
        self.actor_model[i].fit([states, np.asarray(exponential_advantages)], np.asarray(actions),
                             batch_size=16, epochs=1, verbose=0)

        # Syncing the weights between the actor and policy models.
        self.policy_model[i].set_weights(self.actor_model[i].get_weights())

        # Fitting the critic model.
        self.critic_model[i].fit(states, np.asarray(monte_carlo_returns), batch_size=16, epochs=1, verbose=0)

    def train(self):
        """This method performs the agent training."""

        # Environment states for indirectly plotting the policy table.
        test_states = [x for x in range(self.environment.observation_space.n)]
        test_states = tf.keras.utils.to_categorical(test_states)

        # Printing the initial policy table.
        policy_table = [self.policy_model[i].predict(
            np.asarray(test_states).reshape(-1, self.environment.observation_space.n))
            for i in range(self.environment.number_of_agents)]
        for i in range(self.environment.number_of_agents):
            for terminal_state in [2, 8, 10, 15]:  # The list represents terminal states.
                policy_table[i][terminal_state] = 0
        for i in range(self.environment.number_of_agents):
            print(f'\nInitial Policy Table for Agent {i + 1}:\n', policy_table[i])

        average_reward_per_episode_per_iteration = []
        cumulative_average_rewards_per_episode_per_iteration = []
        wumpus_killed_list = []  # List containing the percentage of episodes in which the agents killed the Wumpus per
                                 # iteration.

        for iteration in range(self.iterations):

            # Resetting the offline replay memories for all agents to be empty.
            self.offline_replay_memory = [[] for _ in range(self.environment.number_of_agents)]
            total_reward_iteration = 0  # Total reward acquired in this iteration.
            wumpus_killed = 0  # Initializing the number of episodes in which the agents killed the Wumpus to be 0.
            episodes = 0  # Initializing the number of episodes in this iteration to be 0.

            while len(self.offline_replay_memory[0]) < self.offline_replay_memory_size:

                # Resetting the environment and starting from a random position.
                state = self.environment.reset(random_start=False)
                # One-hot encoding.
                state = [tf.keras.utils.to_categorical(state[i], self.environment.observation_space.n)
                         for i in range(self.environment.number_of_agents)]
                done = False  # Initializing the done parameter which indicates whether the environment has terminated
                              # or not to False.
                episodes += 1  # Increasing the number of episodes in this iteration.

                while not done:
                    # Selecting an action according to the predicted action probabilities.
                    action_probabilities = [self.policy_model[i].predict(
                        np.asarray(state[i]).reshape(-1, self.environment.observation_space.n))[0]
                                            for i in range(self.environment.number_of_agents)]
                    action = [np.random.choice(self.environment.action_space.n, p=action_probabilities[i])
                              for i in range(self.environment.number_of_agents)]

                    # Taking an action.
                    next_state, rewards, done, info = self.environment.step(action)
                    # One-hot encoding.
                    next_state = [tf.keras.utils.to_categorical(next_state[i], self.environment.observation_space.n)
                                  for i in range(self.environment.number_of_agents)]

                    # Incrementing the wumpus_killed counter when the agents kill the Wumpus.
                    if not self.environment.wumpus_alive:
                        wumpus_killed += 1

                    # Incrementing the total reward.
                    total_reward_iteration += sum(rewards)

                    # Appending the state, action, reward, next state and done to the replay memory.
                    for i in range(self.environment.number_of_agents):
                        self.offline_replay_memory[i].append([state[i], action[i], rewards[i], next_state[i], done])

                    state = next_state  # Setting the current state to be equal to the next state.

                    # This condition ensures that we don't append more values than the size of the replay memory.
                    if len(self.offline_replay_memory[0]) == self.offline_replay_memory_size:
                        break

            # Calculating the average reward per episode for this iteration.
            average_reward_per_episode = total_reward_iteration / episodes
            average_reward_per_episode_per_iteration.append(average_reward_per_episode)

            # Appending the cumulative reward.
            if len(cumulative_average_rewards_per_episode_per_iteration) == 0:
                cumulative_average_rewards_per_episode_per_iteration.append(average_reward_per_episode)
            else:
                cumulative_average_rewards_per_episode_per_iteration.append(
                    average_reward_per_episode + cumulative_average_rewards_per_episode_per_iteration[
                        iteration - 1])

            # Calculating the percentage of episodes in which the agent reached the Gold.
            percentage_wumpus_killed = wumpus_killed / episodes * 100
            wumpus_killed_list.append(percentage_wumpus_killed)

            # Calling the replay method.
            for i in range(self.environment.number_of_agents):
                self.replay(i)

            # Printing the policy tables every 5 iterations.
            if (iteration + 1) % 5 == 0:
                policy_table = [self.policy_model[i].predict(
                    np.asarray(test_states).reshape(-1, self.environment.observation_space.n))
                    for i in range(self.environment.number_of_agents)]
                for i in range(self.environment.number_of_agents):
                    for terminal_state in [2, 8, 10, 15]:  # The list represents terminal states.
                        policy_table[i][terminal_state] = 0
                for i in range(self.environment.number_of_agents):
                    print(f'\nPolicy table for agent {i + 1} after {iteration + 1} iterations:\n', policy_table[i])

        # Calling the plots method to plot the reward dynamics.
        self.plots(average_reward_per_episode_per_iteration,
                   cumulative_average_rewards_per_episode_per_iteration,
                   wumpus_killed_list, plot_wumpus_killed=True, iterations=True)

    def evaluate(self):
        """This method evaluates the performance of the agent after it has finished training."""

        total_steps, total_penalties = 0, 0  # Initializing the total steps taken and total penalties incurred
                                             # across all episodes.
        episodes = 100  # Number of episodes for which we are going to test the agent's performance.
        rewards_per_episode = []  # Sum of immediate rewards during the episode.

        for episode in range(episodes):
            state = self.environment.reset(random_start=False)  # Resetting the environment for every new episode.
            # One-hot encoding.
            state = [tf.keras.utils.to_categorical(state[i], self.environment.observation_space.n)
                     for i in range(self.environment.number_of_agents)]
            steps, penalties = 0, 0  # Initializing the steps taken, and penalties incurred in this episode.
            done = False  # Initializing the done parameter indicating the episode termination to be False.
            total_reward_episode = 0  # Initializing the total reward acquired in this episode to be 0.

            while not done:
                # Always choosing the greedy action.
                action = [np.argmax(self.policy_model[i].predict(
                    np.asarray(state[i]).reshape(-1, self.environment.observation_space.n))[0])
                          for i in range(self.environment.number_of_agents)]

                # Taking the greedy action.
                next_state, rewards, done, info = self.environment.step(action)
                # One-hot encoding.
                next_state = [tf.keras.utils.to_categorical(next_state[i], self.environment.observation_space.n)
                              for i in range(self.environment.number_of_agents)]

                # Adding the reward acquired on this step to the total reward acquired during the episode.
                total_reward_episode += sum(rewards)

                # Increasing the penalties incurred in this episode by checking the reward.
                for i in range(self.environment.number_of_agents):
                    if rewards[i] == -1:
                        penalties += 1

                state = next_state  # Setting the current state to the next state.

                steps += 1  # Increasing the number of steps taken in this episode.

            rewards_per_episode.append(total_reward_episode)  # Appending the reward acquired during the episode.

            # Appending the cumulative reward.
            if len(self.cumulative_rewards_evaluation) == 0:
                self.cumulative_rewards_evaluation.append(total_reward_episode)
            else:
                self.cumulative_rewards_evaluation.append(
                    total_reward_episode + self.cumulative_rewards_evaluation[episode - 1])

            total_penalties += penalties  # Adding the penalties incurred in this episode to the total penalties
                                          # across all the episodes.

            total_steps += steps  # Adding the steps taken in this episode to the total steps taken across all episodes

        # Printing some statistics after the evaluation of agent's performance is completed.
        print(f"\nEvaluation of agent's performance across {episodes} episodes:")
        print(f"Average number of steps taken per episode: {total_steps / episodes}")
        print(f"Average penalties incurred per episode: {total_penalties / episodes}")

        # Calling the plots method to plot the reward dynamics.
        self.plots(rewards_per_episode, self.cumulative_rewards_evaluation)

    def render_actions(self):
        # Rendering the actions taken by the agent after learning.
        state = self.environment.reset(random_start=False)  # Resetting the environment for a new episode.
        # One-hot encoding.
        state = [tf.keras.utils.to_categorical(state[i], self.environment.observation_space.n)
                 for i in range(self.environment.number_of_agents)]
        self.environment.render(plot=True)  # Rendering the environment.
        done = False  # Initializing the done parameter indicating the episode termination to be False.

        while not done:
            # Always choosing the greedy action.
            action = [np.argmax(self.policy_model[i].predict(
                np.asarray(state[i]).reshape(-1, self.environment.observation_space.n))[0])
                      for i in range(self.environment.number_of_agents)]

            # Taking the greedy action.
            next_state, rewards, done, info = self.environment.step(action)
            # One-hot encoding.
            next_state = [tf.keras.utils.to_categorical(next_state[i], self.environment.observation_space.n)
                          for i in range(self.environment.number_of_agents)]

            self.environment.render(plot=True)  # Rendering the environment.
            state = next_state  # Setting the current state to the next state.

    @staticmethod
    def plots(rewards_per_episode, cumulative_rewards, wumpus_killed=None, plot_wumpus_killed=False, iterations=False):
        """This method plots the reward dynamics and epsilon decay.

        :param iterations: Boolean indicating that we are plotting for iterations and not episodes.
        :param wumpus_killed: List containing the percentage of episodes in which the agent reached the Gold.
        :param plot_wumpus_killed: Boolean indicating whether of not to plot gold_reached.
        :param rewards_per_episode: List containing the reward values per episode.
        :param cumulative_rewards: List containing the cumulative reward values per episode."""

        plt.figure(figsize=(20, 10))
        plt.plot(rewards_per_episode, 'ro')
        if iterations:
            plt.xlabel('Iterations')
            plt.ylabel('Average Reward Per Episode')
            plt.title('Average Rewards Per Episode Per Iteration')
        else:
            plt.xlabel('Episodes')
            plt.ylabel('Reward Value')
            plt.title('Rewards Per Episode (During Evaluation)')
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.plot(cumulative_rewards)
        if iterations:
            plt.xlabel('Iterations')
            plt.ylabel('Cumulative Average Reward Per Episode')
            plt.title('Cumulative Average Rewards Per Episode Per Iteration')
        else:
            plt.xlabel('Episodes')
            plt.ylabel('Cumulative Reward Per Episode')
            plt.title('Cumulative Rewards Per Episode (During Evaluation)')
        plt.show()  # Displaying the plot.

        if plot_wumpus_killed:
            plt.figure(figsize=(20, 10))
            plt.plot(wumpus_killed)
            plt.xlabel('Iterations')
            plt.ylabel('Percentage')
            plt.title('Percentage of Episodes in Which the Agents Killed the Wumpus.')
            plt.show()


# Instantiating the deterministic Multi-Agent Wumpus World environment.
deterministic_multi_agent_wumpus_world_environment = MultiAgentWumpusWorldEnvironment(environment_type='deterministic')

print('\nVersion 1:\n')
AdvantageWeightedRegression(deterministic_multi_agent_wumpus_world_environment, alternate_network=False,
                            offline_memory_size=10000, iterations=10)

print('\nVersion 2:\n')
AdvantageWeightedRegression(deterministic_multi_agent_wumpus_world_environment, alternate_network=True,
                            offline_memory_size=1000, iterations=10)

print('\nVersion 3:\n')
AdvantageWeightedRegression(deterministic_multi_agent_wumpus_world_environment, alternate_network=True,
                            offline_memory_size=500, iterations=10)

# Instantiating the stochastic Multi-Agent Wumpus World environment.
stochastic_multi_agent_wumpus_world_environment = MultiAgentWumpusWorldEnvironment(environment_type='stochastic')

print('\nVersion 1:\n')
AdvantageWeightedRegression(stochastic_multi_agent_wumpus_world_environment, alternate_network=False,
                            offline_memory_size=10000, iterations=10)

print('\nVersion 2:\n')
AdvantageWeightedRegression(stochastic_multi_agent_wumpus_world_environment, alternate_network=True,
                            offline_memory_size=1000, iterations=10)

print('\nVersion 3:\n')
AdvantageWeightedRegression(stochastic_multi_agent_wumpus_world_environment, alternate_network=True,
                            offline_memory_size=500, iterations=10)
