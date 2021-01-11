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


# Defining the Wumpus World Environment.
class WumpusWorldEnvironment(gym.Env):
    """This class implements the Wumpus World environment."""

    def __init__(self, environment_type):
        """This method initializes the environment.

        :param environment_type: - (It can take two values: 1. 'deterministic' 2. 'stochastic' indicating the type of
                                    environment.)"""

        self.environment_type = environment_type  # This indicates whether the environment is of the type deterministic
                                                  # or stochastic.
        self.observation_space = spaces.Discrete(16)  # This defines that there are 16 states in the environment.
        self.action_space = spaces.Discrete(4)  # This defines that there are 4 discrete actions that the agent can
                                                # perform.
        self.number_of_agents = 1  # This defines the number of agents in the environment.
        self.agent_pos = [0, 0]  # This defines the agent's position in the environment.
        self.breeze_pos = [[1, 0], [2, 1], [2, 3], [3, 0], [3, 2]]  # This defines the positions of breeze in the
                                                                    # environment.
        self.breeze_gold_stench_pos = [1, 2]  # This defines the position of breeze, gold, and stench in the
                                              # environment.
        self.gold_pos = [1, 2]  # This defines the position of gold in the environment.
        self.gold_quantity = 1  # This defines the quantity of gold.
        self.pit_pos = [[2, 0], [2, 2], [3, 3]]  # This defines the positions of pit in the environment.
        self.stench_pos = [[0, 1], [0, 3]]  # This defines the positions of stench in the environment.
        self.wumpus_pos = [0, 2]  # This defines the position of the Wumpus in the environment.
        self.timesteps = 0  # This defines the steps the agent has taken during an episode.
        self.max_timesteps = 10  # This defines the maximum steps the agent can take during an episode.
        self.gold_distance = self.compute_distance(self.agent_pos, self.gold_pos)  # This defines the distance of
                                                                                   # the agent from the Gold.

    def reset(self, exploring_starts=False, random_start=False):
        """This method resets the agent position and returns the state as the observation.

        :param exploring_starts: - (Boolean indicating whether the agent will start in a random position and take a
                                    random action.)
        :param random_start: - (Boolean indicating whether the agent will start in a random or fixed position.)

        :returns observation: - (Integers from 0 to 15 defining the agent's position in the environment."""

        # Creating the mapping from the possible states the agent can start in to the co-ordinates.
        coordinates_state_mapping = {12: [0, 3], 13: [1, 3], 14: [2, 3],
                                     11: [3, 2],
                                     4: [0, 1], 5: [1, 1], 6: [2, 1], 7: [3, 1],
                                     0: [0, 0], 1: [1, 0], 3: [3, 0]}

        if not random_start:
            self.agent_pos = [0, 0]  # Upon resetting the environment the agent's position is set to [0, 0].
            observation = 0  # The state mapping for position [0, 0] is '0'.
        else:
            # Randomly selecting the agent's position.
            random_state = random.choice([0, 1, 3, 4, 5, 6, 7, 11, 12, 13, 14])
            self.agent_pos = coordinates_state_mapping[random_state]
            # Setting the observation to be the state occupied by the agent.
            observation = random_state

        if exploring_starts:
            # Randomly selecting the agent's position.
            random_state = random.choice([0, 1, 3, 4, 5, 6, 7, 11, 12, 13, 14])
            self.agent_pos = coordinates_state_mapping[random_state]
            # Setting the observation to be the state occupied by the agent.
            observation = random_state
            action = self.action_space.sample()  # Randomly selecting an action.
            next_state, reward, done, info = self.step(action)  # Taking an action.
            return observation, action, reward, done, info

        self.timesteps = 0  # Resetting the number of steps taken by the agent.
        self.gold_quantity = 1  # Resetting the Gold quantity to be 1.
        self.gold_distance = self.compute_distance(self.agent_pos, self.gold_pos)  # Resetting the distance of
                                                                                   # the agent to the Gold.

        return observation

    def step(self, action):
        """This function implements what happens when the agent takes a particular action. It changes the agent's
        position (While not allowing it to go out of the environment space.), maps the environment co-ordinates to a
        state, defines the rewards for the various states, and determines when the episode ends.

        :param action: - (Integer in the range 0 to 3 inclusive.)

        :returns observation: - (Integers from 0 to 15 defining the agent's position in the environment.)
                 reward: - (Integer value that's used to measure the performance of the agent.)
                 done: - (Boolean describing whether or not the episode has ended.)
                 info: - (A dictionary that can be used to provide additional implementation information.)"""

        if self.environment_type == 'deterministic':
            # Describing the outcomes of the various possible actions.
            if action == 0:
                self.agent_pos[0] += 1  # This action causes the agent to go right.
            if action == 1:
                self.agent_pos[0] -= 1  # This action causes the agent to go left.
            if action == 2:
                self.agent_pos[1] += 1  # This action causes the agent to go up.
            if action == 3:
                self.agent_pos[1] -= 1  # This action causes the agent to go down.

        if self.environment_type == 'stochastic':
            # Describing the outcomes of the various possible actions.
            if action == 0:  # This action causes the agent to go right with 0.9 probability and remain in the same
                             # position with 0.1 probability.
                probability = random.uniform(0, 1)
                if probability > 0.1:
                    self.agent_pos[0] += 1
            if action == 1:  # This action causes the agent to go left with 0.9 probability and remain in the same
                             # position with 0.1 probability.
                probability = random.uniform(0, 1)
                if probability > 0.1:
                    self.agent_pos[0] -= 1
            if action == 2:  # This action causes the agent to go up with 0.9 probability and remain in the same
                             # position with 0.1 probability.
                probability = random.uniform(0, 1)
                if probability > 0.1:
                    self.agent_pos[1] += 1
            if action == 3:  # This action causes the agent to go down with 0.9 probability and remain in the same
                             # position with 0.1 probability.
                probability = random.uniform(0, 1)
                if probability > 0.1:
                    self.agent_pos[1] -= 1

        # Ensuring that the agent doesn't go out of the environment.
        self.agent_pos = np.clip(self.agent_pos, a_min=0, a_max=3)

        new_gold_distance = self.compute_distance(self.agent_pos, self.gold_pos)  # Computing the new distance of the
                                                                                  # agent from the Gold.

        # Giving the agent different rewards if its distance to the Gold increases, decreases or remains the same.
        if new_gold_distance > self.gold_distance:  # If the agent moves away from the Gold it gets reward -1.
            reward = -1
            self.gold_distance = new_gold_distance

        elif new_gold_distance < self.gold_distance:  # If the agent moves closer to the Gold it gets reward 1.
            reward = 1
            self.gold_distance = new_gold_distance

        else:  # If the agent's distance to the Gold doesn't change it gets reward 0.
            reward = 0

        # Creating the mapping from the co-ordinates to the state.
        coordinates_state_mapping = {'[0 3]': 12, '[1 3]': 13, '[2 3]': 14, '[3 3]': 15,
                                     '[0 2]': 8, '[1 2]': 9, '[2 2]': 10, '[3 2]': 11,
                                     '[0 1]': 4, '[1 1]': 5, '[2 1]': 6, '[3 1]': 7,
                                     '[0 0]': 0, '[1 0]': 1, '[2 0]': 2, '[3 0]': 3}

        # Setting the observation to be the state occupied by the agent.
        observation = coordinates_state_mapping[f'{self.agent_pos}']

        self.timesteps += 1  # Increasing the total number of steps taken by the agent.

        # Setting the reward to 10 if the agent reaches the gold.
        if (self.agent_pos == self.gold_pos).all() and self.gold_quantity > 0:
            self.gold_quantity -= 1
            reward = 10

        for i in range(len(self.pit_pos)):  # Setting the reward to -1 if the agent falls in the pit.
            if (self.agent_pos == self.pit_pos[i]).all():
                reward = -1

        if (self.agent_pos == self.wumpus_pos).all():  # Setting the reward to -1 if the agent is killed by Wumpus.
            reward = -1

        # The episode terminates when the agent reaches the Gold, or is killed by the Wumpus, falls into the pit, or
        # takes more than 10 steps.
        if self.gold_quantity == 0 or \
                (self.agent_pos == self.wumpus_pos).all():
            done = True
        else:
            done = False
        for i in range(len(self.pit_pos)):
            if (self.agent_pos == self.pit_pos[i]).all():
                done = True
        if self.timesteps == self.max_timesteps:
            done = True
        info = {}

        return observation, reward, done, info

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

        agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent.png'), zoom=0.36),  # Plotting the agent.
                               list(map(add, self.agent_pos, [0.5, 0.5])), frameon=False)
        ax.add_artist(agent)

        for i in range(len(self.breeze_pos)):  # Plotting the breeze.

            # Plot for when the agent is not in the same position as the breeze.
            if self.breeze_pos[i][0] != self.agent_pos[0] or self.breeze_pos[i][1] != self.agent_pos[1]:
                breeze = AnnotationBbox(OffsetImage(plt.imread('./images/breeze.png'), zoom=0.36),
                                        list(map(add, self.breeze_pos[i], [0.5, 0.5])), frameon=False)
                ax.add_artist(breeze)

            else:  # Plot for when the agent is in the same position as the breeze.
                agent_breeze = AnnotationBbox(OffsetImage(plt.imread('./images/agent_breeze.png'), zoom=0.36),
                                              list(map(add, self.breeze_pos[i], [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_breeze)

        # Plotting the breeze, gold and stench in a single position.
        if self.gold_quantity > 0:
            breeze_gold_stench = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_gold_stench.png'), zoom=0.36),
                                                list(map(add, self.breeze_gold_stench_pos, [0.5, 0.5])), frameon=False)
            ax.add_artist(breeze_gold_stench)
        else:
            breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_stench.png'), zoom=0.36),
                                                list(map(add, self.breeze_gold_stench_pos, [0.5, 0.5])), frameon=False)
            ax.add_artist(breeze_stench)

        for i in range(len(self.pit_pos)):  # Plotting the pit.

            # Plot for when the agent is not in the same position as the pit.
            if self.pit_pos[i][0] != self.agent_pos[0] or self.pit_pos[i][1] != self.agent_pos[1]:
                pit = AnnotationBbox(OffsetImage(plt.imread('./images/pit.png'), zoom=0.36),
                                     list(map(add, self.pit_pos[i], [0.5, 0.5])), frameon=False)
                ax.add_artist(pit)

            else:  # Plot for when the agent is in the same position as the pit.
                agent_dead_pit = AnnotationBbox(OffsetImage(plt.imread('./images/agent_dead_pit.png'), zoom=0.36),
                                                list(map(add, self.pit_pos[i], [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_dead_pit)

        for i in range(len(self.stench_pos)):  # plotting the stench.

            # Plot for when the agent is not in the same position as the stench.
            if self.stench_pos[i][0] != self.agent_pos[0] or self.stench_pos[i][1] != self.agent_pos[1]:
                stench = AnnotationBbox(OffsetImage(plt.imread('./images/stench.png'), zoom=0.36),
                                        list(map(add, self.stench_pos[i], [0.5, 0.5])), frameon=False)
                ax.add_artist(stench)

            else:  # Plot for when the agent is in the same position as the stench.
                agent_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_stench.png'), zoom=0.36),
                                              list(map(add, self.stench_pos[i], [0.5, 0.5])), frameon=False)
                ax.add_artist(agent_stench)

        # Plotting the Wumpus.
        # Plot for when the agent is not in the same position as the Wumpus.
        if self.agent_pos[0] != self.wumpus_pos[0] or self.agent_pos[1] != self.wumpus_pos[1]:
            wumpus = AnnotationBbox(OffsetImage(plt.imread('./images/wumpus.png'), zoom=0.36),
                                    list(map(add, self.wumpus_pos, [0.5, 0.5])), frameon=False)
            ax.add_artist(wumpus)

        else:  # Plot for when the agent is in the same position as the Wumpus.
            wumpus = AnnotationBbox(OffsetImage(plt.imread('./images/agent_dead_wumpus.png'), zoom=0.36),
                                    list(map(add, self.wumpus_pos, [0.5, 0.5])), frameon=False)
            ax.add_artist(wumpus)

        plt.xticks([0, 1, 2, 3])  # Specifying the ticks on the x-axis.
        plt.yticks([0, 1, 2, 3])  # Specifying the ticks on the y-axis.
        plt.grid()  # Setting the plot to be of the type 'grid'.
        if plot:
            plt.show()  # Displaying the plot.
        else:
            # Preprocessing the image of the environment to be used as a state representation.
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
        :param iterations: Integer indicating the number of iterations for which the agent will train."""

        self.environment = environment  # The environment which we need the agent to solve.
        self.alternate_network = alternate_network  # Boolean indicating whether to use the second deeper network.
        self.offline_replay_memory_size = offline_memory_size  # This specifies the length of the offline replay memory.
        self.offline_replay_memory = []  # Offline replay memory.
        self.iterations = iterations  # Number of episodes for which the agent will train.
        self.discount_factor = 0.99  # Discount factor determines the value of the future rewards.
        self.beta = 0.5  # Hyper-parameter used to calculate the exponential advantage.
        self.actor_model, self.critic_model, self.policy_model = self.neural_network()  # Creating the networks.
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
            common = Dense(128, activation='relu')(input_)  # Common layer for the networks.
            probabilities = Dense(self.environment.action_space.n, activation='softmax')(common)  # Actor output.
            values = Dense(1, activation='linear')(common)  # Critic output.

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

    def monte_carlo_returns(self):
        """This method calculates the Monte Carlo returns given a list of rewards."""

        rewards = [item[2] for item in self.offline_replay_memory]
        monte_carlo_returns = []  # List containing the Monte-Carlo returns.
        monte_carlo_return = 0
        t = 0  # Exponent by which the discount factor is raised.

        for i in range(len(self.offline_replay_memory)):

            while not self.offline_replay_memory[i][4]:  # Execute until you encounter a terminal state.

                # Equation to calculate the Monte-Carlo return.
                monte_carlo_return += self.discount_factor ** t * rewards[i]
                i += 1  # Go to the next sample.
                t += 1  # Increasing the exponent by which the discount factor is raised.

                # Condition to check whether we have reached the end of the replay memory without the episode being
                # terminated, and if so break. (This can happen with the samples at the end of the replay memory as we
                # only store the samples till we reach the replay memory size and not till we exceed it with the episode
                # being terminated.)
                if i == len(self.offline_replay_memory):

                    # If the episode hasn't terminated but you reach the end append the Monte-Carlo return to the list.
                    monte_carlo_returns.append(monte_carlo_return)

                    # Resetting the Monte-Carlo return value and the exponent to 0.
                    monte_carlo_return = 0
                    t = 0

                    break  # Break from the loop.

            # If for one of the samples towards the end we reach the end of the replay memory and it hasn't terminated,
            # we will go back to the beginning of the for loop to calculate the Monte-Carlo return for the future
            # samples if any for whom the episode hasn't terminated.
            if i == len(self.offline_replay_memory):
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

    def replay(self):
        """This is the replay method, that is used to fit the actor and critic networks and synchronize the weights
            between the actor and policy networks."""

        states = [item[0] for item in self.offline_replay_memory]
        states = np.asarray(states).reshape(-1, self.environment.observation_space.n)

        actions = [tf.keras.utils.to_categorical(item[1], self.environment.action_space.n).tolist()
                   for item in self.offline_replay_memory]

        monte_carlo_returns = self.monte_carlo_returns()

        critic_values = self.critic_model.predict(states).flatten()

        exponential_advantages = [np.exp(1/self.beta * (monte_carlo_returns[i] - critic_values[i]))
                      for i in range(len(self.offline_replay_memory))]

        # Fitting the actor model.
        self.actor_model.fit([states, np.asarray(exponential_advantages)], np.asarray(actions),
                             batch_size=16, epochs=1, verbose=0)

        # Syncing the weights between the actor and policy models.
        self.policy_model.set_weights(self.actor_model.get_weights())

        # Fitting the critic model.
        self.critic_model.fit(states, np.asarray(monte_carlo_returns), batch_size=16, epochs=1, verbose=0)

    def train(self):
        """This method performs the agent training."""

        # Environment states for indirectly plotting the policy table.
        test_states = [x for x in range(self.environment.observation_space.n)]
        test_states = tf.keras.utils.to_categorical(test_states)

        # Printing the initial policy table.
        policy_table = self.policy_model.predict(
            np.asarray(test_states).reshape(-1, self.environment.observation_space.n))
        for terminal_state in [2, 8, 9, 10, 15]:  # The list represents terminal states.
            policy_table[terminal_state] = 0
        print('\nInitial Policy Table:\n', policy_table)

        average_reward_per_episode_per_iteration = []
        cumulative_average_rewards_per_episode_per_iteration = []
        gold_reached = []  # List containing the percentage of episodes in which the agent reached the Gold per
                           # iteration.

        for iteration in range(self.iterations):

            self.offline_replay_memory = []  # Resetting the offline replay memory to be empty.
            total_reward_iteration = 0  # Total reward acquired in this iteration.
            gold = 0  # Initializing the number of episodes in which the agent reached the Gold to be 0.
            episodes = 0  # Initializing the number of episodes in this iteration to be 0.

            while len(self.offline_replay_memory) < self.offline_replay_memory_size:

                # Resetting the environment and starting from a random position.
                state = self.environment.reset(random_start=True)
                # One-hot encoding.
                state = tf.keras.utils.to_categorical(state, self.environment.observation_space.n)
                done = False  # Initializing the done parameter which indicates whether the environment has terminated
                              # or not to False.
                episodes += 1  # Increasing the number of episodes in this iteration.

                while not done:
                    # Selecting an action according to the predicted action probabilities.
                    action_probabilities = (self.policy_model.predict(
                        np.asarray(state).reshape(-1, self.environment.observation_space.n))[0])
                    action = np.random.choice(self.environment.action_space.n, p=action_probabilities)

                    # Taking an action.
                    next_state, reward, done, info = self.environment.step(action)
                    # One-hot encoding.
                    next_state = tf.keras.utils.to_categorical(next_state, self.environment.observation_space.n)

                    # Incrementing the Gold counter when the agent reaches the Gold.
                    if self.environment.agent_pos[0] == self.environment.gold_pos[0] and \
                            self.environment.agent_pos[1] == self.environment.gold_pos[1]:
                        gold += 1

                    # Incrementing the total reward.
                    total_reward_iteration += reward

                    # Appending the state, action, reward, next state and done to the replay memory.
                    self.offline_replay_memory.append([state, action, reward, next_state, done])

                    state = next_state  # Setting the current state to be equal to the next state.

                    # This condition ensures that we don't append more values than the size of the replay memory.
                    if len(self.offline_replay_memory) == self.offline_replay_memory_size:
                        break

            # Calculating the average reward per episode for this iteration.
            average_reward_per_episode = total_reward_iteration / episodes
            average_reward_per_episode_per_iteration.append(average_reward_per_episode)

            # Appending the cumulative reward.
            if len(cumulative_average_rewards_per_episode_per_iteration) == 0:
                cumulative_average_rewards_per_episode_per_iteration.append(average_reward_per_episode)
            else:
                cumulative_average_rewards_per_episode_per_iteration.append(
                    average_reward_per_episode + cumulative_average_rewards_per_episode_per_iteration[iteration - 1])

            # Calculating the percentage of episodes in which the agent reached the Gold.
            percentage_gold_reached = gold / episodes * 100
            gold_reached.append(percentage_gold_reached)

            # Calling the replay method.
            self.replay()

            # Printing the policy table every 5 iterations.
            if (iteration + 1) % 5 == 0:
                policy_table = self.policy_model.predict(
                    np.asarray(test_states).reshape(-1, self.environment.observation_space.n))
                for terminal_state in [2, 8, 9, 10, 15]:  # The list represents terminal states.
                    policy_table[terminal_state] = 0
                print(f'\nPolicy table after {iteration + 1} iterations:\n', policy_table)

        # Calling the plots method to plot the reward dynamics.
        self.plots(average_reward_per_episode_per_iteration, cumulative_average_rewards_per_episode_per_iteration,
                   gold_reached, plot_gold_reached=True, iterations=True)

    def evaluate(self):
        """This method evaluates the performance of the agent after it has finished training."""

        total_steps, total_penalties = 0, 0  # Initializing the total steps taken and total penalties incurred
                                             # across all episodes.
        episodes = 100  # Number of episodes for which we are going to test the agent's performance.
        rewards_per_episode = []  # Sum of immediate rewards during the episode.
        gold = 0  # Counter to keep track of the episodes in which the agent reaches the Gold.

        for episode in range(episodes):
            state = self.environment.reset(random_start=True)  # Resetting the environment for every new episode.
            # One-hot encoding.
            state = tf.keras.utils.to_categorical(state, self.environment.observation_space.n)
            steps, penalties = 0, 0  # Initializing the steps taken, and penalties incurred in this episode.
            done = False  # Initializing the done parameter indicating the episode termination to be False.
            total_reward_episode = 0  # Initializing the total reward acquired in this episode to be 0.

            while not done:
                # Always choosing the greedy action.
                action = np.argmax(self.policy_model.predict(
                    np.asarray(state).reshape(-1, self.environment.observation_space.n))[0])

                # Taking the greedy action.
                next_state, reward, done, info = self.environment.step(action)
                # One-hot encoding.
                next_state = tf.keras.utils.to_categorical(next_state, self.environment.observation_space.n)

                total_reward_episode += reward  # Adding the reward acquired on this step to the total reward acquired
                                                # during the episode.

                # Incrementing the Gold counter when the agent reaches the Gold.
                if self.environment.agent_pos[0] == self.environment.gold_pos[0] and \
                        self.environment.agent_pos[1] == self.environment.gold_pos[1]:
                    gold += 1

                if reward == -1:  # Increasing the penalties incurred in this episode by checking the reward.
                    penalties += 1

                    # If the agent gets the Gold in 100 % of the episodes along with a small average penalty value per
                    # episode (~ 0.09 (1/11 where it's the one state 'i.e., state 11' out of a total of 11 possible
                    # states the agent can start in)) it's not that the agent doesn't learn the optimal policy but it
                    # simply arises from the fact that when the agent starts in state 11 it has to go down which
                    # increases the distance to the Gold and it receives a penalty but it has still learned that getting
                    # to the Gold is the optimal action.

                    # If you want to see why we get a penalty even though the environment is solved uncomment the
                    # following lines of code:

                    # print(f'\nPenalty Acquired on: State: {np.argmax(state)}, Action: {action}, Reward: {reward}, '
                    #       f'Next State: {np.argmax(next_state)}, Done: {done}')

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
        print(f"Percentage of episodes in which the agent reaches the Gold: {(gold / episodes) * 100} %")

        # Calling the plots method to plot the reward dynamics.
        self.plots(rewards_per_episode, self.cumulative_rewards_evaluation)

    def render_actions(self):
        # Rendering the actions taken by the agent after learning.
        state = self.environment.reset(random_start=True)  # Resetting the environment for a new episode.
        # One-hot encoding.
        state = tf.keras.utils.to_categorical(state, self.environment.observation_space.n)
        self.environment.render(plot=True)  # Rendering the environment.
        done = False  # Initializing the done parameter indicating the episode termination to be False.

        while not done:
            # Always choosing the greedy action.
            action = np.argmax(self.policy_model.predict(
                np.asarray(state).reshape(-1, self.environment.observation_space.n))[0])

            # Taking the greedy action.
            next_state, reward, done, info = self.environment.step(action)
            # One-hot encoding.
            next_state = tf.keras.utils.to_categorical(next_state, self.environment.observation_space.n)

            self.environment.render(plot=True)  # Rendering the environment.
            state = next_state  # Setting the current state to the next state.

    @staticmethod
    def plots(rewards_per_episode, cumulative_rewards, gold_reached=None, plot_gold_reached=False, iterations=False):
        """This method plots the reward dynamics and epsilon decay.

        :param iterations: Boolean indicating that we are plotting for iterations and not episodes.
        :param gold_reached: List containing the percentage of episodes in which the agent reached the Gold.
        :param plot_gold_reached: Boolean indicating whether or not to plot gold_reached.
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
        plt.show()

        if plot_gold_reached:
            plt.figure(figsize=(20, 10))
            plt.plot(gold_reached)
            plt.xlabel('Iterations')
            plt.ylabel('Percentage')
            plt.title('Percentage of Episodes in Which the Agent Reached the Gold.')
            plt.show()


# Instantiating the deterministic and stochastic Wumpus World environment.
deterministic_wumpus_world_environment = WumpusWorldEnvironment(environment_type='deterministic')
stochastic_wumpus_world_environment = WumpusWorldEnvironment(environment_type='stochastic')

print('\nAdvantage Weighted Regression for deterministic Wumpus World environment:\n')

print('\nVersion 1:\n')
AdvantageWeightedRegression(deterministic_wumpus_world_environment, alternate_network=False, offline_memory_size=10000,
                            iterations=10)

print('\nVersion 2:\n')
AdvantageWeightedRegression(deterministic_wumpus_world_environment, alternate_network=True, offline_memory_size=1000,
                            iterations=5)

print('\nVersion 3:\n')
AdvantageWeightedRegression(deterministic_wumpus_world_environment, alternate_network=True, offline_memory_size=250,
                            iterations=10)

print('\nAdvantage Weighted Regression for stochastic Wumpus World environment:\n')

print('\nVersion 1:\n')
AdvantageWeightedRegression(stochastic_wumpus_world_environment, alternate_network=False, offline_memory_size=10000,
                            iterations=10)

print('\nVersion 2:\n')
AdvantageWeightedRegression(stochastic_wumpus_world_environment, alternate_network=True, offline_memory_size=1000,
                            iterations=5)

print('\nVersion 3:\n')
AdvantageWeightedRegression(stochastic_wumpus_world_environment, alternate_network=True, offline_memory_size=250,
                            iterations=10)
