# Imports
import gym
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# This ensures that all the data isn't loaded into the GPU memory at once.
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Disables eager execution.
tf.compat.v1.disable_eager_execution()


class AdvantageWeightedRegression:
    """This class implements the AWR Agent."""

    def __init__(self, environment,  alternate_network=False, offline_memory_size=10000, iterations=10):
        """This method initializes the AWR parameters, and calls the train, evaluate and render_actions methods.

        :param environment: This is the environment on which the agent will learn.
        :param alternate_network: Boolean indicating whether to use the second deeper network.
        :param offline_memory_size: Integer indicating the size of the offline replay memory.
        :param iterations: Number of iterations for which we will train."""

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

    def neural_network(self):
        """This method builds the actor, critic and policy networks."""

        if not self.alternate_network:
            # Input 1 is the one-hot representation of the environment state.
            input_ = Input(shape=(self.environment.observation_space.shape[0],))
            # Input 2 is the exponential advantage.
            exponential_advantage = Input(shape=[1])
            common = Dense(128, activation='relu')(input_)  # Common layer for the networks.
            probabilities = Dense(self.environment.action_space.n, activation='softmax')(common)  # Actor output.
            values = Dense(1, activation='linear')(common)  # Critic output.

        else:
            # Input 1 is the one-hot representation of the environment state.
            input_ = Input(shape=(self.environment.observation_space.shape[0],))
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
        states = np.asarray(states).reshape(-1, self.environment.observation_space.shape[0])

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

        average_reward_per_episode_per_iteration = []
        cumulative_average_rewards_per_episode_per_iteration = []
        max_reward_reached_list = []  # List containing the percentage of episodes in which the agent balanced the pole
                                      # for 500 time-steps per iteration.

        for iteration in range(self.iterations):

            self.offline_replay_memory = []  # Resetting the replay memory to be empty.
            total_reward_iteration = 0  # Total reward acquired in this iteration.
            max_reward_reached = 0  # Initializing the counter for number of episodes in which the agent reached the
                                    # maximum reward to be 0.
            episodes = 0  # Initializing the counter for number of episodes in this iteration to be 0.

            while len(self.offline_replay_memory) < self.offline_replay_memory_size:

                # Resetting the environment and starting from a random position.
                state = self.environment.reset()
                done = False  # Initializing the done parameter which indicates whether the environment has terminated
                              # or not to False.
                reward_episode = 0
                episodes += 1  # Increasing the number of episodes in this iteration.

                while not done:
                    # Selecting an action according to the predicted action probabilities.
                    action_probabilities = (self.policy_model.predict(
                        np.asarray(state).reshape(-1, self.environment.observation_space.shape[0]))[0])
                    action = np.random.choice(self.environment.action_space.n, p=action_probabilities)

                    # Taking an action.
                    next_state, reward, done, info = self.environment.step(action)

                    # Incrementing the reward acquired in this episode.
                    reward_episode += reward

                    # Incrementing the total reward.
                    total_reward_iteration += reward

                    # Incrementing the Gold counter when the agent reaches the Gold.
                    if reward_episode == 500:
                        max_reward_reached += 1

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
                    average_reward_per_episode + cumulative_average_rewards_per_episode_per_iteration[
                        iteration - 1])

            # Calculating the percentage of episodes in which the agent reached the maximum reward.
            percentage_max_reward_reached = max_reward_reached / episodes * 100
            max_reward_reached_list.append(percentage_max_reward_reached)

            # Calling the replay method.
            self.replay()

        # Calling the plots method to plot the reward dynamics.
        self.plots(average_reward_per_episode_per_iteration, cumulative_average_rewards_per_episode_per_iteration,
                   max_reward_reached=max_reward_reached_list, plot_max_reward_reached=True, plot_rolling_mean=False,
                   iterations=True)

    def evaluate(self):
        """This method evaluates the performance of the agent after it has finished training."""

        episodes = 100  # Number of episodes for which we are going to test the agent's performance.
        rewards_per_episode = []  # Sum of immediate rewards during the episode.
        rolling_mean_rewards = []  # Rolling mean of rewards per episode computed over 10 episodes.

        for episode in range(episodes):
            state = self.environment.reset()  # Resetting the environment for every new episode.
            done = False  # Initializing the done parameter indicating the episode termination to be False.
            total_reward_episode = 0  # Initializing the total reward acquired in this episode to be 0.

            while not done:
                self.environment.render()
                # Always choosing the greedy action.
                action = np.argmax(self.policy_model.predict(
                    np.asarray(state).reshape(-1, self.environment.observation_space.shape[0]))[0])

                # Taking the greedy action.
                next_state, reward, done, info = self.environment.step(action)

                total_reward_episode += reward  # Adding the reward acquired on this step to the total reward acquired
                                                # during the episode.

                state = next_state  # Setting the current state to the next state.

            rewards_per_episode.append(total_reward_episode)  # Appending the reward acquired during the episode.

            # Appending the cumulative reward.
            if len(self.cumulative_rewards_evaluation) == 0:
                self.cumulative_rewards_evaluation.append(total_reward_episode)
            else:
                self.cumulative_rewards_evaluation.append(
                    total_reward_episode + self.cumulative_rewards_evaluation[episode - 1])

            # Rolling mean.
            if len(rewards_per_episode) > 9:
                rolling_mean_rewards.append(np.mean(rewards_per_episode[-10:]))

        # Printing some statistics after the evaluation of agent's performance is completed.
        print(f"\nEvaluation of agent's performance across {episodes} episodes:")
        print(f"Average reward per episode: {self.cumulative_rewards_evaluation[-1] / episodes}")

        # Calling the plots method to plot the reward dynamics.
        self.plots(rewards_per_episode, self.cumulative_rewards_evaluation, rolling_mean_rewards)

    @staticmethod
    def plots(rewards_per_episode, cumulative_rewards, rolling_mean=None, plot_rolling_mean=True,
              max_reward_reached=None, plot_max_reward_reached=False, iterations=False):
        """This method plots the reward dynamics and epsilon decay.

        :param plot_rolling_mean: Boolean indicating whether or not to plot rolling_mean.
        :param plot_max_reward_reached: Boolean indicating whether or not to plot max_reward_reached.
        :param max_reward_reached: List containing the percentage of episodes in which the agent balanced the pole
                                   for the maximum time-steps.
        :param iterations: Boolean indicating that we are plotting for iterations and not episodes.
        :param rewards_per_episode: List containing the reward values per episode.
        :param cumulative_rewards: List containing the cumulative reward values per episode.
        :param rolling_mean: List containing the rolling mean reward values computed over 10 episodes."""

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

        if plot_rolling_mean:
            # Plotting the Rolling mean.
            plt.figure(figsize=(20, 10))
            plt.plot(rolling_mean)
            plt.xlabel('Episodes')
            plt.ylabel('Reward Value')
            plt.title('Rolling Mean Rewards Per Episode')
            plt.show()

        if plot_max_reward_reached:
            plt.figure(figsize=(20, 10))
            plt.plot(max_reward_reached)
            plt.xlabel('Iterations')
            plt.ylabel('Percentage')
            plt.title('Percentage of Episodes in Which the Agent Reached the Maximum Reward.')
            plt.show()


# Instantiating the CartPole environment.
cart_pole = gym.make('CartPole-v1')

print('\nVersion 1:\n')
AdvantageWeightedRegression(cart_pole, alternate_network=False, offline_memory_size=10000,
                            iterations=10)

print('\nVersion 2:\n')
AdvantageWeightedRegression(cart_pole, alternate_network=True, offline_memory_size=10000,
                            iterations=5)

print('\nVersion 3:\n')
AdvantageWeightedRegression(cart_pole, alternate_network=False, offline_memory_size=1000,
                            iterations=10)
