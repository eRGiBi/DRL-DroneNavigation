import tensorflow as tf
import gym
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 3000


class ActorCritic(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.critic_linear2 = tf.keras.layers.Dense(1)

        self.actor_linear1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.actor_linear2 = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        state = tf.convert_to_tensor(tf.reshape(state, (1, 4)))
        value = self.critic_linear2(self.critic_linear1(state))

        policy_dist = self.actor_linear2(self.actor_linear1(state))

        return value, policy_dist


def a2c(env, hidden_size=64, learning_rate=0.001, max_episodes=1000, num_steps=200, gamma=0.99):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = tf.keras.optimizers.Adam(learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            print(state)
            state_array = state[0]
            state_tensor = tf.convert_to_tensor(state_array, dtype=tf.float32)
            state_tensor = tf.reshape(state_tensor, (1, -1))

            print(actor_critic(state_tensor))
            value, policy_dist = actor_critic(
                # state[0].reshape(1, -1).astype(np.float32)
                state_tensor)
            value = value.numpy()[0, 0]
            dist = policy_dist.numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = tf.math.log(policy_dist[0, action])
            entropy = -tf.reduce_sum(tf.reduce_mean(dist) * tf.math.log(dist))
            new_state, reward, done, _ = env.step(action)[:4]

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            state = new_state

            if done or steps == num_steps - 1:
                Qval, _ = actor_critic(new_state.reshape(1, -1).astype(np.float32))
                Qval = Qval.numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    print("episode: {}, reward: {}, total length: {}, average length: {}".format(
                        episode, np.sum(rewards), steps, average_lengths[-1]))
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + gamma * Qval
            Qvals[t] = Qval

        # update actor critic
        values = tf.convert_to_tensor(values, dtype=tf.float32)
        Qvals = tf.convert_to_tensor(Qvals, dtype=tf.float32)
        log_probs = tf.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage)
        critic_loss = 0.5 * tf.square(advantage)
        entropy_term = 0.001 * entropy

        ac_loss = tf.reduce_mean(actor_loss + critic_loss + entropy_term)

        gradients = ac_optimizer.get_gradients(ac_loss, actor_critic.trainable_variables)
        ac_optimizer.apply_gradients(zip(gradients, actor_critic.trainable_variables))

    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()


# Example usage:
# import gym
# env = gym.make('CartPole-v1')
# a2c(env)


if __name__ == "__main__":
    # print(gym.envs.registry.keys() )
    env = gym.make("CartPole-v0")
    a2c(env)
