import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

class PolicyGradient:
    def __init__(self, n_x, n_y, learning_rate=0.01):
        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.build_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.states, self.actions, self.rewards = [], [], []

    def build_network(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.n_x, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.n_y, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))


    def choose_action(self, state):
        state = state.reshape([1, self.n_x])
        probabilities = self.model.predict(state)[0]
        action = np.random.choice(range(self.n_y), p=probabilities)
        return action

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        # Umwandeln der Listen in Numpy-Arrays für das Training
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)

        # One-hot encoding der Aktionen
        actions_onehot = np.zeros([len(actions), self.n_y])
        actions_onehot[np.arange(len(actions)), actions] = 1

        # Discounted rewards berechnen
        discounted_rewards = self.discount_rewards(rewards)

        # Normalisieren der discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)

        # Training des Netzwerks
        with tf.GradientTape() as tape:
            logits = self.model(states)
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actions_onehot)
            loss = tf.reduce_mean(neg_log_prob * discounted_rewards)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Leeren der Listen nach dem Training
        self.states, self.actions, self.rewards = [], [], []

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * 0.99 + rewards[t]
            discounted_rewards[t] = running_add
       
        # Normalisieren der discounted rewards
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards)
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-8)
        return discounted_rewards

class PolicyGradientAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.model.predict(state)[0]
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_state = np.reshape(next_state, [1, self.state_size])
            target = reward + np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(np.reshape(state, [1, self.state_size]))
        target_f[0][action] = target
        self.model.fit(np.reshape(state, [1, self.state_size]), target_f, epochs=1, verbose=0)
