import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Activation, Flatten, BatchNormalization, Input
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from .replybuffer import ReplayBuffer


def build_dqn(lr, input_dims, conv_dims, dense_dims, n_actions=24):
  layers = []
  layers.append(Input(shape=input_dims))

  for i in range(0, len(conv_dims)):
    layers.append(Conv1D(conv_dims[i], 4, strides=2, padding='same'))
    layers.append(Activation(relu))

  layers.append(Flatten())
  for shape in dense_dims:
    layers.append(Dense(shape))
    layers.append(Activation(relu))
    layers.append(BatchNormalization())
  
  layers.append(Dense(n_actions))
  layers.append(Activation(softmax))

  model = Sequential(layers)

  model.compile(optimizer=Adam(lr=lr), loss='mse')
  return model


class Brain(object):
  def __init__(self, alpha, gamma, epsilon, batch_size, input_dims=(3*24 + 4),
              epsilon_dec=0.996, epsilon_end=0.01, mem_size=100, 
              replace_target=10) -> None:
    self.n_actions = 24
    self.action_space = [i for i in  range(self.n_actions)]
    self.gamma = gamma
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.epsilon_dec = epsilon_dec
    self.epsilon_end = epsilon_end
    self.replace_target = replace_target
    self.memory = ReplayBuffer(mem_size, input_dims, 24)
    convs = []
    dnss = [256, 128, 64, 32]
    self.q_eval = build_dqn(alpha, input_dims, convs, dnss, 24)
    self.q_target = build_dqn(alpha, input_dims, convs, dnss, 24)
    self.history = []

    self.actions_frequency = np.zeros(24, dtype=np.int32)


  def remember(self, state, action, reward, state_, done):
    self.memory.store_transition(state, action, reward, state_, done)


  def predict(self, state):
    state = state[np.newaxis,:]
    rand = np.random.random()
    if rand < self.epsilon:
      choice = np.random.choice(self.action_space)
      action = np.zeros(self.n_actions)
      action[choice] = 1
    else:
      action = self.q_eval.predict(state, verbose=0)
      i = np.argmax(action)
      self.actions_frequency[i] += 1

    return action


  def learn(self):
    if self.memory.mem_cntr > self.batch_size:
      state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

      action_values = np.array(self.action_space, dtype=np.int8) 
      action_indices = np.dot(action, action_values)

      q_next = self.q_target.predict(new_state, verbose=0)
      q_eval = self.q_eval.predict(new_state, verbose=0)

      q_pred = self.q_eval.predict(state, verbose=0)

      max_actions = np.argmax(q_eval, axis=1)

      q_target = q_pred

      batch_index = np.arange(self.batch_size, dtype=np.int8)

      q_target[batch_index, action_indices] = reward + self.gamma * q_next[batch_index, max_actions.astype(int)] * done

      self.history.append(self.q_eval.fit(state, q_target, verbose=0).history['loss'])

      self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end
      
      if self.memory.mem_cntr % self.replace_target == 0:
        self.update_network_parameters()


  def update_network_parameters(self):
    self.q_target.set_weights(self.q_eval.get_weights())


  def save_model(self, fname):
    self.q_eval.save(fname)


  def load_model(self, fname):
    self.q_eval = load_model(fname)

    if self.epsilon <= self.epsilon_end:
      self.update_network_parameters()  