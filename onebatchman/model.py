import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input, Softmax, ReLU
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
from .replybuffer import ReplayBuffer

def build_dqn(lr, input_dims, dense_dims, n_actions):
  layers = []
  layers.append(Input(shape=input_dims))

  for shape in dense_dims:
    layers.append(Dense(shape))
    layers.append(Activation(relu))
    # layers.append(BatchNormalization())
  
  layers.append(Dense(n_actions))
  layers.append(Activation(relu))

  model = Sequential(layers)
  model.compile(optimizer=Adam(lr=lr), loss='mse')
  return model

class Brain(object):
  def __init__(self, alpha, gamma, batch_size, input_dims, mem_size, dnss, n_actions) -> None:
    self.n_actions = n_actions
    self.action_space = np.array([i for i in  range(self.n_actions)], dtype=np.int8)
    self.gamma = gamma
    self.epsilon = 1
    self.batch_size = batch_size
    self.epsilon_dec = 0.98
    self.epsilon_end = 0.05
    self.memory = ReplayBuffer(mem_size, input_dims, 24)
    self.network = build_dqn(alpha, input_dims, dnss, 24)
    self.history = []

  def remember(self, state, action, reward, state_, done):
    self.memory.store_transition(state, action, reward, state_, done)

  def predict(self, state):
    state = state[np.newaxis,:]
    return self.network.predict(state, verbose=0)

  def learn(self):
    if self.memory.mem_cntr > self.batch_size:
      state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

      pred = self.network.predict(state, verbose=0)
      # TODO add activation of only those that are proper moves
      next = self.network.predict(new_state, verbose=0)

      batch_index = np.arange(self.batch_size, dtype=np.int8)
      
      # print(f"state: {state}")
      # print(f"action: {action}")
      # print(f"reward: {reward}")

      pred[batch_index, action] = reward + self.gamma * next[batch_index, action] * (1 - done)

      self.history.append(self.network.fit(state, pred, verbose=0).history['loss'])

      self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end

  def save_model(self, fname):
    self.network.save(fname)

  def load_model(self, fname):
    self.network = load_model(fname)