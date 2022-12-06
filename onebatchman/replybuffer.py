import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)

        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        # print(self.get_transition(index))
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        max_batch = min(max_mem, batch_size)
        batch = np.random.choice(max_mem, max_batch, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


    def print_buffer(self):
        for index in range(self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size):
            print(self.transition_rep(index))

    def print_buffer(self, filename):
        with open(filename, 'w') as filehandler:
            for index in range(self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size):
                filehandler.write(self.get_transition(index))

    def get_transition(self, index):
        ester = '\n'
        ester += f"Transition no.: {index}\n"
        ester += str(self.state_memory[index]) + "\n"
        ester += str(self.new_state_memory[index]) + "\n"
        ester += str(self.action_memory[index]) + "\n"
        ester += str(self.reward_memory[index]) + "\n"
        ester += str(self.terminal_memory[index]) + "\n"
        ester += '\n'
        return ester