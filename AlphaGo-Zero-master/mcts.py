from collections import defaultdict

import numpy as np
import torch


class MCTS():

	def __init__(self, game=None, net=None, num_actions=None, num_sims=25, c_puct=1):
		self.game = game
		self.num_actions = self.game.num_actions
		self.N = defaultdict(lambda: defaultdict(int))
		self.Q = defaultdict(lambda: defaultdict(int))
		self.P = defaultdict(np.array)
		self.tree = []
		self.terminal_states = {}
		self.c_puct = c_puct
		self.num_sims = num_sims
		self.nnet = net


	def get_action_probabilities(self, state, t=0):
		state_id = self.game.hash(state)
		counts = [self.N[state_id][action] for action in range(self.num_actions)]
		
		if t == 0:
			move_probs = np.zeros(self.num_actions)
			move_probs[np.argmax(counts)] = 1

		else:
			move_probs = np.array([(x**(1/t)/np.sum(counts)) for x in counts])

		return move_probs

	
	def choose_action(self, state):
		for _ in range(self.num_sims):
			self.search(state.copy())
		
		pi = self.get_action_probabilities(state)
		action = np.random.choice(self.num_actions, p=pi)
		
		return action


	def U(self, state_id, action):
		n_factor =  np.sqrt(sum((b + 1e-8) for k, b in self.N[state_id].items())) / (self.N[state_id][action] + 1)
		return self.Q[state_id][action] + self.c_puct * self.P[state_id][action]*(n_factor)
		


	def search(self, state):

		state_id = self.game.hash(state)

		##################
		# Terminal nodes #
		##################

		if state_id in self.terminal_states:
			return self.terminal_states[state_id]

		reward = -self.game.reward(state)
		if reward != 999:
			self.terminal_states[state_id] = reward
			return reward

		##############
		# leaf nodes #
		##############

		if state_id not in self.tree:
			
			self.tree.append(state_id)
			{self.Q[state_id][action]:0 for action in range(self.num_actions)}
			{self.N[state_id][action]:0 for action in range(self.num_actions)}


			pi, v = self.nnet(torch.FloatTensor(state).view(1, 1, self.game.board_height, self.game.board_width))
			pi, v = pi.data.numpy()[0], v.data.numpy()[0][0]
			
			self.P[state_id] = pi * self.game.get_valid_moves(state)
			return -v

		##################
		# explored nodes #
		##################

		else:
			valid_mask = np.invert(self.game.get_valid_moves(state))*-999
			us = [self.U(state_id, action) for action in range(self.num_actions)] + valid_mask

			action = np.argmax(us)
			state, _ = self.game.next_state(state, action=action)
			
			v = self.search(state)

			self.Q[state_id][action] = (self.N[state_id][action] * self.Q[state_id][action] + v) / (self.N[state_id][action] + 1)
			self.N[state_id][action] += 1

			return -v
