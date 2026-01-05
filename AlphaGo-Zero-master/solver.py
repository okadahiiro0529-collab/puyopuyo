import numpy as np
import torch

from mcts import MCTS
from NnetHelper import NnetHelper


class Solver():

	def __init__(self, game=None, nnet_class=None, num_sims=25, num_iters=5, num_battles=40, num_episodes=25, num_epoch=10, mem_length=2):
		self.game = game
		self.nn_class = nnet_class
		self.nnet_helper = NnetHelper(num_epoch=num_epoch)
		
		self.num_sims = num_sims
		self.num_iters = num_iters
		self.num_battles = num_battles
		self.num_episodes = num_episodes
		self.training_mem_length = mem_length
		
		self.win_threshold = 0.5
		self.temp_threshold = 8

		self.num_inputs = game.starting_board.size
		self.num_actions = game.num_actions


	def policy_iteration(self):
		
		total_training_examples = []
		nnet = self.nn_class(num_inputs=self.num_inputs, num_actions=self.num_actions)
		
		for i in range(self.num_iters):
			
			print('\niteration: ', i)
			
			print('running self play...')
			train_examples = [self.execute_episode(nnet) for _ in range(self.num_episodes)]
			total_training_examples.append(train_examples)
			
			if len(total_training_examples) > self.training_mem_length:
				total_training_examples.pop(0)
			
			print('training network...')
			new_nnet = self.nn_class(num_inputs=self.num_inputs, num_actions=self.num_actions)
			new_nnet = self.nnet_helper.train_network(new_nnet, total_training_examples)

			print('battling...')

			if self.battle(nnet, new_nnet) > self.win_threshold:
				print('new network!')
				nnet = new_nnet

			self.nnet_helper.save_network(nnet, folder='./models', filename='{}_{}.pth.tar'.format(type(self.game).__name__, i))


	def execute_episode(self, nnet):

		examples = []
		mcts = MCTS(game=self.game, net=nnet)
		state = self.game.starting_board
		curr_player = 1
		num_moves = 0
		temperature = 1

		while True:
			for i in range(self.num_sims):
				mcts.search(state.copy())

			if num_moves > self.temp_threshold:
				temperature = 0
			pi = mcts.get_action_probabilities(state, t=temperature)

			for sym_board, sym_pi in self.game.get_symmetries(state, pi):
				examples.append((sym_board, sym_pi, curr_player))

			action = np.random.choice(pi.size, p=pi)
			state, curr_player = self.game.next_state(state, action=action, player=curr_player)

			reward = self.game.reward(state)
			if reward != -999:
				return [(state, pi, reward*((-1) ** (player != curr_player))) for state, pi, player in examples]

			num_moves += 1

	
	def battle(self, champion, challenger):

		results = [0,0,0]

		half_battles = self.num_battles // 2

		for _ in range(half_battles):
			result = self.single_match(champion, challenger, first_player=1)
			results[result] += 1

		for _ in range(half_battles):
			result = self.single_match(challenger, champion, first_player=2)
			results[result] += 1

		print('results: ', results)
		if results[1] + results[2] == 0:
			return self.win_threshold + 1

		return results[2] / (results[1] + results[2])

	
	def single_match(self, first_net, second_net, first_player=1):

		first_net = MCTS(game=self.game, net=first_net)
		second_net = MCTS(game=self.game, net=second_net)

		player = first_net
		curr_player = first_player
		
		state = self.game.starting_board
		results = {0:0, 1:2, 2:1}
		
		while True:

			action = player.choose_action(state)

			state, curr_player = self.game.next_state(state, action, curr_player)
			reward = self.game.reward(state)
			player = first_net if player == second_net else second_net

			if reward != -999:
				return results[curr_player*(reward != 0)]
