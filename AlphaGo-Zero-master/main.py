import argparse

import model
from xandos import xandos
from connect4 import Connect4
from solver import Solver


if __name__ == "__main__":

	game_dict = {'xandos':(xandos(), model.XandosNet),
				 'connect3': (Connect4(board_width=5, board_height=5, win_crit=3), model.C4Net_5x5),
				 'connect4': (Connect4(board_width=7, board_height=6, win_crit=4), model.C4Net_6x7)}

	parser = argparse.ArgumentParser(description='AlphaGo Zero for X\'s and O\'s and Connect4')
	parser.add_argument('--game', type=str, default='xandos', metavar='N',
						help='Game to learn (default X\'s and O\'s')
	parser.add_argument('--num-iters', type=int, default=10, metavar='N',
						help='Number of iterations of self-play/learning cycle')
	parser.add_argument('--num-sims', type=int, default=25, metavar='N',
						help='Number of simulations performed by mcts before choosing move')
	parser.add_argument('--num-eps', type=int, default=25, metavar='N',
						help='Number of games played during self-play')
	parser.add_argument('--num-battles', type=int, default=40, metavar='N',
						help='Number of games between old and new nets')
	parser.add_argument('--num-epochs', type=int, default=40, metavar='N',
						help='Number of epochs during network training')
	parser.add_argument('--mem-len', type=int, default=40, metavar='N',
						help='Number of previous iterations used for training data')

	args = parser.parse_args()
	game, nnet_class = game_dict[args.game]


	solver = Solver(game=game,
					nnet_class=nnet_class,
					num_iters=args.num_iters,
					num_sims=args.num_sims,
					num_episodes=args.num_eps,
					num_epoch=args.num_epochs,
					num_battles=args.num_battles,
					mem_length=args.mem_len)

	solver.policy_iteration()
