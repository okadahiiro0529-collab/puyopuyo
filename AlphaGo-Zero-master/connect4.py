import numpy as np

class Connect4():

	def __init__(self, board_width=None, board_height=None, win_crit=None):
		self.move_mask = np.append(1, np.zeros(board_height))
		self.board_width = board_width
		self.board_height = board_height
		self.starting_board = np.zeros((board_height, board_width))
		self.num_actions = board_width
		self.win_mask = [-1 for _ in range(win_crit)]
		self.win_crit = win_crit

	
	def hash(self, board):
		return hash(board.tostring())

	
	def next_state(self, board, action=None, player=None):
		board = board.copy()
		board = np.flip(board, axis=0)
		np.place(board[:, action], board[:, action] == 0, self.move_mask)
		board = np.flip(board, axis=0)
		player = 1 if player == 2 else 2
		return board*-1, player


	def get_valid_moves(self, board):
		return np.isin(board[0, :], 0)

	
	def print_board(self, board):
		if np.sum(board) == -1:
			display_key = {'-1.0':'X', '1.0':'O', '0.0':'-', '-0.0':'-'}
		else:
			display_key = {'-1.0':'O', '1.0':'X', '0.0':'-', '-0.0':'-'}

		string_board = board.astype(str)
		for k, v in display_key.items(): string_board[string_board==k] = v

		if self.num_actions == 7:
			self.print_board_7(string_board)

		else:
			self.print_board_5(string_board)

	def print_board_7(self, board):
		print(("|| {} | {} | {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} | {} | {} ||\n"
			   "==================================\n"
			   "   0   1   2   3   4   5   6  "
			  ).format(*board.flatten())
			  )

	def print_board_5(self, board):
		print(("|| {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} ||\n"
			   "|| {} | {} | {} | {} | {} ||\n"
			   "=======================\n"
			   "   0   1   2   3   4"
			  ).format(*board.flatten())
			  )

	
	def get_symmetries(self, board, pi):
		board_pis = []
		board_pis.append((board, pi))
		board_pis.append((np.flip(board, axis=1), np.flip(pi, axis=0)))
		return board_pis

	def reward(self, board):
		game_over = True
		for subsquare in self.subsquares(board):
			ss_r = self.subsquare_reward(subsquare)
			if ss_r == -1:
				return -1
			if ss_r != 0:
				game_over = False

		if game_over:
			return 0

		return -999

	def subsquare_reward(self, board):
		test_board = board.copy()

		for _ in range(2):
			if any((test_board[:]==self.win_mask).all(1)):
				return -1

			if np.all(np.diagonal(test_board) == -1):
				return -1

			test_board = np.rot90(test_board)
		
		if 0 not in test_board and -0 not in test_board:
			return 0

		return -999

	def subsquares(self, board):
		ssqs = []
		for i in range(self.board_height - self.win_crit + 1):
			for j in range(self.board_width - self.win_crit + 1):
				ssqs.append(board[i:i+self.win_crit, j:j+self.win_crit])
		return ssqs
