from turtle import st
from sympy import flatten
import torch as torch
import numpy as np

# a board for gomoku game
class ChessBoard():
    def __init__(self, width=10, height=10) -> None:
        self.width = width
        self.height = height
        # state is a 2*width*height tensor
        # state[0] is the current player's pieces
        # state[1] is the opponent's pieces
        self.state = torch.zeros((2, self.width, self.height))
        pass

    def reset(self):
        self.state = torch.zeros((2, self.width, self.height))
        pass

    @staticmethod
    def compress_state_to_grid(state, width=10, height=10):
        grid = torch.zeros((width, height))
        grid[state[0] == 1] = 1
        grid[state[1] == 1] = -1
        return grid
    
    @staticmethod
    def get_legal_moves(state):
        # return a list of legal moves, represented by a list of indices
        grid = ChessBoard().compress_state_to_grid(state)
        # print(grid)
        flattened_grid = grid.flatten()
        # print(flattened_grid)
        # print(torch.nonzero(flattened_grid == 0).flatten())
        return (flattened_grid == 0).tolist()

    @staticmethod
    def get_next_state(state, move: int, width=10, height=10):
        # only return but not change the current state
        state_next = state.clone()
        move_x = move // width
        move_y = move % width
        state_next[0, move_x, move_y] = 1
        state_next[0], state_next[1] = torch.tensor(state_next[1].tolist()), torch.tensor(state_next[0].tolist())
        return state_next
    
    def to_next_state(self, move):
        # change the current state and return the next state
        move_x = move // self.width
        move_y = move % self.width
        self.state[0, move_x, move_y] = 1
        self.state[0], self.state[1] = torch.tensor(self.state[1].tolist()), torch.tensor(self.state[0].tolist())
        return self.state

    def check_game_over(self):
        # check for 5 in a row, column, or diagonal
        # current player
        player_board = self.state[0]
        for i in range(self.width):
            for j in range(self.height):
                if i in range(self.width - 4) and j in range(self.height - 4):
                    if torch.all(player_board[i:i+5, j:j+5].diag() == 1):
                        return True, 1
                if i in range(self.width - 4):
                    if torch.all(player_board[i:i+5, j] == 1):
                        return True, 1
                if j in range(self.height - 4):
                    if torch.all(player_board[i, j:j+5] == 1):
                        return True, 1
        # opponent
        opponent_board = self.state[1]
        for i in range(self.width):
            for j in range(self.height):
                if i in range(self.width - 4) and j in range(self.height - 4):
                    if torch.all(opponent_board[i:i+5, j:j+5].diag() == 1):
                        return True, -1
                if i in range(self.width - 4):
                    if torch.all(opponent_board[i:i+5, j] == 1):
                        return True, -1
                if j in range(self.height - 4):
                    if torch.all(opponent_board[i, j:j+5] == 1):
                        return True, -1
        # check for draw
        grid = ChessBoard().compress_state_to_grid(self.state)
        if torch.all(grid != 0):
            return True, 0
        return False, 0
    
    def get_state(self):
        return self.state

    def get_current_player(self):
        pass
    
    @staticmethod
    def is_over_state(state, w=10, h=10):
        # check for 5 in a row, column, or diagonal
        # current player
        player_board = state[0]
        for i in range(w):
            for j in range(h):
                if i in range(w - 4) and j in range(h - 4):
                    if torch.all(player_board[i:i+5, j:j+5].diag() == 1):
                        return True
                if i in range(w - 4):
                    if torch.all(player_board[i:i+5, j] == 1):
                        return True
                if j in range(h - 4):
                    if torch.all(player_board[i, j:j+5] == 1):
                        return True
        # opponent
        opponent_board = state[1]
        for i in range(w):
            for j in range(h):
                if i in range(w - 4) and j in range(h - 4):
                    if torch.all(opponent_board[i:i+5, j:j+5].diag() == 1):
                        return True
                if i in range(w - 4):
                    if torch.all(opponent_board[i:i+5, j] == 1):
                        return True
                if j in range(h - 4):
                    if torch.all(opponent_board[i, j:j+5] == 1):
                        return True
        # check for draw
        grid = ChessBoard().compress_state_to_grid(state)
        if torch.all(grid != 0):
            return True
        return False
    
    @staticmethod
    def visualize_state(s):
        output = ""
        for i in range(s.shape[1]):
            for j in range(s.shape[2]):
                if s[0, i, j] == 1:
                    output += 'X'
                elif s[1, i, j] == 1:
                    output += 'O'
                else:
                    output += '-'
            output += '\n'
        return output