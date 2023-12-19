from collections import deque
import pickle
from tracemalloc import start
from tqdm import tqdm
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from board import ChessBoard
from networks import Network
from dataset import GameDataset
import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
cuda_id = 2
device = torch.device("cuda:" + str(cuda_id) if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_num_threads(32)

class GomokuModel:
    def __init__(self, size):
        self.size = size
        self.net = Network().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        pass

    def loss(self, p, v, p_target, v_target):
        # loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # v is the value output from forward(), v_target (z) is the target value from self-play
        loss_v = self.mse(v, v_target)
        # p is the policy output from forward(), p_target (pi) is the target policy from self-play
        loss_p = self.kl(torch.log(p), p_target)
        # c is the regularization constant, theta is the model parameters
        c = 0.0001
        parameters = torch.cat([param.flatten() for param in self.net.parameters()])
        loss_norm = c * torch.norm(parameters, p=2)
        # the loss function is a combination of mean squared error for the value and cross entropy loss for the policy
        loss = loss_v - loss_p + loss_norm
        return loss

    def predict(self, state):
        # unsqueeze(0) to add a batch dimension
        with torch.no_grad():
            state = state.requires_grad_(False)
            return self.net(state.unsqueeze(0).to(device))

    def update(self, game):
        # game is a list of [s_t, pi_t, z_t]
        # sample a batch of 32 games from the game list, use dataloader to load the batch
        batch_size = 32
        dataset = GameDataset(game)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # for each batch, perform a forward pass, compute the loss, and perform a backward pass
        for s, pi, z in loader:
            s = s.to(device)
            pi = pi.to(device)
            z = z.to(device)
            p, v = self.net(s)
            loss = self.loss(p, v, pi, z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        pass

    def check_model_size(self):
        num_params = 0
        for i, param in enumerate(self.net.parameters()):
            num_params += param.numel()
        total_bytes = sum(p.numel() * p.element_size() for p in self.net.parameters())
        total_megabytes = total_bytes / 1024 ** 2
        print(f"Number of parameters: {num_params}, Memory size: {total_megabytes:.4f} MB")
        
class AlphaGoZero:
    def __init__(        
            self, 
            board_size: int, 
            num_iterations: int,
            num_self_play: int,
            num_simulations: int, 
            temperature: float
        ):
        self.board_size: int = board_size
        self.num_iterations: int = num_iterations
        self.num_self_play_games: int = num_self_play
        self.num_self_play_memory: int = num_self_play
        self.num_simulations: int = num_simulations
        self.temperature: float = temperature
        self.board = ChessBoard(board_size, board_size)
        self.model = GomokuModel(board_size)

    def pretrain(self):
        with open('games_aug.pickle', 'rb') as f:
            pretrain_games = pickle.load(f)
        pretrain_games = list(zip(*pretrain_games.values()))
        self.model.update(pretrain_games)
        # save the pretrain model
        torch.save(self.model.net.state_dict(), 'model/model_pretrain.pt')
        # load from pretrained model
        self.model.net.load_state_dict(torch.load("model/model_pretrain.pt"))

    def train(self):
        # init log
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))
        logging.basicConfig(filename=f'log/{now}.log', level=logging.DEBUG)
        logging.info('Start training')
        logging.info(f'board_size: {self.board_size}, num_iterations: {self.num_iterations}, num_self_play: {self.num_self_play_games}, num_simulations: {self.num_simulations}, temperature: {self.temperature}')
        # model already initialized to random weights theta_0
        self.model.check_model_size()
        for i in range(self.num_iterations):
            logging.info(f'Iteration: {i}')
            # at each iteration, games of self-play are generated
                # use deque for possiblly 'recent' games
                # but currently game played = game stored
            games = deque(maxlen=self.num_self_play_memory) 
            for g in range(self.num_self_play_games):
                logging.info('________________________________________________________')
                logging.info(f'Self-play game {g} of {self.num_self_play_games-1}, iteration {i} of {self.num_iterations-1}')
                start_time = time.time()
                logging.info(f'self-play starts at time {time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_time))}')
                game = self.self_play(i, g)
                end_time = time.time()
                time_elapsed_in_hours = (end_time - start_time) / 3600
                logging.info(f'self-play ends at time {time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end_time))}, time elapsed: {time_elapsed_in_hours:.2f} hours')
                logging.info('========================================================')
                games.append(game)
                # iterate games deque, and concatenate each game (a list of s, pi, z) into a single list
            game_positions = []
            for game in games:
                game_positions += game
            # train the model using the game positions
            logging.info(f'Train the model using {len(game_positions)} game positions in iteration {i}')
            self.model.update(game_positions)
            # TODO: evaluate the model by playing against the previous version of the model
            # Save the trained model
            logging.info(f'Save the trained model to model/model_{i}.pth')
            torch.save(self.model.net.state_dict(), f"model/model_{i}.pth")

    def self_play(self, iter, g_id) -> list:
        # game = [s_t, pi_t, z_t]
        # s_t is the board state at timestep t, represented as a 10*10*2 tensor
        # pi_t is the policy at timestep t, represented as a 100*1 tensor
        # z_t is the game result at timestep T, represented as a scalar
        
        # initialize game
        self.board.reset()
        s = self.board.get_state()
        game = []
        winner = torch.tensor(0, dtype=torch.float32)
        # at each timestep t, perform a MCTS, max is reached when board full, break if wins
        for t in range(self.board_size * self.board_size):
            print(f"Iteration {iter}, game {g_id}: Step (board state) {t} of Self-play (max {self.board_size**2} in total), use MCTS to simulate {self.num_simulations} times for each move: ")
            logging.info(f'Iter {iter}, game {g_id}, board state {t}: \n{ChessBoard().visualize_state(s)}')
            # Perform Monte Carlo Tree Search
            pi, max_tree_depth, has_an_over_state = self.MCTS(s)
            print(f"... with max search depth: {max_tree_depth} and has an over state: {has_an_over_state}")
            print("--------------------------------------------------")
            # sample a move from the policy searched by MCTS
            move = torch.multinomial(pi, 1).item()
            logging.info(f'move: {move} from board state {t}')
            # add this move to the current game
            game.append([s.clone(), pi, winner])
            # update board state
            s = self.board.to_next_state(move)
            # check if game is finished
            is_game_over, r = self.board.check_game_over()
            # in fact, r is always -1 when game is over (if not 0 for draw)
            # because after A moves, to_next_state makes it B's turn, and B observes the game result
            if is_game_over:
                logging.info(f'Iter {iter}, game {g_id}, game over at board state {t+1}: \n{ChessBoard().visualize_state(s)}')
                # set game result recursively
                for reverse_g in range(len(game)-1, -1, -1):
                    game[reverse_g][2] = torch.tensor(r, dtype=torch.float32)
                    # invert the game result for the previous player
                    r = -r
                break
        return game
    
    def MCTS(self, state) -> tuple[torch.Tensor, int, bool]:
        # perform MCTS from current state and return the policy from this state
        root = MCTS_node(state, None, None, self.board_size**2, self.model)
        max_tree_depth = 0
        has_an_over_state = False
        for t in tqdm(range(self.num_simulations)):
            node = root
            # select
            select_depth = 0
            while not node.is_leaf():
                select_depth += 1
                node = node.select()
            # expand
            if ChessBoard().is_over_state(node.state):
                has_an_over_state = True
            else:
                node.expand()
            # backup
            while not node.is_root():
                node = node.backup()
            max_tree_depth = max(max_tree_depth, select_depth)  
            pass
        pi = root.N / torch.sum(root.N)
        return pi, max_tree_depth, has_an_over_state

class MCTS_node:
    def __init__(
            self, 
            state: torch.Tensor, 
            parent: Optional['MCTS_node'] | None, 
            child_id: Optional[int] | None, 
            children_width: int,
            model: GomokuModel, 
            C_puct=1.0
        ):
        self.state: torch.Tensor = state # board state
        self.parent: MCTS_node | None = parent # parent node
        self.child_id: int | None = child_id # the move that leads to this node
        self.children_width: int = children_width # the width of the children list (=possible moves)
        self.model: GomokuModel = model # the model of the latest neural network
        self.legal_moves = ChessBoard().get_legal_moves(self.state) # binary vector of legal moves
        self.children: list[MCTS_node | None] = [] # child nodes
        # aligned to children, N, W, Q, P are considered as stored on the edges, while V is considered as stored on this node
        self.N = torch.zeros(self.children_width) # visit count
        self.W = torch.zeros(self.children_width) # total action value
        self.Q = torch.zeros(self.children_width) # mean action value
        self.P, self.V = self.model.predict(self.state) # prior probability for each move, and value of the current state
        self.P, self.V = self.P.cpu(), self.V.cpu()
        self.C_puct = C_puct

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent == None

    def select(self) -> Optional['MCTS_node']:
        # select the child node with the highest UCB score
        U = self.C_puct * self.P * torch.sqrt(torch.sum(self.N)) / (1 + self.N)
        Q = self.Q
        # illegal moves are set to -inf, so they will not be selected
        penalty = torch.Tensor([0. if self.legal_moves[i] else -float('inf') for i in range(self.children_width)])
        next_move = torch.argmax(Q + U + penalty)
        assert self.legal_moves[next_move]
        return self.children[next_move]

    def expand(self):
        # child nodes
        self.children: list[MCTS_node | None] = [None] * self.children_width 
        # evaluate the policy and value at the current node
        pi, v = self.model.predict(self.state)
        self.P = pi.cpu()
        self.V = v.cpu()
        # N, W, Q are already initialized to 0 when the leaf node is created
        # add a child node for each valid move
        for i in range(self.children_width):
            if self.legal_moves[i]:
                self.children[i] = MCTS_node(ChessBoard().get_next_state(self.state, i), self, i, self.children_width, self.model)
            else:
                self.children[i] = None

    def backup(self):
        parent = self.parent
        # update N, W, Q
        parent.N[self.child_id] += 1
        parent.W[self.child_id] += self.V
        parent.Q[self.child_id] = parent.W[self.child_id] / parent.N[self.child_id]
        return parent

alpha_go_zero = AlphaGoZero(
    board_size=10, 
    num_iterations=100,
    num_self_play=5,
    num_simulations=1600, 
    temperature=1.0)
# alpha_go_zero.pretrain()
alpha_go_zero.train()
