import chess
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = chess.Board()
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)
        self.action_space = spaces.Discrete(4672)
        self.all_moves = self._generate_all_moves()


    def _generate_all_moves(self):
        moves = []
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                for promo in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    try:
                        move = chess.Move(from_square, to_square, promotion=promo)
                        chess.Board().push(move)
                        moves.append(move)
                    except:
                        continue
        return moves


    def _decode_action(self, action):
        if 0 <= action < len(self.all_moves):
            return self.all_moves[action]
        return chess.Move.null()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((8, 8, 12), dtype=np.float32)
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                idx = self._piece_to_index(piece)
                obs[sq // 8, sq % 8, idx] = 1
        return obs

    def _piece_to_index(self, piece):
        mapping = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        return mapping[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)

    def step(self, action):
        move = self._decode_action(action)

        if move not in self.board.legal_moves:
            move = random.choice(list(self.board.legal_moves))

        self.board.push(move)

        reward = 0
        done = False
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            done = True

        return self._get_obs(), reward, done, False, {}

