import chess
from chess_env import ChessEnv
from stable_baselines3 import PPO

env = ChessEnv()
model = PPO.load("ppo_chess")

board = chess.Board()

while not board.is_game_over():
    print(board)

    if board.turn == chess.WHITE:
        # حرکت عامل
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        move = list(board.legal_moves)[action % len(list(board.legal_moves))]
        board.push(move)
        print(f"🤖 Agent plays: {move}")
    else:
        # حرکت انسان
        user_move = input("حرکت خود را بده (مثل e2e4): ")
        try:
            board.push_uci(user_move)
        except:
            print("❌ حرکت نامعتبر!")

print("🏁 نتیجه بازی:", board.result())
