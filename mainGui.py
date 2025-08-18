import pygame
import chess
import numpy as np
from chess_env import ChessEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys
import random
import imageio

frames = []

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Constants
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
FPS = 30
PANEL_WIDTH = 160

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
HIGHLIGHT_YELLOW = (255, 255, 0, 120)
HIGHLIGHT_GREEN = (144, 238, 144, 120)
HIGHLIGHT_RED = (255, 0, 0, 120)

# Create screen
screen = pygame.display.set_mode((WIDTH + PANEL_WIDTH, HEIGHT))
pygame.display.set_caption("Chess")

# Load sounds
def load_sounds():
    sounds = {}
    sound_files = {
        'move': 'sounds/move.wav',
        'capture': 'sounds/capture.wav',
        'win': 'sounds/win.wav',
        'loss': 'sounds/loss.wav',
        'draw': 'sounds/draw.wav',
        'click': 'sounds/click.wav'
    }
    for name, path in sound_files.items():
        if os.path.exists(path):
            sounds[name] = pygame.mixer.Sound(path)
        else:
            sounds[name] = pygame.mixer.Sound(buffer=bytearray([128] * 44100))
    return sounds

# Load piece images
def load_piece_images(theme="classic"):
    pieces = {}
    for color in ['w', 'b']:
        for piece in ['p', 'n', 'b', 'r', 'q', 'k']:
            name = color + piece
            path = f"assets/{theme}/{name}.png"
            if os.path.exists(path):
                pieces[name] = pygame.transform.scale(
                    pygame.image.load(path), (SQUARE_SIZE, SQUARE_SIZE)
                )
            else:
                surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                surf.fill((255, 0, 0, 100) if color == 'w' else (0, 0, 255, 100))
                pieces[name] = surf
    return pieces

# Load backgrounds - only specific ones
def load_backgrounds():
    backgrounds = {}
    bg_folder = "backgrounds"
    allowed = ["glass", "green", "normal", "wood"]
    if os.path.exists(bg_folder):
        for file in os.listdir(bg_folder):
            name = os.path.splitext(file)[0]
            if name.lower() in allowed and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(bg_folder, file)
                backgrounds[name] = pygame.transform.scale(
                    pygame.image.load(path), (WIDTH, HEIGHT)
                )
    return backgrounds

# Draw board
def draw_board(screen, board, images, selected=None, legal_moves=[], last_move=None, background=None, skip_square=None):
    if background:
        screen.blit(background, (0, 0))
    else:
        colors = [pygame.Color("white"), pygame.Color("gray")]
        for rank in range(8):
            for file in range(8):
                base_color = colors[(rank + file) % 2]
                pygame.draw.rect(screen, base_color, pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)
            overlay = None
            if selected == square:
                overlay = HIGHLIGHT_YELLOW
            elif square in legal_moves:
                overlay = HIGHLIGHT_GREEN
            elif last_move and (square == last_move.from_square or square == last_move.to_square):
                overlay = HIGHLIGHT_RED
            if overlay:
                surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                surface.fill(overlay)
                screen.blit(surface, (file * SQUARE_SIZE, rank * SQUARE_SIZE))
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)
            if skip_square is not None and square == skip_square:
                continue
            piece = board.piece_at(square)
            if piece:
                img_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().lower()
                screen.blit(images[img_key], pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Animate piece movement (remove origin piece first)
def animate_piece_move(screen, board, images, move, background):
    start_file = chess.square_file(move.from_square)
    start_rank = 7 - chess.square_rank(move.from_square)
    end_file = chess.square_file(move.to_square)
    end_rank = 7 - chess.square_rank(move.to_square)
    dx = (end_file - start_file) * SQUARE_SIZE
    dy = (end_rank - start_rank) * SQUARE_SIZE
    piece = board.piece_at(move.from_square)
    if not piece:
        return
    img_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().lower()
    piece_image = images[img_key]
    frames = 10
    clock = pygame.time.Clock()
    for i in range(1, frames + 1):
        screen.fill((30, 30, 30))
        board_surface = screen.subsurface((0, 0, WIDTH, HEIGHT))
        draw_board(board_surface, board, images, background=background, skip_square=move.from_square)
        x = start_file * SQUARE_SIZE + dx * (i / frames)
        y = start_rank * SQUARE_SIZE + dy * (i / frames)
        board_surface.blit(piece_image, (x, y))
        pygame.display.flip()
        clock.tick(FPS)

def get_square_from_mouse(pos):
    x, y = pos
    if x >= WIDTH:
        return None
    return chess.square(x // SQUARE_SIZE, 7 - (y // SQUARE_SIZE))

def choose_promotion(screen, images, color):
    font = pygame.font.SysFont(None, 36)
    pieces = ['q', 'r', 'b', 'n']
    piece_names = {'q': 'Queen', 'r': 'Rook', 'b': 'Bishop', 'n': 'Knight'}
    selected = 0
    while True:
        screen.fill(DARK_GRAY)
        title = font.render("Choose Promotion Piece", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
        for i, p in enumerate(pieces):
            img_key = ('w' if color == chess.WHITE else 'b') + p
            img = images[img_key]
            rect = img.get_rect(center=(100 + i * 150, HEIGHT // 2))
            screen.blit(img, rect)
            name_text = font.render(piece_names[p], True, WHITE)
            screen.blit(name_text, (rect.centerx - name_text.get_width() // 2, rect.bottom + 10))
            if i == selected:
                pygame.draw.rect(screen, WHITE, rect.inflate(20, 20), 3)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    selected = (selected - 1) % len(pieces)
                elif event.key == pygame.K_RIGHT:
                    selected = (selected + 1) % len(pieces)
                elif event.key == pygame.K_RETURN:
                    return {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}[pieces[selected]]

def show_settings(screen, backgrounds):
    font = pygame.font.SysFont(None, 40)
    themes = ["classic", "wood", "glass"]
    bg_names = list(backgrounds.keys()) if backgrounds else ["None"]
    selected_theme = "classic"
    selected_bg = "None"
    while True:
        screen.fill(DARK_GRAY)
        title = font.render("Theme & Background", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))
        theme_title = font.render("Pieces", True, LIGHT_GRAY)
        screen.blit(theme_title, (50, 100))
        for i, theme in enumerate(themes):
            color = WHITE if theme == selected_theme else LIGHT_GRAY
            label = font.render(theme.capitalize(), True, color)
            rect = label.get_rect(topleft=(100 + i * 150, 150))
            screen.blit(label, rect)
            if theme == selected_theme:
                pygame.draw.rect(screen, WHITE, rect.inflate(20, 10), 2)
        bg_title = font.render("Background", True, LIGHT_GRAY)
        screen.blit(bg_title, (50, 220))
        for i, bg in enumerate(bg_names):
            color = WHITE if bg == selected_bg else LIGHT_GRAY
            label = font.render(bg.capitalize(), True, color)
            rect = label.get_rect(topleft=(100 + i * 150, 270))
            screen.blit(label, rect)
            if bg == selected_bg:
                pygame.draw.rect(screen, WHITE, rect.inflate(20, 10), 2)
        ok_label = font.render("Ok", True, WHITE)
        ok_rect = ok_label.get_rect(center=(WIDTH // 2, HEIGHT - 100))
        screen.blit(ok_label, ok_rect)
        pygame.draw.rect(screen, LIGHT_GRAY, ok_rect.inflate(20, 10), 2)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                for i, theme in enumerate(themes):
                    if pygame.Rect(100 + i * 150, 150, 120, 40).collidepoint(x, y):
                        selected_theme = theme
                for i, bg in enumerate(bg_names):
                    if pygame.Rect(100 + i * 150, 270, 120, 40).collidepoint(x, y):
                        selected_bg = bg
                if ok_rect.inflate(20, 10).collidepoint(x, y):
                    return selected_theme, selected_bg if selected_bg != "None" else None

def show_play_menu(screen):
    font = pygame.font.SysFont(None, 48)
    options = ["Easy", "Medium", "Hard", "Pro"]
    while True:
        screen.fill((20, 20, 20))
        title = font.render("Choose level", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
        for i, text in enumerate(options):
            label = font.render(text, True, WHITE)
            rect = label.get_rect(center=(WIDTH // 2, 150 + i * 80))
            screen.blit(label, rect)
            pygame.draw.rect(screen, LIGHT_GRAY, rect, 2)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                _, y = event.pos
                for i in range(len(options)):
                    if 120 + i * 80 < y < 180 + i * 80:
                        return options[i].lower()

def show_color_selection(screen):
    font = pygame.font.SysFont(None, 48)
    options = ["White", "Black"]
    while True:
        screen.fill((25, 25, 25))
        title = font.render("Select side", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
        for i, text in enumerate(options):
            label = font.render(text, True, WHITE)
            rect = label.get_rect(center=(WIDTH // 2, 150 + i * 100))
            screen.blit(label, rect)
            pygame.draw.rect(screen, LIGHT_GRAY, rect, 2)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                _, y = event.pos
                for i in range(len(options)):
                    if 120 + i * 100 < y < 180 + i * 100:
                        return "white" if i == 0 else "black"

def show_main_menu(screen, backgrounds):
    font = pygame.font.SysFont(None, 48)
    options = ["Play", "Settings"]
    selected_theme = "classic"
    selected_background = None
    while True:
        screen.fill((30, 30, 30))
        for i, text in enumerate(options):
            label = font.render(text, True, WHITE)
            rect = label.get_rect(center=(WIDTH // 2, 200 + i * 100))
            screen.blit(label, rect)
            pygame.draw.rect(screen, LIGHT_GRAY, rect, 2)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                _, y = event.pos
                for i in range(len(options)):
                    if 180 + i * 100 < y < 240 + i * 100:
                        if options[i] == "Settings":
                            selected_theme, bg_name = show_settings(screen, backgrounds)
                            selected_background = backgrounds.get(bg_name) if bg_name in backgrounds else None
                        else:
                            difficulty = show_play_menu(screen)
                            return difficulty, selected_theme, selected_background

def move_to_san(board, move):
    return board.san(move)

def show_game_result(screen, result):
    font_large = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 36)
    overlay = pygame.Surface((WIDTH + PANEL_WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))
    if result == "1-0":
        message = "White is winner!"
    elif result == "0-1":
        message = "Black is winner!"
    else:
        message = "Draw"
    result_text = font_large.render(message, True, WHITE)
    screen.blit(result_text, (WIDTH // 2 - result_text.get_width() // 2, HEIGHT // 2 - 50))
    continue_text = font_small.render("Continue", True, WHITE)
    continue_rect = continue_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    pygame.draw.rect(screen, LIGHT_GRAY, continue_rect.inflate(20, 10))
    screen.blit(continue_text, continue_rect)
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if continue_rect.inflate(20, 10).collidepoint(event.pos):
                    return



def run_game(difficulty, theme, player_color, background=None):
    screen = pygame.display.set_mode((WIDTH + PANEL_WIDTH, HEIGHT))
    pygame.display.set_caption("Chess")
    clock = pygame.time.Clock()
    images = load_piece_images(theme)
    sounds = load_sounds()
    board = chess.Board()

    env = DummyVecEnv([lambda: ChessEnv()])

    model_path = {
        "easy": "chess_agent/ppo_easy",
        "medium": "chess_agent/ppo_medium",
        "hard": "chess_agent/ppo_hard",
        "pro": "chess_agent/ppo_pro"
    }[difficulty]

    if difficulty == "pro":
        try:
            model = PPO.load(model_path, env=env)
        except:
            model = PPO("MlpPolicy", env, verbose=0)
    else:
        try:
            model = PPO.load(model_path)
        except:
            print(f"Could not load model from {model_path}")
            return

    agent_color = chess.WHITE if player_color == "black" else chess.BLACK
    selected_square = None
    legal_squares = []
    last_agent_move = None
    move_history = []
    undone_moves = []
    turn_start_time = pygame.time.get_ticks()
    turn_time_limit = 30000

    font = pygame.font.SysFont(None, 32)
    small_font = pygame.font.SysFont(None, 24)

    back_label = font.render("Menu", True, (255, 255, 255))
    back_rect = back_label.get_rect(center=(WIDTH + PANEL_WIDTH // 2, 50))
    restart_label = font.render("Restart", True, (255, 255, 255))
    restart_rect = restart_label.get_rect(center=(WIDTH + PANEL_WIDTH // 2, 100))
    undo_label = font.render("Undo", True, (255, 255, 255))
    undo_rect = undo_label.get_rect(center=(WIDTH + PANEL_WIDTH // 2, 150))
    redo_label = font.render("Redo", True, (255, 255, 255))
    redo_rect = redo_label.get_rect(center=(WIDTH + PANEL_WIDTH // 2, 200))

    running = True
    game_over = False

    while running:
        frame_str = pygame.image.tostring(screen, 'RGB')
        frame_surf = pygame.image.fromstring(frame_str, (WIDTH + PANEL_WIDTH, HEIGHT), 'RGB')
        frame_array = pygame.surfarray.array3d(frame_surf).swapaxes(0, 1)
        frames.append(frame_array)
        current_time = pygame.time.get_ticks()
        time_left = max(0, turn_time_limit - (current_time - turn_start_time))

        screen.fill((30, 30, 30))
        board_surface = screen.subsurface((0, 0, WIDTH, HEIGHT))
        draw_board(board_surface, board, images, selected_square, legal_squares, last_agent_move, background)

        pygame.draw.rect(screen, (60, 60, 60), (WIDTH, 0, PANEL_WIDTH, HEIGHT))
        for rect, label in [(back_rect, back_label), (restart_rect, restart_label),
                            (undo_rect, undo_label), (redo_rect, redo_label)]:
            pygame.draw.rect(screen, (100, 100, 100), rect.inflate(20, 10))
            screen.blit(label, rect)

        turn_label = font.render(
            "Your turn" if board.turn == (chess.WHITE if player_color == "white" else chess.BLACK) else "Bot Turn",
            True, (255, 255, 255)
        )
        screen.blit(turn_label, (WIDTH + PANEL_WIDTH // 2 - turn_label.get_width() // 2, 250))
        timer_label = font.render(f"Time: {time_left // 1000}s", True, (255, 255, 255))
        screen.blit(timer_label, (WIDTH + PANEL_WIDTH // 2 - timer_label.get_width() // 2, 300))

        screen.blit(small_font.render("Move History:", True, (255, 255, 255)), (WIDTH + 10, 350))
        for i, move_text in enumerate(move_history[-10:]):
            screen.blit(small_font.render(move_text, True, (255, 255, 255)), (WIDTH + 10, 380 + i * 20))

        pygame.display.flip()
        clock.tick(FPS)

        if board.is_game_over() and not game_over:
            game_over = True
            result = board.result()
            if (result == "1-0" and player_color == "white") or (result == "0-1" and player_color == "black"):
                sounds['win'].play()
            elif result == "1/2-1/2":
                sounds['draw'].play()
            else:
                sounds['loss'].play()
            show_game_result(screen, result)
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                sounds['click'].play()

                if back_rect.collidepoint(event.pos):
                    return
                elif restart_rect.collidepoint(event.pos):
                    board.reset()
                    selected_square = None
                    legal_squares = []
                    last_agent_move = None
                    move_history.clear()
                    undone_moves.clear()
                    turn_start_time = pygame.time.get_ticks()
                    game_over = False
                    continue
                elif undo_rect.collidepoint(event.pos):
                    if len(board.move_stack) >= 1:
                        undone_moves.append(board.pop())
                        if len(board.move_stack) >= 1:
                            undone_moves.append(board.pop())
                        move_history = move_history[:-2] if len(move_history) >= 2 else []
                        turn_start_time = pygame.time.get_ticks()
                elif redo_rect.collidepoint(event.pos):
                    if undone_moves:
                        board.push(undone_moves.pop())
                        if undone_moves:
                            board.push(undone_moves.pop())
                        turn_start_time = pygame.time.get_ticks()
                elif event.pos[0] < WIDTH:
                    square = get_square_from_mouse(event.pos)
                    if square is not None:
                        piece = board.piece_at(square)
                        if selected_square is None:
                            if piece and piece.color == (chess.WHITE if player_color == "white" else chess.BLACK):
                                selected_square = square
                                legal_squares = [m.to_square for m in board.legal_moves if m.from_square == square]
                        else:
                            from_piece = board.piece_at(selected_square)
                            move = chess.Move(selected_square, square)
                            if from_piece and from_piece.piece_type == chess.PAWN and (chess.square_rank(square) in [0, 7]):
                                promo_piece = choose_promotion(screen, images, board.turn)
                                move = chess.Move(selected_square, square, promotion=promo_piece)
                            if move in board.legal_moves:
                                sounds['capture' if board.is_capture(move) else 'move'].play()
                                move_history.append(move_to_san(board, move))
                                animate_piece_move(screen, board, images, move, background)
                                board.push(move)
                                undone_moves.clear()
                            selected_square = None
                            legal_squares = []

        if not game_over and board.turn == agent_color and not board.is_game_over():
            if time_left <= 0:
                result = "0-1" if player_color == "white" else "1-0"
                show_game_result(screen, result)
                return
            try:
                obs = env.envs[0]._get_obs()
                action, _ = model.predict(obs, deterministic=True)

                move = env.envs[0].all_moves[action]
                if move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))

                sounds['capture' if board.is_capture(move) else 'move'].play()
                move_history.append(move_to_san(board, move))
                animate_piece_move(screen, board, images, move, background)
                board.push(move)
                last_agent_move = move
                undone_moves.clear()
                turn_start_time = pygame.time.get_ticks()

                if difficulty == "pro":
                    model.learn(total_timesteps=1, reset_num_timesteps=False)
                    model.save(model_path)

            except Exception as e:
                print(f"Error during agent move: {e}")




if __name__ == "__main__":
    backgrounds = load_backgrounds()
    screen = pygame.display.set_mode((WIDTH + PANEL_WIDTH, HEIGHT))
    while True:
        difficulty, theme, background = show_main_menu(screen, backgrounds)
        player_color = show_color_selection(screen)
        run_game(difficulty, theme, player_color, background)
        imageio.mimsave("gameplay.mp4", frames, fps=30)

