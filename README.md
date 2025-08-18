# ♟️ Chess RL GUI

An interactive and user-friendly chess game powered by reinforcement learning agents using Stable-Baselines3 and Gymnasium. Designed with rich visual feedback, dynamic difficulty selection and customizable themes to make AI chess both accessible and fun.

![gameplay-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/9eac1f9e-22eb-4258-b066-495069b69259)

## 🚀 Features

- **Play vs AI**: Choose from four difficulty levels - Easy, Medium, Hard and Pro (online learning feature - trial).

- **Customizable experience**:

- Choose your side: White or Black.

- Choose piece theme: Classic, Wooden, Glass.

- Selection screen backgrounds: Wooden, Green, Glass, Normal.

**Smooth gameplay**:

- Highlight allowed moves, last piece move and selected piece.

- Animated piece movement for better visual clarity.

- Sound effects for moves, captures, wins, losses and clicks.

- **Game Controls**:

- Undo/Redo moves.

- Restart the game.

- Return to the main menu without exiting. - **Move History Panel**: Show the last 10 moves.

**Upgrade Menu**: Intuitive piece selection when pawns reach the final rank.

**Game Results Page**: Show the result with overlay and sound.

## 🧠 AI Agent

- Built using **Stable-Baselines3 PPO**.

- Trained in a custom environment `ChessEnv`.

- "Professional" mode supports online learning and saving the model during gameplay.

## 📦 Requirements

Make sure you have the following dependencies installed:

```bash
pip install pygame chess numpy stable-baselines3 imageio
```

You will also need the following:
- `chess_env.py`: Gym-compatible custom environment. - Pre-trained models in `chess_agent/` directory:
- `ppo_easy`
- `ppo_medium`
- `ppo_hard`
- `ppo_pro`
- Piece images in `assets/<theme>/`
- Background images in `backgrounds/`
- Sound files in `sounds/`

## 🖼️ Folder structure

```
├── assets/
│ └── classic/, wood/, glass/ # Piece images
├── backgrounds/ # Board backgrounds
├── sounds/ # Sound effects
├── chess_agent/ # Pre-trained PPO models
├── chess_env.py # Custom machine learning environment
├── mainGui.py # Main GUI of the script
```

## ▶️ How to run

```bash
python mainGui.py
```

## 🙌 Credits

- Made with ❤️ by Using Pygame, Python-Chess and Stable-Baselines3.

- Designed for intuitive interaction and playful learning

Looking forward to your collaboration and issues

> Github.com/RezaGooner
