"""
games/bci_pong.py
-----------------
Brain-wave controlled Pong game.

Left  paddle → LEFT HAND motor imagery
Right paddle → RIGHT HAND motor imagery

When use_real_hardware=False (default), the paddles are driven by simulated
motor imagery predictions so the game works without a real EEG device.

Dependencies: pygame

Usage
-----
    python main.py --mode bci-pong
    # or:
    from games.bci_pong import start_bci_pong
    start_bci_pong()
"""

import sys
import random
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Project-root import guard
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 800, 500
FPS = 60

PADDLE_W, PADDLE_H = 12, 90
BALL_SIZE = 14
PADDLE_SPEED = 6
BALL_SPEED_INIT = 5

COLORS = {
    "bg":         (10,  10,  20),
    "ball":       (255, 255, 255),
    "left_pad":   (33,  150, 243),   # blue  — LEFT HAND
    "right_pad":  (76,  175,  80),   # green — RIGHT HAND
    "text":       (220, 220, 220),
    "overlay_bg": (0,   0,   0),
    "focus":      (33,  150, 243),
    "relax":      (76,  175,  80),
    "stress":     (244,  67,  54),
    "sleep":      (156,  39, 176),
    "meditation": (255, 235,  59),
}


# ---------------------------------------------------------------------------
# Simulated motor imagery source (used when use_real_hardware=False)
# ---------------------------------------------------------------------------
def _simulated_intent() -> Tuple[str, float]:
    """Return a random motor imagery intent for demo purposes."""
    import random
    import numpy as np
    options = ["LEFT HAND", "RIGHT HAND", "REST"]
    intent = random.choices(options, weights=[0.35, 0.35, 0.30])[0]
    conf   = random.uniform(0.65, 0.97)
    return intent, conf


# ---------------------------------------------------------------------------
# Main game function
# ---------------------------------------------------------------------------
def start_bci_pong(use_real_hardware: bool = False) -> None:
    """Launch the BCI Pong game.

    Args:
        use_real_hardware: If True, attempt to load the motor model for real
                           predictions.  If False (default), use simulated
                           motor imagery intents.
    """
    try:
        import pygame  # type: ignore
    except ImportError:
        print("pygame is required. Install with: pip install pygame")
        return

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("🧠 BCI Pong — Brain-Controlled!")
    clock = pygame.time.Clock()

    font_large  = pygame.font.SysFont("monospace", 32, bold=True)
    font_medium = pygame.font.SysFont("monospace", 20)
    font_small  = pygame.font.SysFont("monospace", 14)

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------
    left_y  = SCREEN_H // 2 - PADDLE_H // 2
    right_y = SCREEN_H // 2 - PADDLE_H // 2

    ball_x  = float(SCREEN_W // 2)
    ball_y  = float(SCREEN_H // 2)
    ball_vx = BALL_SPEED_INIT * random.choice([-1, 1])
    ball_vy = BALL_SPEED_INIT * random.choice([-1, 1])

    score_left  = 0
    score_right = 0

    current_state = "FOCUS"
    current_conf  = 0.0

    # Intent prediction tick (updates every ~200 ms)
    last_predict_ms = 0
    PREDICT_INTERVAL_MS = 200

    running = True
    while running:
        dt_ms = clock.tick(FPS)
        now_ms = pygame.time.get_ticks()

        # ------------------------------------------------------------------
        # Events
        # ------------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # ------------------------------------------------------------------
        # Get motor imagery intent
        # ------------------------------------------------------------------
        if now_ms - last_predict_ms >= PREDICT_INTERVAL_MS:
            last_predict_ms = now_ms
            if use_real_hardware:
                try:
                    from data.simulate_motor_eeg import generate_motor_eeg
                    from inference.predict_motor import predict_motor_intent
                    sig, _ = generate_motor_eeg(n_samples=1, n_channels=64)
                    intent, conf = predict_motor_intent(sig[0], fs=250)
                except Exception:
                    intent, conf = _simulated_intent()
            else:
                intent, conf = _simulated_intent()

            current_state = intent.upper() if intent != "REST" else "REST"
            current_conf  = conf * 100

            # Move paddles based on intent
            if intent == "LEFT HAND":
                left_y -= PADDLE_SPEED * 3
            elif intent == "RIGHT HAND":
                right_y -= PADDLE_SPEED * 3

        # Clamp paddles
        left_y  = max(0, min(SCREEN_H - PADDLE_H, left_y))
        right_y = max(0, min(SCREEN_H - PADDLE_H, right_y))

        # AI-assist for opposite paddle (keeps game playable)
        if ball_vx > 0:
            if ball_y < right_y + PADDLE_H // 2:
                right_y -= PADDLE_SPEED
            elif ball_y > right_y + PADDLE_H // 2:
                right_y += PADDLE_SPEED
        else:
            if ball_y < left_y + PADDLE_H // 2:
                left_y -= PADDLE_SPEED
            elif ball_y > left_y + PADDLE_H // 2:
                left_y += PADDLE_SPEED

        # ------------------------------------------------------------------
        # Ball physics
        # ------------------------------------------------------------------
        ball_x += ball_vx
        ball_y += ball_vy

        # Top / bottom bounce
        if ball_y <= 0 or ball_y >= SCREEN_H - BALL_SIZE:
            ball_vy = -ball_vy

        # Left paddle collision
        if (ball_x <= PADDLE_W + 10 and
                left_y <= ball_y <= left_y + PADDLE_H):
            ball_vx = abs(ball_vx) * 1.05
            ball_vy += random.uniform(-1, 1)

        # Right paddle collision
        if (ball_x >= SCREEN_W - PADDLE_W - BALL_SIZE - 10 and
                right_y <= ball_y <= right_y + PADDLE_H):
            ball_vx = -abs(ball_vx) * 1.05
            ball_vy += random.uniform(-1, 1)

        # Limit max speed
        ball_vx = max(-15.0, min(15.0, ball_vx))
        ball_vy = max(-15.0, min(15.0, ball_vy))

        # Score
        if ball_x < 0:
            score_right += 1
            ball_x, ball_y = SCREEN_W // 2, SCREEN_H // 2
            ball_vx = BALL_SPEED_INIT * random.choice([-1, 1])
            ball_vy = BALL_SPEED_INIT * random.choice([-1, 1])

        if ball_x > SCREEN_W:
            score_left += 1
            ball_x, ball_y = SCREEN_W // 2, SCREEN_H // 2
            ball_vx = BALL_SPEED_INIT * random.choice([-1, 1])
            ball_vy = BALL_SPEED_INIT * random.choice([-1, 1])

        # ------------------------------------------------------------------
        # Draw
        # ------------------------------------------------------------------
        screen.fill(COLORS["bg"])

        # Centre line
        for y_dashed in range(0, SCREEN_H, 20):
            pygame.draw.rect(screen, (40, 40, 60), (SCREEN_W // 2 - 2, y_dashed, 4, 10))

        # Paddles
        pygame.draw.rect(
            screen, COLORS["left_pad"],
            (5, left_y, PADDLE_W, PADDLE_H), border_radius=4
        )
        pygame.draw.rect(
            screen, COLORS["right_pad"],
            (SCREEN_W - PADDLE_W - 5, right_y, PADDLE_W, PADDLE_H), border_radius=4
        )

        # Ball
        pygame.draw.ellipse(
            screen, COLORS["ball"],
            (int(ball_x), int(ball_y), BALL_SIZE, BALL_SIZE)
        )

        # Score
        score_surf = font_large.render(f"{score_left}   {score_right}", True, COLORS["text"])
        screen.blit(score_surf, (SCREEN_W // 2 - score_surf.get_width() // 2, 12))

        # Brain-state overlay (bottom)
        overlay_h = 54
        overlay_surf = pygame.Surface((SCREEN_W, overlay_h), pygame.SRCALPHA)
        overlay_surf.fill((0, 0, 0, 160))
        screen.blit(overlay_surf, (0, SCREEN_H - overlay_h))

        state_key = current_state.lower().replace("_", "")
        state_color = COLORS.get(state_key, COLORS["text"])

        brain_text = font_medium.render(
            f"🧠 Brain Intent: {current_state}  ({current_conf:.0f}%)",
            True, state_color
        )
        screen.blit(brain_text, (10, SCREEN_H - overlay_h + 6))

        hint_text = font_small.render(
            "LEFT HAND = left paddle up  |  RIGHT HAND = right paddle up  |  ESC = quit",
            True, (120, 120, 140)
        )
        screen.blit(hint_text, (10, SCREEN_H - overlay_h + 32))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    start_bci_pong()
