import os
import sys
import warnings
import contextlib
import io

warnings.filterwarnings("ignore", category=DeprecationWarning)
with contextlib.redirect_stderr(io.StringIO()):
    import gym
import gymnasium
import slimevolleygym
from slimevolleygym.slimevolley import setPixelObsMode
import neat
import numpy as np
import cv2
import pygame

gym.logger.min_level = gym.logger.ERROR
gymnasium.logger.min_level = gymnasium.logger.ERROR
setPixelObsMode()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO = "--auto" in sys.argv
cp_args = [a for a in sys.argv[1:] if a != "--auto"]
CHECKPOINT = cp_args[0] if cp_args else os.path.join(SCRIPT_DIR, "ckpts_parallel/ckpt-100")

pop = neat.Checkpointer.restore_checkpoint(CHECKPOINT)
valid = [(k, g) for k, g in pop.population.items() if g.fitness is not None]
_, winner = max(valid, key=lambda x: x[1].fitness)

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    os.path.join(SCRIPT_DIR, "config-feedforward"),
)
ai_net = neat.nn.FeedForwardNetwork.create(winner, config)
print(f"Loaded: fitness={winner.fitness:.1f}  nodes={len(winner.nodes)}  conns={len(winner.connections)}")

INPUT_NAMES = {
    -1: "my_x", -2: "my_y", -3: "my_vx", -4: "my_vy",
    -5: "b_x", -6: "b_y", -7: "b_vx", -8: "b_vy",
    -9: "op_x", -10: "op_y", -11: "op_vx", -12: "op_vy",
}
OUTPUT_NAMES = ["FWD", "BWD", "JMP"]

SCALE = 1.5
GAME_W, GAME_H = int(336 * SCALE), int(336 * SCALE)
NET_W, NET_H = int(420 * SCALE), int(560 * SCALE)
NET_X = GAME_W + int(20 * SCALE)
TOTAL_W = GAME_W + int(20 * SCALE) + NET_W + int(20 * SCALE)
TOTAL_H = max(GAME_H, NET_H) + int(10 * SCALE)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    env = gym.make("SlimeVolley-v0")
obs = env.reset()
game = env.game

ai_net.activate(obs)
inp_nodes = sorted(ai_net.input_nodes)
out_nodes = sorted(ai_net.output_nodes)
hid_nodes = sorted(set(ai_net.values.keys()) - set(inp_nodes) - set(out_nodes))

def make_layout(hidden):
    pos = {}
    ih = NET_H / (len(inp_nodes) + 1)
    for i, n in enumerate(inp_nodes):
        pos[n] = (0.08, (i + 1) * ih / NET_H)
    oh = NET_H / (len(out_nodes) + 1)
    for i, n in enumerate(out_nodes):
        pos[n] = (0.92, (i + 1) * oh / NET_H)
    hh = NET_H / (len(hidden) + 1)
    for i, n in enumerate(hidden):
        pos[n] = (0.50, (i + 1) * hh / NET_H)
    return pos

node_pos = make_layout(hid_nodes)

conns = []
for nid, _, _, _, _, links in ai_net.node_evals:
    for s, w in links:
        conns.append((s, nid, w))
max_w = max((abs(w) for _, _, w in conns), default=1.0) or 1.0
connected_inputs = set(s for s, _, _ in conns if s in inp_nodes)

pygame.init()
pygame.key.stop_text_input()
screen = pygame.display.set_mode((TOTAL_W, TOTAL_H))
pygame.display.set_caption(f"{'Base' if AUTO else 'You'}(Yellow) vs AI(Blue) \u2014 {os.path.basename(CHECKPOINT)}")
clock = pygame.time.Clock()
font = pygame.font.Font(None, int(16 * SCALE))
small_font = pygame.font.Font(None, int(13 * SCALE))

net_bg = pygame.Surface((NET_W, NET_H))
net_bg.fill((25, 25, 35))
for src, dst, w in conns:
    if src not in node_pos or dst not in node_pos:
        continue
    x1 = node_pos[src][0] * NET_W
    y1 = node_pos[src][1] * NET_H
    x2 = node_pos[dst][0] * NET_W
    y2 = node_pos[dst][1] * NET_H
    t = max(1, int(abs(w) / max_w * 5 * SCALE))
    r = int(max(0, min(255, 180 * abs(w) / max_w + 75)))
    c = (r, 80, 80) if w > 0 else (80, 80, r)
    pygame.draw.line(net_bg, c, (x1, y1), (x2, y2), t)
for nid, (rx, ry) in node_pos.items():
    cx, cy = rx * NET_W, ry * NET_H
    is_inp = nid in inp_nodes
    is_out = nid in out_nodes
    rad = int((10 if is_inp or is_out else 7) * SCALE)
    pygame.draw.circle(net_bg, (50, 50, 60), (int(cx), int(cy)), rad)
    pygame.draw.circle(net_bg, (200, 200, 200), (int(cx), int(cy)), rad, 1)
    if is_inp:
        lbl = INPUT_NAMES.get(nid, str(nid))
    elif is_out:
        idx = out_nodes.index(nid)
        lbl = OUTPUT_NAMES[idx] if idx < len(OUTPUT_NAMES) else str(nid)
    else:
        lbl = str(nid)
    t = small_font.render(lbl, True, (180, 180, 180))
    net_bg.blit(t, (cx - t.get_width() // 2, cy + rad + int(2 * SCALE)))

def get_human_action():
    keys = pygame.key.get_pressed()
    return [1 if keys[pygame.K_d] else 0, 1 if keys[pygame.K_a] else 0, 1 if keys[pygame.K_w] else 0]

running = True
score = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    ai_raw = ai_net.activate(obs)
    ai_action = [1 if a > 0.5 else 0 for a in ai_raw]
    if AUTO:
        obs, reward, done, info = env.step(ai_action)
    else:
        obs, reward, done, info = env.unwrapped.step(ai_action, get_human_action())
    score -= reward

    if done:
        running = False

    screen.fill((0, 0, 0))
    canvas = game.display(None)
    canvas = cv2.resize(canvas, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
    surf = pygame.surfarray.make_surface(canvas.swapaxes(0, 1))
    screen.blit(surf, (0, 0))

    screen.blit(net_bg, (NET_X, 0))
    for nid, (rx, ry) in node_pos.items():
        cx, cy = NET_X + rx * NET_W, ry * NET_H
        is_inp = nid in inp_nodes
        is_out = nid in out_nodes
        rad = int((10 if is_inp or is_out else 7) * SCALE)
        if is_inp and nid not in connected_inputs:
            continue
        val = max(-1.0, min(1.0, ai_net.values.get(nid, 0.0)))
        norm = (val + 1) / 2
        r = int(max(0, min(255, norm * 255)))
        b = int(max(0, min(255, (1 - norm) * 255)))
        g = int(max(0, min(255, 255 - abs(val) * 200)))
        pygame.draw.circle(screen, (r, g, b), (int(cx), int(cy)), rad)
        pygame.draw.circle(screen, (200, 200, 200), (int(cx), int(cy)), rad, 1)

    you_label = "Base" if AUTO else "You"
    you_lives = info['ale.otherLives']
    ai_lives = info['ale.lives']
    screen.blit(font.render(f"Score: {score:+d}  {you_label}/{you_lives}  AI/{ai_lives}", True, (200, 200, 200)), (int(8 * SCALE), GAME_H + int(4 * SCALE)))
    if not AUTO:
        screen.blit(font.render("W:Jump  A:Left  D:Right   ESC:Exit", True, (180, 180, 180)), (int(8 * SCALE), GAME_H + int(24 * SCALE)))

    pygame.display.flip()
    clock.tick(50)

pygame.quit()
env.close()
print(f"Final score: {score}")
