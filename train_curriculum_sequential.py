import os
import sys
import warnings
import contextlib
import io

import neat
with contextlib.redirect_stderr(io.StringIO()):
    import gym
import gymnasium
import slimevolleygym
import numpy as np
import pickle


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTAL_GENS = 200
EVAL_EPISODES = 3
MAX_STEP = 3000
CKPT_DIR = os.path.join(SCRIPT_DIR, "ckpts_sequential")
CKPT_EVERY = 10
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config-feedforward")
WINNER_PATH = os.path.join(SCRIPT_DIR, "winner_sequential.pkl")


def _rally_reward(obs, prev_lives, cur_lives, step, streak, was_hit, env_score):
    bx, by, bvx = obs[4], obs[5], obs[6]
    mx, my = obs[0], obs[1]
    r = 0.0

    if env_score != 0:
        return r, 0, False

    if cur_lives < prev_lives:
        r -= 1.0

    r += 0.003

    if bx > 0:
        dist = np.hypot(mx - bx, my - by)
        r += max(0.0, 0.1 - dist * 0.05)
        was_hit = False
    elif bx < 0 and bvx < 0 and not was_hit:
        streak += 1
        r += 0.5 * streak
        was_hit = True

    return r, streak, was_hit


def _eval_one(genome, config):
    gym.logger.min_level = gym.logger.ERROR
    gymnasium.logger.min_level = gymnasium.logger.ERROR
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_fit = 0.0
    best_streak = 0

    for _ in range(EVAL_EPISODES):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = gym.make("SlimeVolley-v0")

        obs = env.reset()
        done = False
        lives = 5
        streak = 0
        was_hit = False
        ep_fit = 0.0
        ep_streak = 0
        step = 0

        while not done and step < MAX_STEP:
            raw = net.activate(obs)
            act = [1 if a > 0.5 else 0 for a in raw]
            obs, score, done, info = env.step(act)
            cur = info["ale.lives"]

            rw, streak, was_hit = _rally_reward(
                obs, lives, cur, step, streak, was_hit, score
            )

            ep_fit += rw
            ep_streak = max(ep_streak, streak)
            lives = cur
            step += 1

        env.close()
        total_fit += ep_fit
        best_streak = max(best_streak, ep_streak)

    return total_fit / EVAL_EPISODES, best_streak


class RallyEvaluator:
    def __init__(self):
        self.gen = 0

    def __call__(self, genomes, config):
        self.gen += 1
        print(f"\n  === Gen {self.gen} ===")

        all_streaks = []

        for _, genome in genomes:
            fit, streak = _eval_one(genome, config)
            genome.fitness = fit
            all_streaks.append(streak)

        avg_streak = np.mean(all_streaks)
        print(f"  \u2192 avg rally: {avg_streak:.1f}")


def main(checkpoint=None):
    gym.logger.min_level = gym.logger.ERROR
    gymnasium.logger.min_level = gymnasium.logger.ERROR
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        CONFIG_PATH,
    )

    if checkpoint:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
        print(f"Restored from {checkpoint}")
    else:
        pop = neat.Population(config)
        print("Starting sequential training (rally reward only)")

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    os.makedirs(CKPT_DIR, exist_ok=True)
    pop.add_reporter(neat.Checkpointer(CKPT_EVERY, filename_prefix=f"{CKPT_DIR}/ckpt-"))

    evaluator = RallyEvaluator()
    remaining = max(0, TOTAL_GENS - (pop.generation if checkpoint else 0))

    if remaining == 0:
        print("Already reached target generations.")
        sys.exit(0)

    winner = pop.run(evaluator, remaining)

    with open(WINNER_PATH, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nDone. Saved {WINNER_PATH}\n{winner}")


if __name__ == "__main__":
    cp = sys.argv[1] if len(sys.argv) > 1 else None
    main(cp)
