import os, sys, math, random, time, json, argparse
from dataclasses import dataclass
import numpy as np

# ======= Config chung =======
SCREEN_W, SCREEN_H = 400, 600
GRAVITY = 0.35
FLAP_VELOCITY = -6.0
PIPE_GAP = 160
PIPE_W = 60
PIPE_SPEED = 2.5
PIPE_INTERVAL = 150  # px between pipes
GROUND_Y = SCREEN_H - 80

def lazy_import_pygame():
    global pygame
    import pygame
    return pygame

@dataclass
class Bird:
    x: float
    y: float
    vel: float

@dataclass
class Pipe:
    x: float
    gap_y: float 

class FlappyEnv:
    """
    Quan sát (5 features, đã chuẩn hoá):
      [bird_y_norm, bird_vel_norm, next_pipe_x_norm, top_gap_norm, bot_gap_norm]
    Hành động: 0 = no flap, 1 = flap
    Fitness: số ống vượt qua * 100 + frames sống (để phân biệt khi cùng số ống)
    """
    def __init__(self, seed=None, render=False):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.render_enabled = render
        self._pygame = None
        self.reset()

    def reset(self):
        self.bird = Bird(x=80, y=SCREEN_H//2, vel=0.0)
        self.pipes = []
        self.frame = 0
        self.score = 0
        self.alive = True
        self.distance_since_last_pipe = 0.0
        # tạo 4 pipes để có sẵn “đường chạy”
        x = SCREEN_W + 50
        for _ in range(4):
            self._spawn_pipe(x)
            x += PIPE_INTERVAL
        return self._get_obs()

    def _spawn_pipe(self, x):
        # tránh đặt gap quá sát viền trên/dưới
        margin = 60
        gap_y = self.rng.randint(margin+PIPE_GAP//2, GROUND_Y - margin - PIPE_GAP//2)
        self.pipes.append(Pipe(x=float(x), gap_y=float(gap_y)))

    def step(self, action):
        if not self.alive:
            return self._get_obs(), 0.0, True, {}

        # bird physics
        if action == 1:
            self.bird.vel = FLAP_VELOCITY
        self.bird.vel += GRAVITY
        self.bird.y += self.bird.vel

        # move pipes
        for p in self.pipes:
            p.x -= PIPE_SPEED
        # remove off-screen + spawn new
        if self.pipes and self.pipes[0].x + PIPE_W < 0:
            self.pipes.pop(0)
        # ensure spacing
        if self.pipes:
            rightmost = max(self.pipes, key=lambda p: p.x)
            if rightmost.x < SCREEN_W - PIPE_INTERVAL:
                self._spawn_pipe(SCREEN_W + 10)

        # scoring: khi bird vượt qua center của 1 pipe
        reward = 0.0
        for p in self.pipes:
            if (p.x + PIPE_W/2) < self.bird.x <= (p.x + PIPE_W/2 + PIPE_SPEED):
                self.score += 1
                reward += 100.0  # dồn vào fitness cuối, ở đây cũng cho tí “vui”

        self.frame += 1
        reward += 1.0  # sống thêm 1 frame

        # collision
        if self._collide():
            self.alive = False
            done = True
        else:
            done = False

        return self._get_obs(), reward, done, {}

    def _collide(self):
        # chạm đất/mái
        if self.bird.y < 0 or self.bird.y > GROUND_Y:
            return True
        # chạm pipe
        bx, by = self.bird.x, self.bird.y
        bird_r = 12  # bán kính hitbox đơn giản
        for p in self.pipes:
            # thân pipe rect
            top_rect = (p.x, 0, PIPE_W, p.gap_y - PIPE_GAP/2)
            bot_rect = (p.x, p.gap_y + PIPE_GAP/2, PIPE_W, SCREEN_H - (p.gap_y + PIPE_GAP/2))
            if circle_rect_collision(bx, by, bird_r, top_rect) or circle_rect_collision(bx, by, bird_r, bot_rect):
                return True
        return False

    def _get_obs(self):
        # tìm ống sắp tới (có x + PIPE_W/2 >= bird.x)
        next_pipes = sorted([p for p in self.pipes if p.x + PIPE_W/2 >= self.bird.x], key=lambda p: p.x)
        if not next_pipes:
            target = self.pipes[0]
        else:
            target = next_pipes[0]
        # chuẩn hoá về [0,1] tương đối
        bird_y_norm = np.clip(self.bird.y / GROUND_Y, 0, 1)
        bird_vel_norm = np.tanh(self.bird.vel / 8.0)  # -1..1
        next_pipe_x_norm = np.clip((target.x - self.bird.x) / SCREEN_W, 0, 1)
        top_gap_norm = np.clip((target.gap_y - PIPE_GAP/2) / GROUND_Y, 0, 1)
        bot_gap_norm = np.clip((target.gap_y + PIPE_GAP/2) / GROUND_Y, 0, 1)
        return np.array([bird_y_norm, bird_vel_norm, next_pipe_x_norm, top_gap_norm, bot_gap_norm], dtype=np.float32)

    # ------- Render -------
    def render(self):
        if not self.render_enabled:
            return
        if self._pygame is None:
            self._pygame = lazy_import_pygame()
            pygame = self._pygame
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Flappy GA")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)

        self.screen.fill((135, 206, 235))  # sky
        # ground
        pygame.draw.rect(self.screen, (222, 184, 135), (0, GROUND_Y, SCREEN_W, SCREEN_H-GROUND_Y))

        # pipes
        for p in self.pipes:
            top_h = int(p.gap_y - PIPE_GAP/2)
            bot_y = int(p.gap_y + PIPE_GAP/2)
            pygame.draw.rect(self.screen, (34,139,34), (int(p.x), 0, PIPE_W, top_h))
            pygame.draw.rect(self.screen, (34,139,34), (int(p.x), bot_y, PIPE_W, SCREEN_H-bot_y))

        # bird
        pygame.draw.circle(self.screen, (255,215,0), (int(self.bird.x), int(self.bird.y)), 12)

        # text
        txt = self.font.render(f"Score: {self.score}  Frame:{self.frame}", True, (0,0,0))
        self.screen.blit(txt, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

def circle_rect_collision(cx, cy, r, rect):
    rx, ry, rw, rh = rect
    # clamp point
    nx = max(rx, min(cx, rx + rw))
    ny = max(ry, min(cy, ry + rh))
    dx = cx - nx
    dy = cy - ny
    return (dx*dx + dy*dy) <= (r*r)

# ======= Agent NN =======
class TinyNN:
    """
    Kiến trúc: 5 -> 3 -> 1
    act: hidden = tanh, out = sigmoid (>=0.5 thì flap)
    Genome là vector (weights + biases) flatten.
    """
    def __init__(self, genome=None, rng=None):
        self.rng = rng or np.random.default_rng()
        self.shapes = [(5,3), (3,), (3,1), (1,)]
        self.size = sum(np.prod(s) for s in self.shapes)
        if genome is None:
            self.genome = self.rng.normal(0, 0.5, self.size).astype(np.float32)
        else:
            self.genome = genome.astype(np.float32)
        self._unpack()

    def _unpack(self):
        g = self.genome
        idx = 0
        self.W1 = g[idx: idx+15].reshape(5,3); idx += 15
        self.b1 = g[idx: idx+3]; idx += 3
        self.W2 = g[idx: idx+3].reshape(3,1); idx += 3
        self.b2 = g[idx: idx+1]; idx += 1

    def act(self, obs):
        # obs shape (5,)
        h = np.tanh(obs @ self.W1 + self.b1)  # (3,)
        o = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))  # (1,)
        return 1 if o[0] >= 0.5 else 0

# ======= GA =======
class GA:
    def __init__(self, pop_size=100, elitism=0.1, cx_rate=0.7, mut_rate=0.02, mut_sigma=0.2, seed=None):
        self.pop_size = pop_size
        self.elite_n = max(1, int(pop_size * elitism))
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.mut_sigma = mut_sigma
        self.rng = np.random.default_rng(seed)

    def evaluate(self, genome, n_episodes=1, max_steps=20000, render=False, seed=None):
        total = 0.0
        best = -1e9
        for ep in range(n_episodes):
            env = FlappyEnv(seed=(seed if seed is not None else self.rng.integers(1e9)),
                            render=render)
            obs = env.reset()
            agent = TinyNN(genome=genome, rng=self.rng)
            steps = 0
            done = False
            fit = 0.0
            while not done and steps < max_steps:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                fit += reward
                if render:
                    env.render()
                steps += 1
            total += fit
            best = max(best, fit)
        return total / n_episodes, best

    def tournament_select(self, pop, fitness, k=3):
        idxs = self.rng.integers(0, len(pop), size=k)
        best_idx = idxs[0]
        best_fit = fitness[best_idx]
        for i in idxs[1:]:
            if fitness[i] > best_fit:
                best_idx = i; best_fit = fitness[i]
        return pop[best_idx].copy()

    def crossover(self, g1, g2):
        if self.rng.random() > self.cx_rate:
            return g1.copy(), g2.copy()
        # 1-point crossover
        point = self.rng.integers(1, len(g1)-1)
        c1 = np.concatenate([g1[:point], g2[point:]])
        c2 = np.concatenate([g2[:point], g1[point:]])
        return c1, c2

    def mutate(self, g):
        mask = self.rng.random(g.shape) < self.mut_rate
        noise = self.rng.normal(0, self.mut_sigma, g.shape).astype(np.float32)
        g = g.copy()
        g[mask] += noise[mask]
        # clip nhẹ để tránh bùng nổ
        np.clip(g, -5, 5, out=g)
        return g
    
def run(self, n_gen=50, eval_episodes=1, seed=None, log_path="ga_log.json", model_path="best_genome.npz", resume=False):
    temp = TinyNN(rng=self.rng)

    if resume and os.path.exists(model_path):
        data = np.load(model_path)
        best = data["genome"]
        # tạo quần thể mới quanh best genome
        pop = [best.copy()]
        for _ in range(self.pop_size-1):
            g = self.mutate(best)   # clone + đột biến
            pop.append(g)
        print("Resumed training from saved best genome.")
    else:
        pop = [TinyNN(rng=self.rng).genome for _ in range(self.pop_size)]
        logs = {"gen": [], "fitness_mean": [], "fitness_best": [], "fitness_std": []}
        best_overall = None
        best_fit_overall = -1e9

        for gen in range(1, n_gen+1):
            fits = np.zeros(self.pop_size, dtype=np.float32)
            for i, g in enumerate(pop):
                mean_fit, _ = self.evaluate(g, n_episodes=eval_episodes, seed=(seed or gen*1337+i))
                fits[i] = mean_fit

            order = np.argsort(fits)[::-1]
            elites = [pop[i].copy() for i in order[:self.elite_n]]
            fit_best = float(fits[order[0]])
            fit_mean = float(fits.mean())
            fit_std = float(fits.std())

            # cập nhật best overall
            if fit_best > best_fit_overall:
                best_fit_overall = fit_best
                best_overall = pop[order[0]].copy()
                np.savez(model_path, genome=best_overall)

            # log
            logs["gen"].append(gen)
            logs["fitness_mean"].append(fit_mean)
            logs["fitness_best"].append(fit_best)
            logs["fitness_std"].append(fit_std)
            print(f"[Gen {gen:03d}] best={fit_best:.1f} mean={fit_mean:.1f} std={fit_std:.1f}")

            # tạo thế hệ mới
            new_pop = elites[:]  # giữ elite
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select(pop, fits)
                p2 = self.tournament_select(pop, fits)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_pop.extend([c1, c2])
            pop = new_pop[:self.pop_size]

        # lưu log
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved best genome to {model_path} and logs to {log_path}")
        return logs, best_overall

# ======= Vẽ biểu đồ =======
def plot_logs(log_path):
    import matplotlib.pyplot as plt
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)
    gens = logs["gen"]
    mean = logs["fitness_mean"]
    best = logs["fitness_best"]
    std = logs["fitness_std"]

    plt.figure()
    plt.plot(gens, mean, label="fitness_mean")
    plt.plot(gens, best, label="fitness_best")
    # dải std
    upper = np.array(mean) + np.array(std)
    lower = np.array(mean) - np.array(std)
    plt.fill_between(gens, lower, upper, alpha=0.2, label="±1 std")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Progress")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ======= Xem con giỏi nhất =======
def watch(model_path="best_genome.npz", seed=None):
    data = np.load(model_path)
    genome = data["genome"]
    ga = GA()
    mean_fit, best_fit = ga.evaluate(genome, n_episodes=1, render=True, seed=seed, max_steps=50000)
    print(f"Watch run — fitness {mean_fit:.1f}")

# ======= CLI =======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train GA")
    parser.add_argument("--watch", action="store_true", help="Watch best genome play")
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--gen", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log", type=str, default="ga_log.json")
    parser.add_argument("--model", type=str, default="best_genome.npz")
    parser.add_argument("--plot", action="store_true", help="Plot training logs")
    args = parser.parse_args()

    if args.train:
        ga = GA(pop_size=args.pop, elitism=0.1, cx_rate=0.7, mut_rate=0.02, mut_sigma=0.2, seed=args.seed)
        logs, best = ga.run(n_gen=args.gen, eval_episodes=args.episodes, seed=args.seed,
                            log_path=args.log, model_path=args.model)
        if args.plot:
            plot_logs(args.log)
    elif args.watch:
        watch(model_path=args.model, seed=args.seed)
    else:
        print("Use --train to train or --watch to watch best agent.")

if __name__ == "__main__":
    main()


#===== python flappy_ga.py --train --pop 100 --gen 50 --plot <--- lệnh này là để train AI theo số lần gen
#===== python flappy_ga.py --watch <---- lệnh này để chạy xem con giỏi nhất được train
#===== pip install --pre numpy pygame matplotlib <---- cài numpy
