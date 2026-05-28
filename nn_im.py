import math
import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation


# ================= CONSTANTS =================
DT = 0.1
LANE_HALF_WIDTH = 0.2
INTERSECTION_RADIUS = 0.4
WORLD_HALF_SIZE = 3.0
SPEED = 0.4

APPROACH_MARGIN = 0.5
STOP_MARGIN = 0.15 

MAX_ROBOTS = 15 
TOP_K = 6

# ================= MODEL =================
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MAX_ROBOTS*6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, MAX_ROBOTS)
        )

    def forward(self, x):
        return self.net(x)

# ================= ROBOT =================
class Robot:
    def __init__(self, x, y, gx, gy, idx, start_time, direction):
        self.x = x
        self.y = y
        self.goal_x = gx
        self.goal_y = gy
        self.idx = idx
        self.start_time = start_time
        self.direction = direction

        self.theta = 0
        self.v = 0.0
        self.stopped = False
        self.reached = False

        self.wait_time = 0.0
        self.finish_time = None

        # ===== STOP LOGGING =====
        self.prev_stopped = False
        self.num_stops = 0

        self.stop_intervals = []
        self._current_stop_start = None

    def update(self, dt, t):
        if t < self.start_time or self.reached:
            return

        # detect stop start
        if self.stopped and not self.prev_stopped:
            self._current_stop_start = t
            self.num_stops += 1

        # detect stop end
        if not self.stopped and self.prev_stopped:
            if self._current_stop_start is not None:
                self.stop_intervals.append(
                    (self._current_stop_start, t)
                )
                self._current_stop_start = None

        self.prev_stopped = self.stopped

        if self.stopped:
            self.wait_time += dt

        self.v = 0 if self.stopped else SPEED

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.hypot(dx, dy)

        if dist < 0.05:
            self.reached = True
            self.finish_time = t

            if self._current_stop_start is not None:
                self.stop_intervals.append(
                    (self._current_stop_start, t)
                )
                self._current_stop_start = None

            return

        self.theta = math.atan2(dy, dx)
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

# ================= COORDINATOR =================
class NeuralCoordinator:
    def __init__(self):
        checkpoint = torch.load("nn_controller_best.pth", map_location="cpu")

        self.model = NN()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.threshold = checkpoint["threshold"]
        self.scaler = joblib.load("scaler.pkl")

        self.collisions = set()

    def dist(self, r):
        return math.hypot(r.x, r.y)

    def crossed(self, r):
        if r.direction == 0: return r.x > 0
        if r.direction == 1: return r.x < 0
        if r.direction == 2: return r.y > 0
        if r.direction == 3: return r.y < 0

    def is_conflict(self, r1, r2):
        if {r1.direction, r2.direction} == {0,1}: return False
        if {r1.direction, r2.direction} == {2,3}: return False
        return True

    def is_behind(self, r1, r2):
        if r1.direction != r2.direction:
            return False

        if r1.direction == 0: return r1.x < r2.x
        if r1.direction == 1: return r1.x > r2.x
        if r1.direction == 2: return r1.y < r2.y
        if r1.direction == 3: return r1.y > r2.y

    def build_state(self, robots, t):
        state = []
        robots_sorted = sorted(robots, key=lambda r: r.idx)

        for i in range(MAX_ROBOTS):
            if i < len(robots_sorted):
                r = robots_sorted[i]
                vx = r.v * math.cos(r.theta)
                vy = r.v * math.sin(r.theta)
                active = int(t >= r.start_time and not r.reached)
                state += [r.x, r.y, vx, vy, r.direction, active]
            else:
                state += [0]*6

        return np.array(state).reshape(1, -1), robots_sorted

    def step(self, robots, t):
        state, sorted_r = self.build_state(robots, t)
        state = self.scaler.transform(state)

        with torch.no_grad():
            logits = self.model(torch.tensor(state, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy()[0]

        #threshold selection
        sorted_indices = np.argsort(-probs[:len(sorted_r)])

        valid_indices = []
        for i in sorted_indices:
            if probs[i] >= (self.threshold - 0.15):
                valid_indices.append(i)

        if len(valid_indices) == 0:
            valid_indices = sorted_indices[:TOP_K]

        allowed = []

        for idx in valid_indices:
            r = sorted_r[idx]

            if t < r.start_time or r.reached:
                continue

            d = self.dist(r)

            # earlier decision region
            if d > INTERSECTION_RADIUS + 1.0:
                continue

            # lane ordering
            blocked = False
            for other in sorted_r:
                if other == r:
                    continue
                if self.is_behind(r, other):
                    if self.dist(other) < self.dist(r):
                        blocked = True
                        break

            if blocked:
                continue

            # stronger safety near intersection
            safe = True
            for a in allowed:
                if self.is_conflict(r, a):
                    if self.dist(r) < 0.5:
                        safe = False
                        break

            if safe:
                allowed.append(r)

            if len(allowed) >= TOP_K:
                break

        # APPLY CONTROL
        for r in robots:
            if r.reached or t < r.start_time:
                r.stopped = False
                continue

            d = self.dist(r)

            if d < INTERSECTION_RADIUS + STOP_MARGIN:
                if self.crossed(r):
                    r.stopped = False
                else:
                    r.stopped = (r not in allowed)
            else:
                r.stopped = False

        # collision tracking
        in_zone = [r for r in robots if self.dist(r) < INTERSECTION_RADIUS]

        for i in range(len(in_zone)):
            for j in range(i+1, len(in_zone)):
                if self.is_conflict(in_zone[i], in_zone[j]):
                    pair = tuple(sorted((in_zone[i].idx, in_zone[j].idx)))
                    self.collisions.add(pair)

# ================= CREATE ROBOTS =================
def create_robots(n=15):
    robots = []
    for i in range(n):
        d = i % 4
        start_time = i * 1.0

        if d == 0:
            x,y,gx,gy = -3,0.2,3,0.2
        elif d == 1:
            x,y,gx,gy = 3,-0.2,-3,-0.2
        elif d == 2:
            x,y,gx,gy = 0.2,-3,0.2,3
        else:
            x,y,gx,gy = -0.2,3,-0.2,-3

        robots.append(Robot(x,y,gx,gy,i,start_time,d))
    return robots

# ================= SIM =================
def run():
    robots = create_robots()
    coord = NeuralCoordinator()

    fig, ax = plt.subplots()
    t = 0

    def update(frame):
        nonlocal t
        t = frame * DT

        coord.step(robots, t)

        for r in robots:
            r.update(DT, t)

        if all(r.reached for r in robots):

            finished = [r for r in robots if r.finish_time]

            n = len(finished)

            avg_wait = sum(r.wait_time for r in finished)/n

            avg_travel = sum(
                (r.finish_time - r.start_time)
                for r in finished
            ) / n

            avg_stops = sum(
                r.num_stops for r in finished
            ) / n

            print("\n=== NN-IM RESULTS ===")
            print(f"Robots finished:     {n}")
            print(f"Average wait time:   {avg_wait:.2f} s")
            print(f"Average travel time: {avg_travel:.2f} s")
            print(f"Average #stops:      {avg_stops:.2f}")
            print(f"Collision errors:    {len(coord.collisions)}")
            print("===================================")

            # ===== FILE OUTPUT =====
            output_file = "nn_robot_stop_log.txt"

            with open(output_file, "w") as f:

                f.write("=== NN-IM RESULTS ===\n")
                f.write(f"Robots finished:     {n}\n")
                f.write(f"Average wait time:   {avg_wait:.2f} s\n")
                f.write(f"Average travel time: {avg_travel:.2f} s\n")
                f.write(f"Average #stops:      {avg_stops:.2f}\n")
                f.write(f"Collision errors:    {len(coord.collisions)}\n")
                f.write("===================================\n\n")

                f.write("Robot stop intervals (seconds):\n\n")

                for r in robots:

                    f.write(f"Robot {r.idx} stop intervals:\n")

                    if not r.stop_intervals:
                        f.write("  None\n")

                    else:
                        for s, e in r.stop_intervals:
                            f.write(f"  {s:.2f} -> {e:.2f}\n")

                    f.write("\n")

            print(f"Stop timing log written to {output_file}")

            anim.event_source.stop()
            return []

        ax.clear()
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        ax.set_aspect('equal')

        ax.axhline(y=LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
        ax.axhline(y=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(x=LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(x=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)

        ax.add_patch(plt.Circle((0,0), INTERSECTION_RADIUS, color='red', alpha=0.2))

        xs = [r.x for r in robots]
        ys = [r.y for r in robots]

        colors = ['green' if r.reached else 'orange' if r.stopped else 'blue' for r in robots]

        ax.scatter(xs, ys, c=colors)
        ax.set_title(f"t={t:.1f}")

        return []

    global anim
    anim = animation.FuncAnimation(fig, update, frames=1000, interval=50)
    plt.show()

# ================= MAIN =================
if __name__ == "__main__":
    run()
