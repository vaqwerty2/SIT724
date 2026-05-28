import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

import csv

# ===== DATASET LOGGER =====
DATASET_FILE = "fcfs_dataset.csv"

def init_dataset():
    with open(DATASET_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        MAX_ROBOTS = 20  

        for i in range(MAX_ROBOTS):
            header += [
                f"r{i}_x", f"r{i}_y",
                f"r{i}_vx", f"r{i}_vy",
                f"r{i}_dir",
                f"r{i}_active"
            ]

        for i in range(MAX_ROBOTS):
            header.append(f"r{i}_stopped")

        writer.writerow(header)


def log_state_action(robots, t_now):
    MAX_ROBOTS = 20
    row = []

    # SORT robots by index (important consistency)
    robots_sorted = sorted(robots, key=lambda r: r.idx)

    for i in range(MAX_ROBOTS):
        if i < len(robots_sorted):
            r = robots_sorted[i]

            vx = r.v * math.cos(r.theta)
            vy = r.v * math.sin(r.theta)

            active = 1 if (t_now >= r.start_time and not r.reached) else 0

            row += [
                r.x, r.y,
                vx, vy,
                r.direction,
                active
            ]
        else:
            row += [0, 0, 0, 0, 0, 0]

    # ACTIONS (labels)
    for i in range(MAX_ROBOTS):
        if i < len(robots_sorted):
            r = robots_sorted[i]
            row.append(1 if r.stopped else 0)
        else:
            row.append(0)

    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# Simple constants
DT = 0.1
ROBOT_RADIUS = 0.1
LANE_HALF_WIDTH = 0.2
INTERSECTION_RADIUS = 0.4  # defines the conflict zone
WORLD_HALF_SIZE = 3.0
SPEED = 0.4  # m/s

# How far from the center we consider "approach region"
APPROACH_MARGIN = 0.5
STOP_MARGIN = 0.3


class Robot:
    def __init__(self, x, y, theta, goal_x, goal_y, idx, start_time=0.0, direction=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.idx = idx
        self.direction = direction  # 0,1,2,3 for lane
        self.v = 0.0
        self.stopped = False
        self.reached = False
        self.start_time = start_time

        # Logging / metrics
        self.wait_time = 0.0
        self.first_move_time = None
        self.finish_time = None
        self.prev_stopped = False
        self.num_stops = 0

        # stop interval logging
        self.stop_intervals = []
        self._current_stop_start = None

    def pose(self):
        return (self.x, self.y)

    def update(self, dt, t_now):
        if t_now < self.start_time:
            return

        if self.reached:
            return

        # detect stop start
        if self.stopped and not self.prev_stopped:
            self._current_stop_start = t_now

        # detect stop end
        if not self.stopped and self.prev_stopped:
            if self._current_stop_start is not None:
                self.stop_intervals.append(
                    (self._current_stop_start, t_now)
                )
                self._current_stop_start = None

        # Logging: waiting time and stops
        if self.stopped:
            self.wait_time += dt
        if (not self.prev_stopped) and self.stopped:
            self.num_stops += 1
        self.prev_stopped = self.stopped

        if (self.first_move_time is None) and (not self.stopped):
            self.first_move_time = t_now

        if self.stopped:
            self.v = 0.0
        else:
            self.v = SPEED

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.hypot(dx, dy)

        if dist < 0.05:
            self.reached = True
            self.v = 0.0
            if self.finish_time is None:
                self.finish_time = t_now

            # close open stop if goal reached
            if self._current_stop_start is not None:
                self.stop_intervals.append(
                    (self._current_stop_start, t_now)
                )
                self._current_stop_start = None
            return

        self.theta = math.atan2(dy, dx)
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt


class FCFSCoordinator:
    def __init__(self):
        self.zone_center = (0.0, 0.0)
        self.zone_radius = INTERSECTION_RADIUS
        self.queue = []
        self.current_occupant = None
        self.collision_errors = 0

    def _distance_to_zone(self, robot):
        cx, cy = self.zone_center
        return math.hypot(robot.x - cx, robot.y - cy)

    def step(self, robots, t_now):
        idx_to_robot = {r.idx: r for r in robots}

        if self.current_occupant is not None:
            occ = idx_to_robot[self.current_occupant]
            d_occ = self._distance_to_zone(occ)
            if d_occ > self.zone_radius + APPROACH_MARGIN or occ.reached:
                old_idx = self.current_occupant
                self.current_occupant = None
                if self.queue and self.queue[0] == old_idx:
                    self.queue.pop(0)

        for r in robots:
            if r.reached:
                continue
            d = self._distance_to_zone(r)
            if d < self.zone_radius + APPROACH_MARGIN:
                if r.idx not in self.queue:
                    self.queue.append(r.idx)

        if self.current_occupant is None and self.queue:
            self.current_occupant = self.queue[0]

        for r in robots:
            if r.reached:
                r.stopped = False
                continue

            d = self._distance_to_zone(r)

            if d >= self.zone_radius + APPROACH_MARGIN:
                r.stopped = False
                continue

            if self.current_occupant is None:
                if (d < self.zone_radius + STOP_MARGIN) and (r.idx in self.queue) and (r.idx != self.queue[0]):
                    r.stopped = True
                else:
                    r.stopped = False
            else:
                if r.idx == self.current_occupant:
                    r.stopped = False
                else:
                    if (d < self.zone_radius + STOP_MARGIN) and (r.idx in self.queue):
                        r.stopped = True
                    else:
                        r.stopped = False

        in_zone = [r for r in robots if self._distance_to_zone(r) < self.zone_radius]
        if len(in_zone) > 1:
            self.collision_errors += 1
            ids = [r.idx for r in in_zone]
            print(f"[ERROR t={t_now:.2f}] Multiple robots in intersection zone: {ids}")


def create_robots_fcfs(num_robots=4):
    if num_robots < 2 or num_robots > 100:
        raise ValueError("num_robots must be between 2 and 100")

    robots = []
    base_dirs = [0, 1, 2, 3]
    base_start = 0.0
    delta_start = 1.0

    for idx in range(num_robots):
        d = base_dirs[idx % 4]
        start_time = base_start + idx * delta_start
        lane_offset = 0.3 * (idx // 4)

        if d == 0:
            x = -WORLD_HALF_SIZE - lane_offset
            y = LANE_HALF_WIDTH
            theta = 0.0
            goal_x = WORLD_HALF_SIZE
            goal_y = LANE_HALF_WIDTH
        elif d == 1:
            x = WORLD_HALF_SIZE + lane_offset
            y = -LANE_HALF_WIDTH
            theta = math.pi
            goal_x = -WORLD_HALF_SIZE
            goal_y = -LANE_HALF_WIDTH
        elif d == 2:
            x = -LANE_HALF_WIDTH
            y = -WORLD_HALF_SIZE - lane_offset
            theta = math.pi / 2
            goal_x = -LANE_HALF_WIDTH
            goal_y = WORLD_HALF_SIZE
        else:
            x = LANE_HALF_WIDTH
            y = WORLD_HALF_SIZE + lane_offset
            theta = -math.pi / 2
            goal_x = LANE_HALF_WIDTH
            goal_y = -WORLD_HALF_SIZE

        robots.append(Robot(x, y, theta, goal_x, goal_y, idx, start_time, d))
    return robots


def draw_world(ax):
    ax.clear()
    ax.set_xlim(-WORLD_HALF_SIZE, WORLD_HALF_SIZE)
    ax.set_ylim(-WORLD_HALF_SIZE, WORLD_HALF_SIZE)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(y=LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    circle = plt.Circle((0, 0), INTERSECTION_RADIUS, color='red', alpha=0.2)
    ax.add_patch(circle)


def summarize_results(robots, coordinator_label, collision_errors):
    finished = [r for r in robots if r.finish_time is not None]
    if not finished:
        print(f"--- {coordinator_label}: no robots finished ---")
        return

    n = len(finished)
    avg_wait = sum(r.wait_time for r in finished) / n
    avg_travel = sum((r.finish_time - r.start_time) for r in finished) / n
    avg_stops = sum(r.num_stops for r in finished) / n

    print(f"=== {coordinator_label} RESULTS ===")
    print(f"Robots finished:     {n}")
    print(f"Average wait time:   {avg_wait:.2f} s")
    print(f"Average travel time: {avg_travel:.2f} s")
    print(f"Average #stops:      {avg_stops:.2f}")
    print(f"Collision errors (multi in zone): {collision_errors}")
    print("===================================")

    # print & save stop timings
    output_file = "fcfs_robot_stop_log.txt"
    with open(output_file, "w") as f:
        f.write("=== FCFS ===\n")
        f.write(f"Robots finished:     {n}\n")
        f.write(f"Average wait time:   {avg_wait:.2f} s\n")
        f.write(f"Average travel time: {avg_travel:.2f} s\n")
        f.write(f"Average #stops:      {avg_stops:.2f}\n")
        f.write(f"Collision errors (multi in zone): {collision_errors}\n")
        f.write("===================================\n\n")

        f.write("Robot stop intervals (seconds):\n\n")
        for r in robots:
            print(f"Robot {r.idx} stop intervals:")
            f.write(f"Robot {r.idx} stop intervals:\n")
            if not r.stop_intervals:
                print("  None")
                f.write("  None\n")
            else:
                for start, end in r.stop_intervals:
                    print(f"  {start:.2f} → {end:.2f}")
                    f.write(f"  {start:.2f} → {end:.2f}\n")
            print()
            f.write("\n")

    print(f"Stop timing log written to {output_file}")


def run_mini_sim_fcfs(num_robots=4, save_video=True):
    init_dataset()
    robots = create_robots_fcfs(num_robots=num_robots)
    coordinator = FCFSCoordinator()

    fig, ax = plt.subplots()
    t_now = 0.0

    def init():
        draw_world(ax)
        return []

    def update(frame):
        nonlocal t_now
        t_now = frame * DT
        coordinator.step(robots, t_now)
        for r in robots:
            r.update(DT, t_now)
        log_state_action(robots, t_now)

        draw_world(ax)
        xs = [r.x for r in robots]
        ys = [r.y for r in robots]
        colors = []
        for r in robots:
            if r.reached:
                colors.append('green')
            elif t_now < r.start_time:
                colors.append('gray')
            elif r.stopped:
                colors.append('orange')
            else:
                colors.append('blue')

        ax.scatter(xs, ys, c=colors)
        ax.set_title(f"FCFS - N={len(robots)}, Frame {frame}, t={t_now:.1f}s")
        return []

    
    frames = 1000 if num_robots > 30 else (700 if num_robots > 8 else 500)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=50, blit=False, repeat=False
    )

    if save_video:
        writer = FFMpegWriter(fps=20, bitrate=1800)
        ani.save("mini_sim_fcfs.mp4", writer=writer)
        plt.close(fig)
    else:
        plt.show()

    summarize_results(robots, "FCFS", coordinator.collision_errors)


if __name__ == "__main__":
    run_mini_sim_fcfs(num_robots=15, save_video=True)
