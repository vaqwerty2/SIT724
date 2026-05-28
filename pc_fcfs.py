import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter


# ================= CONSTANTS =================
DT = 0.1
ROBOT_RADIUS = 0.1
LANE_HALF_WIDTH = 0.2
INTERSECTION_RADIUS = 0.4
WORLD_HALF_SIZE = 3.0
SPEED = 0.4

APPROACH_MARGIN = 0.5
STOP_MARGIN = 0.3


# ================= ROBOT =================
class Robot:
    def __init__(self, x, y, theta, goal_x, goal_y, idx, start_time=0.0, direction=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.idx = idx
        self.direction = direction
        self.v = 0.0
        self.stopped = False
        self.reached = False
        self.start_time = start_time

        # Metrics
        self.wait_time = 0.0
        self.finish_time = None
        self.prev_stopped = False
        self.num_stops = 0

        # Stop interval logging
        self.stop_intervals = []
        self._current_stop_start = None

    def update(self, dt, t_now):
        if t_now < self.start_time or self.reached:
            return

        # Stop interval detection
        if self.stopped and not self.prev_stopped:
            self._current_stop_start = t_now
            self.num_stops += 1
        if not self.stopped and self.prev_stopped:
            if self._current_stop_start is not None:
                self.stop_intervals.append((self._current_stop_start, t_now))
                self._current_stop_start = None
        self.prev_stopped = self.stopped

        if self.stopped:
            self.wait_time += dt
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
            if self._current_stop_start is not None:
                self.stop_intervals.append((self._current_stop_start, t_now))
                self._current_stop_start = None
            return

        self.theta = math.atan2(dy, dx)
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt


# ================= PARALLEL-COMPATIBLE FCFS =================
class ParallelFCFSCoordinator:
    """
    Parallel-Compatible FCFS (PC-FCFS):

    - Maintains FCFS arrival order.
    - Allows MULTIPLE robots to occupy the intersection
      if and only if their movements are parallel and non-conflicting.
    - Compatible pairs:
        (left -> right) + (right -> left)
        (top -> bottom) + (bottom -> top)
    """

    def __init__(self):
        self.zone_center = (0.0, 0.0)
        self.zone_radius = INTERSECTION_RADIUS
        self.queue = []
        self.current_occupants = set()
        self.collision_errors = 0

    def _distance_to_zone(self, robot):
        return math.hypot(robot.x, robot.y)

    def _is_parallel_compatible(self, r1, r2):
        # Horizontal pair
        if {r1.direction, r2.direction} == {0, 1}:
            return True
        # Vertical pair
        if {r1.direction, r2.direction} == {2, 3}:
            return True
        return False

    def step(self, robots, t_now):
        idx_to_robot = {r.idx: r for r in robots}

        to_remove = []
        for idx in self.current_occupants:
            r = idx_to_robot[idx]
            if self._distance_to_zone(r) > self.zone_radius + APPROACH_MARGIN or r.reached:
                to_remove.append(idx)
        for idx in to_remove:
            self.current_occupants.remove(idx)
            if self.queue and self.queue[0] == idx:
                self.queue.pop(0)

        for r in robots:
            if r.reached:
                continue
            if self._distance_to_zone(r) < self.zone_radius + APPROACH_MARGIN:
                if r.idx not in self.queue:
                    self.queue.append(r.idx)

        if not self.current_occupants and self.queue:
            first = self.queue[0]
            self.current_occupants.add(first)

            for idx in self.queue[1:]:
                r1 = idx_to_robot[first]
                r2 = idx_to_robot[idx]
                if self._is_parallel_compatible(r1, r2):
                    self.current_occupants.add(idx)
                else:
                    break

        for r in robots:
            if r.reached:
                r.stopped = False
                continue

            d = self._distance_to_zone(r)

            if d >= self.zone_radius + APPROACH_MARGIN:
                r.stopped = False
            elif r.idx in self.current_occupants:
                r.stopped = False
            else:
                r.stopped = d < self.zone_radius + STOP_MARGIN

        # Safety check
        in_zone = [r for r in robots if self._distance_to_zone(r) < self.zone_radius]
        for i in range(len(in_zone)):
            for j in range(i + 1, len(in_zone)):
                if not self._is_parallel_compatible(in_zone[i], in_zone[j]):
                    self.collision_errors += 1
                    print(f"[ERROR t={t_now:.2f}] Collision risk between {in_zone[i].idx} and {in_zone[j].idx}")


# ================= ROBOT CREATION =================
def create_robots(num_robots=4):
    robots = []
    base_dirs = [0, 1, 2, 3]
    base_start = 0.0
    delta_start = 1.0

    for idx in range(num_robots):
        d = base_dirs[idx % 4]
        start_time = base_start + idx * delta_start
        offset = 0.3 * (idx // 4)

        if d == 0:
            x, y, gx, gy, th = -WORLD_HALF_SIZE - offset, LANE_HALF_WIDTH, WORLD_HALF_SIZE, LANE_HALF_WIDTH, 0.0
        elif d == 1:
            x, y, gx, gy, th = WORLD_HALF_SIZE + offset, -LANE_HALF_WIDTH, -WORLD_HALF_SIZE, -LANE_HALF_WIDTH, math.pi
        elif d == 2:
            x, y, gx, gy, th = -LANE_HALF_WIDTH, -WORLD_HALF_SIZE - offset, -LANE_HALF_WIDTH, WORLD_HALF_SIZE, math.pi / 2
        else:
            x, y, gx, gy, th = LANE_HALF_WIDTH, WORLD_HALF_SIZE + offset, LANE_HALF_WIDTH, -WORLD_HALF_SIZE, -math.pi / 2

        robots.append(Robot(x, y, th, gx, gy, idx, start_time, d))

    return robots


# ================= VISUAL =================
def draw_world(ax):
    ax.clear()
    ax.set_xlim(-WORLD_HALF_SIZE, WORLD_HALF_SIZE)
    ax.set_ylim(-WORLD_HALF_SIZE, WORLD_HALF_SIZE)
    ax.set_aspect('equal', adjustable='box')

    ax.axhline(y=LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)

    ax.add_patch(plt.Circle((0, 0), INTERSECTION_RADIUS, color='red', alpha=0.2))


# ================= SIM =================
def run_mini_sim_pc_fcfs(num_robots=7, save_video=True):
    robots = create_robots(num_robots)
    coordinator = ParallelFCFSCoordinator()

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

        draw_world(ax)
        xs = [r.x for r in robots]
        ys = [r.y for r in robots]
        colors = [
            'green' if r.reached else
            'gray' if t_now < r.start_time else
            'orange' if r.stopped else
            'blue'
            for r in robots
        ]
        ax.scatter(xs, ys, c=colors)
        ax.set_title(f"PC-FCFS (Parallel-Compatible FCFS) | N={len(robots)} | t={t_now:.1f}s")
        return []

    
    frames = 900 if num_robots > 30 else (700 if num_robots > 8 else 500)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=50, repeat=False, init_func=init
    )

    if save_video:
        writer = FFMpegWriter(fps=20, bitrate=1800)
        ani.save("mini_sim_pc_fcfs.mp4", writer=writer)
        plt.close(fig)
    else:
        plt.show()

    # ================= OUTPUT =================
    finished = [r for r in robots if r.finish_time is not None]
    n = len(finished)

    if n > 0:
        avg_wait = sum(r.wait_time for r in finished) / n
        avg_travel = sum(r.finish_time - r.start_time for r in finished) / n
        avg_stops = sum(len(r.stop_intervals) for r in finished) / n
    else:
        avg_wait = 0.0
        avg_travel = 0.0
        avg_stops = 0.0

    print("=== PC-FCFS RESULTS ===")
    print(f"Robots finished:     {n}")
    print(f"Average wait time:   {avg_wait:.2f} s")
    print(f"Average travel time: {avg_travel:.2f} s")
    print(f"Average #stops:      {avg_stops:.2f}")
    print(f"Collision errors (multi in zone): {coordinator.collision_errors}")
    print("===================================\n")

    output_file = "pc_fcfs_robot_stop_log.txt"
    with open(output_file, "w") as f:
        f.write("=== PC-FCFS RESULTS ===\n")
        f.write(f"Robots finished:     {n}\n")
        f.write(f"Average wait time:   {avg_wait:.2f} s\n")
        f.write(f"Average travel time: {avg_travel:.2f} s\n")
        f.write(f"Average #stops:      {avg_stops:.2f}\n")
        f.write(f"Collision errors (multi in zone): {coordinator.collision_errors}\n")
        f.write("===================================\n\n")
        f.write("Robot stop intervals (seconds):\n\n")

        for r in robots:
            print(f"Robot {r.idx} stop intervals:")
            f.write(f"Robot {r.idx} stop intervals:\n")
            if not r.stop_intervals:
                print("  None")
                f.write("  None\n")
            else:
                for s, e in r.stop_intervals:
                    print(f"  {s:.2f} → {e:.2f}")
                    f.write(f"  {s:.2f} → {e:.2f}\n")
            print()
            f.write("\n")

    print(f"Stop timing log written to {output_file}")


if __name__ == "__main__":
    run_mini_sim_pc_fcfs(num_robots=15, save_video=True)
