import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# ================= CONSTANTS (match your other sims) =================
DT = 0.1
ROBOT_RADIUS = 0.1
LANE_HALF_WIDTH = 0.2
INTERSECTION_RADIUS = 0.4
WORLD_HALF_SIZE = 3.0
SPEED = 0.4

APPROACH_MARGIN = 0.5
STOP_MARGIN = 0.3

# Safety gap used in the conflict-matrix logic (seconds)
SAFETY_GAP = 1.0


# ================= ROBOT MODEL =================
class Robot:
    def __init__(self, x, y, theta, goal_x, goal_y, idx,
                 lane_id, start_time=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.idx = idx
        self.lane_id = lane_id   # 0: left->right, 1: right->left, 2: bottom->top, 3: top->bottom
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

        # For matrix-based IM: assigned release time at intersection
        self.release_time = None

    def update(self, dt, t_now):
        # Not started or already finished
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

        # Waiting and speed
        if self.stopped:
            self.wait_time += dt
            self.v = 0.0
        else:
            self.v = SPEED

        # Motion towards goal
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


# ================= MATRIX-BASED IM COORDINATOR =================
class MatrixIMCoordinator:
    """
    Matrix-based intersection manager (Li & Liu style, simplified).

    - Each robot belongs to a lane_id in {0,1,2,3}.
    - A 4x4 conflict matrix encodes which lane pairs conflict.
    - The coordinator maintains, for each lane, the latest "leave time"
      of any robot whose passage has been reserved/observed.
    - When a robot first enters the approach region, it is assigned
      a release_time = max(leave times of conflicting lanes) + SAFETY_GAP.
    - The robot waits near the intersection until t_now >= release_time,
      then is allowed to traverse at constant SPEED.
    """

    def __init__(self):
        self.zone_center = (0.0, 0.0)
        self.zone_radius = INTERSECTION_RADIUS

        # Simple lane conflict matrix: rows = lane i, cols = lane j.
        # 1 means i and j conflict inside the intersection.
        # Here: opposite lanes do NOT conflict, perpendicular lanes DO.
        self.conflict_matrix = [
            [0, 0, 1, 1],  # lane 0 (L->R) conflicts with lanes 2,3
            [0, 0, 1, 1],  # lane 1 (R->L) conflicts with lanes 2,3
            [1, 1, 0, 0],  # lane 2 (B->T) conflicts with lanes 0,1
            [1, 1, 0, 0],  # lane 3 (T->B) conflicts with lanes 0,1
        ]

        # Latest reserved leave time per lane
        self.lane_leave_time = [0.0, 0.0, 0.0, 0.0]

        self.collision_errors = 0

    def _distance_to_zone(self, robot):
        cx, cy = self.zone_center
        return math.hypot(robot.x - cx, robot.y - cy)

    def _estimate_traverse_time(self, robot):
        """
        Rough estimate of how long the robot will spend from entering
        the zone boundary (radius) to leaving it, assuming SPEED.
        """
        # Distance across diagonal of conflict disk + some margin
        path_length = 2.0 * self.zone_radius + 0.5
        return path_length / SPEED

    def step(self, robots, t_now):
        """
        Assign release times and issue stop/go commands.
        """
        # First, for robots that have not yet been assigned a release time
        # and are in the approach region, assign one based on conflict matrix.
        for r in robots:
            if r.reached or t_now < r.start_time:
                continue

            d = self._distance_to_zone(r)
            # Only consider robots within approach region that don't have a release time yet
            if d < self.zone_radius + APPROACH_MARGIN and r.release_time is None:
                lane = r.lane_id
                # Find conflicting lanes
                conflicting_lanes = [
                    j for j in range(4) if self.conflict_matrix[lane][j] == 1
                ]
                # Max leave time among conflicting lanes
                max_leave = 0.0
                for j in conflicting_lanes:
                    max_leave = max(max_leave, self.lane_leave_time[j])
                # Assign release time
                r.release_time = max(t_now, max_leave + SAFETY_GAP)
                # Estimate when this robot will leave the zone
                traverse_time = self._estimate_traverse_time(r)
                self.lane_leave_time[lane] = r.release_time + traverse_time

        # Second, decide stop/go based on release_time and position
        for r in robots:
            if r.reached or t_now < r.start_time:
                r.stopped = False
                continue

            d = self._distance_to_zone(r)

            # Far away from intersection: always go
            if d >= self.zone_radius + APPROACH_MARGIN:
                r.stopped = False
                continue

            # If release_time not yet assigned (rare), stop near the zone
            if r.release_time is None:
                r.stopped = d < self.zone_radius + STOP_MARGIN
                continue

            # If we are before release_time and near the stop line, stop
            if t_now < r.release_time and d < self.zone_radius + STOP_MARGIN:
                r.stopped = True
            else:
                r.stopped = False

        # Simple safety check: if multiple robots inside the conflict disk,
        # and they belong to conflicting lanes, count an error.
        in_zone = [r for r in robots if self._distance_to_zone(r) < self.zone_radius]
        for i in range(len(in_zone)):
            for j in range(i + 1, len(in_zone)):
                li = in_zone[i].lane_id
                lj = in_zone[j].lane_id
                if self.conflict_matrix[li][lj] == 1:
                    self.collision_errors += 1
                    print(f"[ERROR t={t_now:.2f}] Conflicting robots in zone: "
                          f"{in_zone[i].idx} (lane {li}) and {in_zone[j].idx} (lane {lj})")


# ================= ROBOT CREATION (match your other scripts) =================
def create_robots_matrix_im(num_robots=15):
    robots = []
    base_dirs = [0, 1, 2, 3]  # lane_ids
    base_start = 0.0
    delta_start = 1.0

    for idx in range(num_robots):
        lane = base_dirs[idx % 4]
        start_time = base_start + idx * delta_start
        offset = 0.3 * (idx // 4)

        if lane == 0:  # left -> right
            x = -WORLD_HALF_SIZE - offset
            y = LANE_HALF_WIDTH
            gx = WORLD_HALF_SIZE
            gy = LANE_HALF_WIDTH
            theta = 0.0
        elif lane == 1:  # right -> left
            x = WORLD_HALF_SIZE + offset
            y = -LANE_HALF_WIDTH
            gx = -WORLD_HALF_SIZE
            gy = -LANE_HALF_WIDTH
            theta = math.pi
        elif lane == 2:  # bottom -> top
            x = -LANE_HALF_WIDTH
            y = -WORLD_HALF_SIZE - offset
            gx = -LANE_HALF_WIDTH
            gy = WORLD_HALF_SIZE
            theta = math.pi / 2.0
        else:  # lane == 3: top -> bottom
            x = LANE_HALF_WIDTH
            y = WORLD_HALF_SIZE + offset
            gx = LANE_HALF_WIDTH
            gy = -WORLD_HALF_SIZE
            theta = -math.pi / 2.0

        robots.append(
            Robot(x, y, theta, gx, gy, idx, lane_id=lane, start_time=start_time)
        )

    return robots


# ================= DRAW WORLD =================
def draw_world(ax):
    ax.clear()
    ax.set_xlim(-WORLD_HALF_SIZE, WORLD_HALF_SIZE)
    ax.set_ylim(-WORLD_HALF_SIZE, WORLD_HALF_SIZE)
    ax.set_aspect('equal', adjustable='box')

    # Lanes
    ax.axhline(y=LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)

    # Conflict zone
    ax.add_patch(plt.Circle((0, 0), INTERSECTION_RADIUS, color='red', alpha=0.2))


# ================= SUMMARY =================
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
    print(f"Collision errors:    {collision_errors}")
    print("===================================")

    output_file = "matrix_im_robot_stop_log.txt"
    with open(output_file, "w") as f:
        f.write("=== MATRIX-IM RESULTS ===\n")
        f.write(f"Robots finished:     {n}\n")
        f.write(f"Average wait time:   {avg_wait:.2f} s\n")
        f.write(f"Average travel time: {avg_travel:.2f} s\n")
        f.write(f"Average #stops:      {avg_stops:.2f}\n")
        f.write(f"Collision errors:    {collision_errors}\n")
        f.write("===================================\n\n")
        f.write("Robot stop intervals (seconds):\n\n")
        for r in robots:
            f.write(f"Robot {r.idx} (lane {r.lane_id}) stop intervals:\n")
            if not r.stop_intervals:
                f.write("  None\n")
            else:
                for s, e in r.stop_intervals:
                    f.write(f"  {s:.2f} -> {e:.2f}\n")
            f.write("\n")

    print(f"Stop timing log written to {output_file}")


# ================= SIMULATION LOOP =================
def run_mini_sim_matrix_im(num_robots=15, save_video=True):
    robots = create_robots_matrix_im(num_robots=num_robots)
    coordinator = MatrixIMCoordinator()

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
        ax.set_title(
            f"Matrix-IM (conflict matrix) - N={len(robots)}, "
            f"Frame {frame}, t={t_now:.1f}s"
        )
        return []

    frames = 700 if num_robots > 8 else 500

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=50, blit=False, repeat=False
    )

    if save_video:
        writer = FFMpegWriter(fps=20, bitrate=1800)
        ani.save("mini_sim_matrix_im.mp4", writer=writer)
        plt.close(fig)
    else:
        plt.show()

    summarize_results(robots, "MATRIX-IM", coordinator.collision_errors)


if __name__ == "__main__":
    run_mini_sim_matrix_im(num_robots=15, save_video=True)
