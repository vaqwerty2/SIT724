import math
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# Simple constants
DT = 0.1
ROBOT_RADIUS = 0.1
LANE_HALF_WIDTH = 0.2
INTERSECTION_RADIUS = 0.4  # defines the conflict zone
WORLD_HALF_SIZE = 3.0
SPEED = 0.4  # m/s

APPROACH_MARGIN = 0.5  # how far from center is "approach region"
STOP_MARGIN = 0.3      # where robots actually get stopped


class Robot:
    def __init__(self, x, y, theta, goal_x, goal_y, idx, color, start_time=0.0, direction=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.idx = idx
        self.color = color
        self.direction = direction
        self.v = 0.0          # start stopped, only move after start_time
        self.stopped = False
        self.reached = False
        self.start_time = start_time

        # Metrics
        self.wait_time = 0.0
        self.first_move_time = None
        self.finish_time = None
        self.prev_stopped = False
        self.num_stops = 0

    def pose(self):
        return (self.x, self.y)

    def update(self, dt, t_now):
        # If not yet started, do nothing
        if t_now < self.start_time:
            return

        if self.reached:
            return

        # Logging: waiting time and stops
        if self.stopped:
            self.wait_time += dt
        if (not self.prev_stopped) and self.stopped:
            self.num_stops += 1
        self.prev_stopped = self.stopped

        # Record when it first actually moves
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
            return

        self.theta = math.atan2(dy, dx)
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt


class ColorReservationCoordinator:
    """
    Color-based reservation intersection manager:
    - Each robot has a color (e.g. 'red', 'blue', ...).
    - A global color priority order is used.
    - Among robots near the intersection, the robot whose color has
      the highest priority gets the reservation.
    - Once a robot leaves the approach region, it is never stopped again.
    Also detects multi-robot occupancy in the zone (optional metric).
    """
    def __init__(self, color_priority_order):
        self.zone_center = (0.0, 0.0)
        self.zone_radius = INTERSECTION_RADIUS
        self.color_priority_order = list(color_priority_order)
        self.current_occupant = None  # robot index
        self.collision_errors = 0

    def _distance_to_zone(self, robot):
        cx, cy = self.zone_center
        return math.hypot(robot.x - cx, robot.y - cy)

    def _color_rank(self, color):
        # Lower index = higher priority
        if color in self.color_priority_order:
            return self.color_priority_order.index(color)
        else:
            return len(self.color_priority_order)  # lowest priority if unknown

    def step(self, robots, t_now):
        idx_to_robot = {r.idx: r for r in robots}

        # Check if current occupant has left the approach region
        if self.current_occupant is not None:
            occ = idx_to_robot[self.current_occupant]
            d_occ = self._distance_to_zone(occ)
            if d_occ > self.zone_radius + APPROACH_MARGIN or occ.reached:
                self.current_occupant = None

        # Determine which robots are close enough to be considered
        approaching = []
        for r in robots:
            if r.reached:
                continue
            d = self._distance_to_zone(r)
            if d < self.zone_radius + APPROACH_MARGIN:
                color_rank = self._color_rank(r.color)
                approaching.append((color_rank, d, r))

        # Decide occupant if free
        if self.current_occupant is None and approaching:
            # Sort by color priority first, then by distance (tie-break)
            approaching.sort(key=lambda t: (t[0], t[1]))
            _, _, chosen = approaching[0]
            self.current_occupant = chosen.idx

        # Issue stop/pass commands
        for r in robots:
            if r.reached:
                r.stopped = False
                continue

            d = self._distance_to_zone(r)

            # If robot is outside approach region, never stop it
            if d >= self.zone_radius + APPROACH_MARGIN:
                r.stopped = False
                continue

            if self.current_occupant is None:
                # No one has reservation yet: allow movement until explicitly stopped
                r.stopped = False
            else:
                if r.idx == self.current_occupant:
                    # Occupant is allowed to move
                    r.stopped = False
                else:
                    # If close to the zone, they must wait
                    if d < self.zone_radius + STOP_MARGIN:
                        r.stopped = True
                    else:
                        r.stopped = False

        # Optional: detect multiple robots in the core zone
        in_zone = [r for r in robots if self._distance_to_zone(r) < self.zone_radius]
        if len(in_zone) > 1:
            self.collision_errors += 1
            ids = [r.idx for r in in_zone]
            print(f"[ERROR t={t_now:.2f}] Multiple robots in intersection zone: {ids}")


def create_robots_reservation(num_robots=4):
    """
    Create between 2 and 15 robots with assigned colors.
    Directions are cycled:
      0: left -> right  (use south lane, y = -LANE_HALF_WIDTH)
      1: right -> left  (use north lane, y = +LANE_HALF_WIDTH)
      2: bottom -> top  (use right lane, x = +LANE_HALF_WIDTH)
      3: top -> bottom  (use left lane,  x = -LANE_HALF_WIDTH)
    """
    if num_robots < 2 or num_robots > 15:
        raise ValueError("num_robots must be between 2 and 15")

    robots = []
    directions = []
    base_dirs = [0, 1, 2, 3]
    for i in range(num_robots):
        directions.append(base_dirs[i % len(base_dirs)])

    # Assign colors (repeat if > len(color_list))
    color_list = ["red", "blue", "green", "yellow", "purple"]
    # Stagger start times
    base_start = 0.0
    delta_start = 1.5

    for idx, d in enumerate(directions):
        start_time = base_start + idx * delta_start
        color = color_list[idx % len(color_list)]

        if d == 0:  # left -> right, south lane
            x = -WORLD_HALF_SIZE
            y = -LANE_HALF_WIDTH
            theta = 0.0
            goal_x = WORLD_HALF_SIZE
            goal_y = -LANE_HALF_WIDTH
        elif d == 1:  # right -> left, north lane
            x = WORLD_HALF_SIZE
            y = LANE_HALF_WIDTH
            theta = math.pi
            goal_x = -WORLD_HALF_SIZE
            goal_y = LANE_HALF_WIDTH
        elif d == 2:  # bottom -> top, right lane
            x = LANE_HALF_WIDTH
            y = -WORLD_HALF_SIZE
            theta = math.pi / 2
            goal_x = LANE_HALF_WIDTH
            goal_y = WORLD_HALF_SIZE
        elif d == 3:  # top -> bottom, left lane
            x = -LANE_HALF_WIDTH
            y = WORLD_HALF_SIZE
            theta = -math.pi / 2
            goal_x = -LANE_HALF_WIDTH
            goal_y = -WORLD_HALF_SIZE
        else:
            x = -WORLD_HALF_SIZE
            y = 0.0
            theta = 0.0
            goal_x = WORLD_HALF_SIZE
            goal_y = 0.0

        robots.append(Robot(x=x, y=y, theta=theta,
                            goal_x=goal_x, goal_y=goal_y,
                            idx=idx, color=color,
                            start_time=start_time, direction=d))
    return robots


def draw_world(ax, color_priority_order):
    ax.clear()
    ax.set_xlim(-WORLD_HALF_SIZE, WORLD_HALF_SIZE + 2.0)  # extra space on right for text
    ax.set_ylim(-WORLD_HALF_SIZE, WORLD_HALF_SIZE)
    ax.set_aspect('equal', adjustable='box')

    # Left-hand driving lanes:
    # Horizontal: westbound on y = -LANE_HALF_WIDTH, eastbound on y = +LANE_HALF_WIDTH
    ax.axhline(y=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=+LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    # Vertical: northbound on x = +LANE_HALF_WIDTH, southbound on x = -LANE_HALF_WIDTH
    ax.axvline(x=+LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=-LANE_HALF_WIDTH, color='gray', linestyle='--', linewidth=0.5)

    # Draw conflict zone
    circle = plt.Circle((0, 0), INTERSECTION_RADIUS, color='red', alpha=0.2)
    ax.add_patch(circle)

    # Show color priority on the right side
    x_text = WORLD_HALF_SIZE + 0.5
    y_top = WORLD_HALF_SIZE - 0.5
    ax.text(x_text, y_top + 0.3, "Priority order:", fontsize=10, fontweight="bold")
    for i, c in enumerate(color_priority_order):
        ax.text(x_text, y_top - i * 0.4, f"{i+1}: {c}", color=c, fontsize=9)


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


def write_results_csv(robots, filename="reservation_results.csv"):
    finished = [r for r in robots if r.finish_time is not None]
    fieldnames = [
        "id", "direction", "color",
        "start_time", "finish_time", "travel_time",
        "wait_time", "num_stops"
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in finished:
            travel_time = r.finish_time - r.start_time if r.finish_time is not None else None
            writer.writerow({
                "id": r.idx,
                "direction": r.direction,
                "color": r.color,
                "start_time": f"{r.start_time:.2f}",
                "finish_time": f"{r.finish_time:.2f}" if r.finish_time is not None else "",
                "travel_time": f"{travel_time:.2f}" if travel_time is not None else "",
                "wait_time": f"{r.wait_time:.2f}",
                "num_stops": r.num_stops,
            })
    print(f"Per-robot results written to {os.path.abspath(filename)}")


def run_mini_sim_reservation(num_robots=4, save_video=True):
    # Fixed color priority: red > blue > green > yellow > purple
    color_priority_order = ["red", "blue", "green", "yellow", "purple"]
    robots = create_robots_reservation(num_robots=num_robots)
    coordinator = ColorReservationCoordinator(color_priority_order)

    fig, ax = plt.subplots()
    scat = ax.scatter([], [])
    t_now = 0.0

    def init():
        draw_world(ax, color_priority_order)
        return scat,

    def update(frame):
        nonlocal t_now
        t_now = frame * DT

        # Coordinator decides STOP/PASS
        coordinator.step(robots, t_now)

        # Update robot motions
        for r in robots:
            r.update(DT, t_now)

        # Plot
        draw_world(ax, color_priority_order)
        xs = [r.x for r in robots]
        ys = [r.y for r in robots]

        plot_colors = []
        for r in robots:
            if r.reached:
                plot_colors.append('black')  # reached
            elif t_now < r.start_time:
                plot_colors.append('gray')   # not started
            elif r.stopped:
                plot_colors.append('orange') # waiting near intersection
            else:
                plot_colors.append(r.color)  # moving with its reservation color

        scat = ax.scatter(xs, ys, c=plot_colors)
        ax.set_title(f"Reservation (color-based, left-hand) - N={len(robots)}, Frame {frame}, t={t_now:.1f}s")
        return scat,

    # More frames for more robots
    frames = 600 if num_robots > 8 else 400

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=50, blit=False, repeat=False
    )

    if save_video:
        print("Saving video to:", os.path.join(os.getcwd(), "mini_sim_reservation.mp4"))
        writer = FFMpegWriter(fps=20, bitrate=1800)
        ani.save("mini_sim_reservation.mp4", writer=writer)
        plt.close(fig)
    else:
        plt.show()

    # Print metrics and write CSV
    summarize_results(robots, coordinator_label="Reservation (color-based, left-hand)",
                      collision_errors=coordinator.collision_errors)
    write_results_csv(robots, filename="reservation_results.csv")


if __name__ == "__main__":
    # You can change num_robots between 2 and 15
    run_mini_sim_reservation(num_robots=15, save_video=True)
