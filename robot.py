import numpy as np
import matplotlib.pylab as plt
import random

import util


class Robot(object):
  def __init__(self, identifier, size):
    self.id = identifier
    self.size = size
    self.current = util.Cell(0, 0)
    self.goal = util.Cell(0, 0)
    self.distance_to_goal = None
    self.distance_to_goal_naive = None
    self.priority = None
    self.time = 0
    self.path_taken = []
    self.arrived = False
    self.group = None
    self.neighbors = set()
    # Random number of random tiebreak that is
    # consistent over a run.
    self.random = random.random()
    self.reset_plan()

  def __repr__(self):
    return 'Robot[{}, {}]'.format(self.id, self.size)

  def __hash__(self):
    return hash(self.id)

  def __lt__(self, other):
    return self.id < other.id

  def __le__(self, other):
    return self.id <= other.id

  def __eq__(self, other):
    return self.id == other.id

  def distance(self, environment):
    if self.distance_to_goal is None:
      self.build_distances(environment)
    return self.distance_to_goal[self.current.x, self.current.y]

  @property
  def has_planned(self):
    return self.path is not None

  @property
  def center(self):
    return np.array([(self.current.x + self.size) / 2., (self.current.y + self.size) / 2.])

  def add_neighbor(self, r):
    previous_num_neighbors = len(self.neighbors)
    self.neighbors.add(r)
    return previous_num_neighbors != len(self.neighbors)

  def remove_neighbor(self, r):
    previous_num_neighbors = len(self.neighbors)
    self.neighbors.discard(r)
    return previous_num_neighbors != len(self.neighbors)

  def move(self):
    if self.path is None:
      raise ValueError('No plans available.')
    if not self.path_taken:
      self.path_taken.append(self.current)
    if self.current == self.goal:
      self.arrived = True
      return True
    self.time += 1
    self.time_on_path += 1
    self.current = self.path[self.time_on_path]
    self.path_taken.append(self.current)
    self.arrived = self.current == self.goal
    return self.arrived

  def reset_plan(self):
    self.path_creation_time = -1
    self.path = None
    self.time_on_path = 0
    self.dependencies = []
    self.path_cost = None

  def reset_priority(self):
    self.priority = None

  def _distance_to(self, environment, destination, size=None):
    if size is None:
      size = self.size
    return util.distance_to(environment, destination, size)

  def build_distances(self, environment):
    self.distance_to_goal = self._distance_to(environment, self.goal)
    self.distance_to_goal_naive = self._distance_to(environment, self.goal, size=1)

  @staticmethod
  def _count_obstacles(grid):
    # Flood map to count obstacles.
    for x in range(grid.shape[0]):
      util.flood(grid, util.Cell(x, 0), 1)
      util.flood(grid, util.Cell(x, grid.shape[1] - 1), 1)
    for y in range(grid.shape[1]):
      util.flood(grid, util.Cell(0, y), 1)
      util.flood(grid, util.Cell(grid.shape[0] - 1, y), 1)
    count = 0
    for x in range(grid.shape[0]):
      for y in range(grid.shape[1]):
        success = util.flood(grid, util.Cell(x, y), 1)
        count += int(success)
    return count

  def _count_heuristic(self, fn, environment, verbose=False, distance_to_goal=None, size=None, **kwargs):
    if self.distance_to_goal is None or self.distance_to_goal_naive is None:
      self.build_distances(environment)
    if size is None:
      size = self.size
    if distance_to_goal is None:
      distance_to_goal = self.distance_to_goal
    prospect_map = fn(environment, self.current, distance_to_goal, size, **kwargs)
    if verbose:
      grid = prospect_map.copy()
    count = self._count_obstacles(prospect_map)
    if verbose:
      self._draw(grid, self.current, value=2)
      self._draw(grid, self.goal, value=2)
      plt.figure()
      ax = plt.subplot(111)
      ax.matshow(grid)
      ax.set_title('Robot {}: {} obstacles'.format(self.id, count))
      plt.show()
    return count

  def prospects(self, environment, verbose=False, **kwargs):
    return self._count_heuristic(util.forward, environment, verbose, **kwargs)

  def naive_prospects(self, environment, verbose=False, **kwargs):
    return self._count_heuristic(util.forward, environment, verbose,
                                 distance_to_goal=self.distance_to_goal_naive,
                                 size=1, **kwargs)

  def obstacles(self, environment, verbose=False, **kwargs):
    return self._count_heuristic(util.around, environment, verbose, **kwargs)

  def naive_obstacles(self, environment, verbose=False, **kwargs):
    return self._count_heuristic(util.around, environment, verbose,
                                 distance_to_goal=self.distance_to_goal_naive,
                                 size=1, **kwargs)

  def _draw(self, environment, position, value=1):
    if len(environment.shape) == 3:
      environment[:, position.x:position.x + self.size, position.y:position.y + self.size] = value
      return
    environment[position.x:position.x + self.size, position.y:position.y + self.size] = value

  def draw(self, environment, value=1):
    self._draw(environment, self.current, value)

  def draw_path(self, environment):
    assert self.path is not None
    for t, p in enumerate(self.path[self.time_on_path:]):
      self._draw(environment[t], p)
    # For simplicity allow robot to disappear at goal. Otherwise uncomment.
    # self._draw(environment[len(self.path[self.time_on_path:]):], self.path[-1])

  def plan(self, time_environment, verbose=False, show_plots=False):
    if self.arrived:
      return 0.
    # Check if we have new highest priority neighbors with newer plans.
    current_dependencies = sorted([(r.id, r.path_creation_time) for r in self.neighbors if r.priority < self.priority])
    if self.path is not None:
      if current_dependencies == self.dependencies:
        return self.path_cost - self.time_on_path
    if verbose:
      print('Robot {} is re-planning with dependencies {} at time {}'.format(self.id, current_dependencies, self.time))
    # If there are any changes, update plan.
    self.reset_plan()
    grid = time_environment.copy()
    soft_grid = np.zeros_like(grid[0])
    # Create obstacle map.
    for r in self.neighbors:
      if r.arrived:
        continue
      assert r.priority != self.priority
      if r.priority > self.priority:
        # Penalize other robots position if possible.
        r.draw(soft_grid)
        continue
      r.draw_path(grid)

    # Plan.
    cost, path = util.plan(grid, self.current, self.goal, self.distance_to_goal, self.size, soft_grid)

    if show_plots:
      self._draw(grid, self.current, value=2)
      self._draw(grid, self.goal, value=2)
      # ts = [0, 10, 20]
      ts = [0, 50, 100]
      ncols = 3
      nrows = (len(ts) - 1) // ncols + 1
      plt.figure(figsize=(6 * ncols, 6 * nrows))
      for i, t in enumerate(ts):
        if t < len(grid):
          ax = plt.subplot(nrows, ncols, i + 1)
          ax.matshow(grid[t])
          for r in self.neighbors:
            if r.arrived or r.priority > self.priority:
              continue
            xs, ys = zip(*[(p.x, p.y) for p in r.path])
            ax.plot(ys, xs, lw=1)
          if path:
            xs, ys = zip(*[(p.x, p.y) for p in path])
            ax.plot(ys, xs, lw=2)
          ax.set_title(str(t))
        plt.suptitle('{}'.format(current_dependencies))
      plt.show()

    if cost is None:
      return None
    self.path = path
    self.path_creation_time = self.time
    self.dependencies = current_dependencies
    self.path_cost = cost
    return cost
