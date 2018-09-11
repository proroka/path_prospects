import collections
from queue import PriorityQueue
import numba as nb
import numpy as np


class Cell(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def __lt__(self, other):
    return 0

  def __hash__(self):
    return hash((self.x, self.y))

  def __repr__(self):
    return 'Cell[{},{}]'.format(self.x, self.y)

  def hit(self, environment, size):
    return np.any(environment[self.x:self.x + size, self.y:self.y + size] > 0)

  def allowed(self, environment, size):
    return self.quick_allowed(environment, size) and not self.hit(environment, size)

  def quick_allowed(self, environment, size=1):
    return (self.x + size <= environment.shape[0] and self.x >= 0 and
            self.y + size <= environment.shape[0] and self.y >= 0)

  def neighbors(self, environment, size):
    n = []
    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)):
      v = Cell(self.x + dx, self.y + dy)
      if v.allowed(environment, size):
        n.append(v)
    return n

  def quick_neighbors(self, environment, size=1):
    n = []
    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
      v = Cell(self.x + dx, self.y + dy)
      if v.quick_allowed(environment, size):
        n.append(v)
    return n


@nb.jit(nopython=False)
def distance_to(grid, destination, size):
  distances = np.ones_like(grid, dtype=np.float32) * float(np.inf)
  visited = set()
  q = PriorityQueue()
  q.put((0, destination))
  while not q.empty():
    d, u = q.get()
    if u in visited:
      continue
    visited.add(u)
    distances[u.x, u.y] = d
    for v in u.neighbors(grid, size):
      q.put((d + 1, v))
  return distances


@nb.jit(nopython=False)
def flood(grid, start, value):
  if grid[start.x, start.y] > 0:
    return False
  s = [start]
  while s:
    u = s.pop()
    if grid[u.x, u.y] > 0:
      continue
    grid[u.x, u.y] = value
    for v in u.quick_neighbors(grid):
      s.append(v)
  return True


@nb.jit(nopython=False)
def forward(grid, start, distance_to_goal, size, slack=None, limit=None):
  visited = set()
  s = collections.deque([(start, 0.)])
  prospect_map = np.zeros_like(grid)
  if slack is not None:
    d = distance_to_goal[start.x, start.y] * slack
  elif limit is not None:
    d = limit
  else:
    raise ValueError('Must specify either slack or limit.')
  while s:
    u, du = s.popleft()
    if u in visited:
      continue
    visited.add(u)
    prospect_map[u.x, u.y] = 1
    for v in u.quick_neighbors(grid, size):
      dv = du + 1.
      total_distance = distance_to_goal[v.x, v.y] + dv
      if total_distance <= d:
        s.append((v, dv))
  return prospect_map


@nb.jit(nopython=False)
def around(grid, start, distance_to_goal, size, limit):
  visited = set()
  s = collections.deque([(start, 0.)])
  prospect_map = np.zeros_like(grid)
  while s:
    u, du = s.popleft()
    if u in visited:
      continue
    visited.add(u)
    prospect_map[u.x, u.y] = 1
    for v in u.quick_neighbors(grid, size):
      dv = du + 1.
      if dv <= limit and distance_to_goal[v.x, v.y] < np.inf:
        s.append((v, dv))
  return prospect_map


def plan(time_grid, start, goal, distances, size, soft_obstacles):
  def g(v):
    return v[2]

  # Make the heuristic slightly non-preferable (makes things a lot faster).
  def h(v):
    return distances[v[0].x, v[0].y] * 1.00001

  def f(v):
    return g(v) + h(v)

  def key(v):
    return (v[0], v[1])

  def at_goal(v):
    return v[0] == goal

  parents = {}
  def build_path(u):
    path = [u[0]]
    u = parents[u]
    while u is not None:
      path.append(u[0])
      u = parents[u]
    return list(reversed(path))

  visited = set()
  q = PriorityQueue()
  start_node = (start, 0, 0., None)
  q.put((h(start_node), start_node))
  while not q.empty():
    total_cost, u = q.get()
    if total_cost > time_grid.shape[0]:
      break
    if key(u) in visited:
      continue
    node_u, t, _, parent = u
    parents[key(u)] = parent
    if at_goal(u):
      return g(u), build_path(key(u))
    visited.add(key(u))
    if len(visited) > time_grid.shape[1] * time_grid.shape[2] * 5:  # Do not go crazy.
      break
    if t + 1 >= time_grid.shape[0]:
      continue
    for node_v in node_u.neighbors(time_grid[t + 1], size):
      # Check if we are hitting a soft_obstacle.
      added_cost = 0.
      if node_v.hit(soft_obstacles, size):
        added_cost = 1e-4  # Something really small that can't add up to more than one.
      v = (node_v, t + 1, g(u) + 1. + added_cost, key(u))
      q.put((f(v), v))
  return None, None
