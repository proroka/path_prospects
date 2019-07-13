"""Create problems from image."""

import argparse
import numpy as np
import os
from PIL import Image
import random

import util

_MAX_TRIES = 50


def load_image(filename):
  img = Image.open(filename)
  img.load()
  return np.asarray(img, dtype=np.uint8)


def write_prefix(fp, size, obstacles):
  fp.write('{}\n'.format(size))
  for p in obstacles:
    fp.write('({},{})\n'.format(p[0], p[1]))


def create_problem(environment, start_positions, goal_positions):
  robots = []
  current_environment = environment.copy()
  random.shuffle(robot_sizes)
  for i, size in enumerate(robot_sizes):
    np.random.shuffle(start_positions)
    np.random.shuffle(goal_positions)
    # Create set of allowed cells.
    start_cells = []
    for start in start_positions:
      c = util.Cell(start[0], start[1])
      if c.allowed(current_environment, size):
        start_cells.append(c)
    goal_cells = []
    for goal in goal_positions:
      c = util.Cell(goal[0], goal[1])
      if c.allowed(environment, size):
        goal_cells.append(c)
      if len(goal_cells) >= _MAX_TRIES:
        break

    # Pick random starts.
    found = False
    for g in goal_cells:
      dist = util.distance_to(environment, g, size)
      for s in start_cells:
        d = dist[s.x, s.y]
        if not np.isinf(d):
          robots.append((size, (s.x, s.y), (g.x, g.y)))
          current_environment[s.x:s.x + size, s.y:s.y + size] = 1
          found = True
          break
      if found:
        break
    if not found:
      raise ValueError('Unable to find valid start and goal positions.')
    print('Robot #{} done.'.format(i))
  return robots


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--image', action='store', default=None, help='Path to image.')
  parser.add_argument('--output', action='store', default=None, help='Path to output problems.')
  parser.add_argument('--n', type=int, action='store', default=1, help='Number of problems.')
  parser.add_argument('--offset', type=int, action='store', default=0, help='Number of problems.')
  parser.add_argument('--num_robots', type=int, action='store', default=100, help='Number of robots.')
  args = parser.parse_args()

  max_size = 4
  n = args.num_robots // max_size
  n1 = args.num_robots - n * (max_size - 1)
  robot_sizes = [1] * n1
  for i in range(2, max_size + 1):
    robot_sizes += [i] * n

  image = load_image(args.image)
  assert len(image.shape) == 3, 'Wrong format.'
  if image.shape[-1] == 4:
    image = image[:, :, :3]
  assert image.shape[0] != 3, 'Wrong format. Needs RGB.'

  name = os.path.splitext(os.path.basename(args.image))[0]
  env_size = max(image.shape[0], image.shape[1])
  obstacles = np.where(np.logical_and(image[:, :, 0] < 20, np.logical_and(image[:, :, 1] < 20, image[:, :, 2] < 20)))  # Black.
  start_positions = np.where(np.logical_and(image[:, :, 0] < 150, np.logical_and(image[:, :, 1] > 150, image[:, :, 2] < 50)))  # Green.
  goal_positions = np.where(np.logical_and(image[:, :, 0] > 150, np.logical_and(image[:, :, 1] < 150, image[:, :, 2] < 150)))  # Red.
  obstacles = np.array(obstacles).T
  start_positions = np.array(start_positions).T
  goal_positions = np.array(goal_positions).T
  print(obstacles.shape)

  # Environment.
  height = int(env_size)
  width = height
  environment = np.zeros((width, height), dtype=np.uint8)
  print('Creating environment of size: {}x{}'.format(width, height))
  # Obstacles.
  for p in obstacles:
    environment[p[0], p[1]] = 1
  # Remove surrounding.
  idx = np.where(environment == 0)
  min_i = np.min(idx[0])
  min_j = np.min(idx[1])
  max_i = np.max(idx[0]) + 1
  max_j = np.max(idx[1]) + 1
  environment = environment[min_i:max_i, min_j:max_j]

  # Create all problems.
  for j in range(args.n):
    robots = create_problem(environment, start_positions, goal_positions)
    with open(os.path.join(args.output, 'problem{}'.format(j + args.offset)), 'w') as fp:
      write_prefix(fp, env_size, obstacles)
      fp.write('-\n')
      for i, r in enumerate(robots):
        fp.write('({}, {}, 1.0, -1, 0.5)\n'.format(i, r[0]))
      fp.write('-\n')
      for i, r in enumerate(robots):
        fp.write('({}, {}, {})\n'.format(i, r[1], r[2]))
