"""Runs a battery of experiments.

Usage:
python3 run.py \
  --problem=problems/** \
  --communication_radius=30,50,70 \
  --mode="['dynamic']" \
  --scheme="[('longest_priority', 'tiebreak_random'), \
             ('forward_looking_priority', 'tiebreak_longest_first'), \
             ('forward_looking_priority', 'tiebreak_random'), \
             ('naive_forward_looking_priority', 'tiebreak_longest_first'), \
             ('surroundings_50_priority', 'tiebreak_longest_first'), \
             ('naive_surroundings_50_priority', 'tiebreak_longest_first'), \
             ('random_priority', 'tiebreak_random')] \
  --output_results=/tmp/results.bin
"""

import argparse
import ast
import collections
from concurrent import futures
import glob
import itertools
import msgpack
import msgpack_numpy
import numpy as np
import os
import random
import subprocess
import sys
import tempfile
import threading
import toposort
import tqdm

import robot

_MAX_TIME_FACTOR = 2
_NUM_THREADS = 24


# Dimensions that can be varied and their defaults.
Arguments = collections.namedtuple('Arguments', [
    'problem', 'communication_radius', 'mode', 'scheme'])
Arguments.__new__.__defaults__ = (
    'problems/basic/multi_corridor',
    '30',  # Can be comma-separated.
    'dynamic',
    '("forward_looking_priority", "tiebreak_longest_first")')

Statistics = collections.namedtuple('Statistics', [
    'success', 'ideal_path_lengths', 'path_lengths'])


def store_results(results, filename):
  with open(filename, 'wb') as fp:
    buf = msgpack.packb(results, use_bin_type=True)
    fp.write(buf)


def read_results(filename, delete=False):
  with open(filename, 'rb') as fp:
    r = msgpack.unpackb(fp.read(), raw=False, use_list=False)
  if delete:
    os.remove(filename)
  return r


def get_setup_fns(priority_scheme, tiebreak_scheme, static=False):
  # Tiebreak functions (only depend on robot).
  def tiebreak_shortest_first(robot, environment):
    return robot.distance(environment)
  def tiebreak_longest_first(robot, environment):
    return -robot.distance(environment)
  def tiebreak_lowerid_first(robot, environment):
    return robot.id
  def tiebreak_random(robot, environment):
    return robot.random

  # Priority functions.
  def longest_priority(robot, environment, limit=None):
    return -robot.distance(environment)
  def forward_looking_priority(robot, environment, limit=None):
    return robot.prospects(environment, limit=limit)
  def naive_forward_looking_priority(robot, environment, limit=None):
    return robot.naive_prospects(environment, limit=limit)
  def surroundings_20_priority(robot, environment, limit=None):
    return robot.obstacles(environment, limit=20)
  def naive_surroundings_20_priority(robot, environment, limit=None):
    return robot.naive_obstacles(environment, limit=20)
  def surroundings_50_priority(robot, environment, limit=None):
    return robot.obstacles(environment, limit=50)
  def naive_surroundings_50_priority(robot, environment, limit=None):
    return robot.naive_obstacles(environment, limit=50)
  def surroundings_80_priority(robot, environment, limit=None):
    return robot.obstacles(environment, limit=80)
  def naive_surroundings_80_priority(robot, environment, limit=None):
    return robot.naive_obstacles(environment, limit=80)
  def constant_priority(robot, environment, limit=None):
    return 0
  def random_priority(robot, environment, limit=None):
    return random.random()

  # For static setup, we cache the results.
  def wrap_static(fn):
    cache = {}
    def _fn(r, *args, **kwargs):
      if r.id in cache:
        return cache[r.id]
      ret = fn(r, *args, **kwargs)
      cache[r.id] = ret
      return ret
    return _fn

  priority_fn = locals()[priority_scheme]
  tiebreak_fn = locals()[tiebreak_scheme]
  if static:
    return wrap_static(priority_fn), wrap_static(tiebreak_fn)
  return priority_fn, tiebreak_fn


def print_stats(stats):
  if stats.success:
    v = np.max(stats.path_lengths)
    b = np.max(stats.ideal_path_lengths)
    print('Makespan: {:.2f} (baseline is {:.2f}) => {:.2f}%'.format(v, b, (v - b) * 100. / b))
    v = np.mean(stats.path_lengths)
    b = np.mean(stats.ideal_path_lengths)
    print('Flowtime: {:.2f} (baseline is {:.2f}) => {:.2f}%'.format(v, b, (v - b) * 100. / b))
    v = np.mean((stats.path_lengths - stats.ideal_path_lengths) / stats.ideal_path_lengths)
    print('Prolongation: {:.2f}%'.format(v * 100.))
  else:
    print('Failure')


def simulate(robots, environment, radius, ordering, tiebreak=None, verbose=False):
  # Record time taken.
  time_taken = [None] * len(robots)
  minimal_time = [r.distance(environment) for r in robots]
  success = True

  # Limit simulation time.
  longest_path_length = int(max(minimal_time))
  for t in range(longest_path_length * _MAX_TIME_FACTOR):
    for r in robots:
      if not r.arrived:
        break
    else:
      break  # All robots arrived.

    # Build neighbors list.
    # The priorities need to be recomputed if neighbors changed.
    priority_changed = False
    for r1 in robots:
      for r2 in robots:
        if r1 <= r2:
          continue
        d = np.linalg.norm(r1.center - r2.center)
        if d <= radius:
          if r1.add_neighbor(r2):
            r1.reset_priority()
            priority_changed = True
          if r2.add_neighbor(r1):
            r2.reset_priority()
            priority_changed = True
        else:
          if r1.remove_neighbor(r2):
            r1.reset_priority()
            priority_changed = True
          if r2.remove_neighbor(r1):
            r2.reset_priority()
            priority_changed = True
    if priority_changed:
      for r in robots:
        r.reset_priority()

    # Compute priorities if needed.
    # We cheat slightly here by assuming that there was a
    # multiple hop communication happening for getting the
    # longest path length (allows to have consistent priority values).
    current_longest_path_length = max(r.distance(environment) for r in robots)
    for r in robots:
      if r.priority is None:
        if r.arrived:
          r.priority = (float('inf'), float('inf'), float('inf'))
          continue
        # Important that this is computed once for efficiency.
        r.priority = (ordering(r, environment, limit=current_longest_path_length),
                      tiebreak(r, environment), r.random)
        if verbose:
          print('Re-computed priority for robot {} at time {}: {}'.format(r.id, t, r.priority))

    # Find robot ordering (as per communication radius).
    dependencies = collections.defaultdict(set)
    for r1 in robots:
      for r2 in r1.neighbors:
        if r2.priority < r1.priority:
          dependencies[r1].add(r2)
    ordered_robots = toposort.toposort_flatten(dependencies)
    # Robots plan in order now.
    time_obstacles = np.tile(environment, [int(current_longest_path_length) * _MAX_TIME_FACTOR, 1, 1])
    for r in ordered_robots:
      if r.plan(time_obstacles) is None:
        if verbose:
          print('Impossibility when planning for robot {} at time {}'.format(r.id, t))
        success = False
        break
    if not success:
      break

    # Move robots.
    for r in robots:
      if r.arrived:
        continue
      if r.move():
        if verbose:
          print('Robot {} arrived at destination at time {}.'.format(r.id, t))
        time_taken[r.id] = float(r.time)

    # Verify that no collision occured.
    for r1 in robots:
      if r1.arrived:
        continue
      for r2 in robots:
        if r2.arrived:
          continue
        if r2 <= r1:
          continue
        if not (r1.current.x >= r2.current.x + r2.size or r1.current.x + r1.size <= r2.current.x or
                r1.current.y >= r2.current.y + r2.size or r1.current.y + r1.size <= r2.current.y):
          if verbose:
            print('Collision detected at time {} between {} and {} :/', t, r1, r2)
          success = False
          break
      if not success:
        break
    if not success:
      break
  else:
    # Some robots did not arrive at destination.
    success = False

  time_taken = np.array(time_taken, dtype=np.float32)
  minimal_time = np.array(minimal_time, dtype=np.float32)
  stats = Statistics(success, ideal_path_lengths=minimal_time, path_lengths=time_taken)
  if verbose:
    print_stats(stats)
  return stats


def read_problem(filename, verbose=False):
  # Read problem.
  with open(filename, 'r') as fp:
    lines = fp.readlines()

  height = int(lines[0])
  width = height
  environment = np.zeros((width, height), dtype=np.uint8)
  if verbose:
    print('Opening problem of size: {}x{}'.format(width, height))

  # Obstacles.
  i = 1
  while i < len(lines):
    line = lines[i].strip()
    i += 1
    if not line:
      continue
    if line.startswith('-'):
      break
    p = eval(line)
    environment[p] = 1
  # Remove surrounding.
  idx = np.where(environment == 0)
  min_i = np.min(idx[0])
  min_j = np.min(idx[1])
  max_i = np.max(idx[0]) + 1
  max_j = np.max(idx[1]) + 1
  environment = environment[min_i:max_i, min_j:max_j]

  # Robots.
  robots = []
  while i < len(lines):
    line = lines[i]
    i += 1
    if line.startswith('-'):
      break
    r = eval(line)
    robots.append(robot.Robot(r[0], r[1]))
  while i < len(lines):
    line = lines[i]
    i += 1
    idx, start, end = eval(line)
    robots[idx].current.x = start[0]
    robots[idx].current.y = start[1]
    robots[idx].goal.x = end[0]
    robots[idx].goal.y = end[1]
  robots = sorted(robots, key=lambda r: r.id)
  return robots, environment


def run_problem(outfile, args):
  verbose = outfile is None
  ordering, tiebreak = get_setup_fns(*args.scheme)
  robots, environment = read_problem(args.problem, verbose=verbose)
  stats = simulate(robots, environment, args.communication_radius, ordering, tiebreak, verbose=verbose)
  if outfile:
    store_results(stats, outfile)


def run_task(filename, arguments):
  args = [sys.executable, __file__, '--internal_output', filename]
  for field in Arguments._fields:
    args.append('--{}'.format(field))
    args.append(str(getattr(arguments, field)))
  return subprocess.call(args)


def done(fn, counter):
  counter.inc()


class AtomicProgressBar(object):
  def __init__(self, total):
    self._value = 0
    self._lock = threading.Lock()
    self._tqdm = tqdm.tqdm(total=total)

  def inc(self):
    with self._lock:
      self._value += 1
      self._tqdm.update(1)

  def close(self):
    self._tqdm.close()


def run(final_filename, args):
  directory = tempfile.mkdtemp()
  all_args = list(set(args))

  threads = []
  executor = futures.ProcessPoolExecutor(max_workers=_NUM_THREADS)
  counter = AtomicProgressBar(len(all_args))
  for i, a in enumerate(all_args):
    filename = os.path.join(directory, 'results_{}.bin'.format(i))
    threads.append((executor.submit(run_task, filename, a), filename, i))
    threads[-1][0].add_done_callback(lambda fn: done(fn, counter))

  all_results = collections.defaultdict(list)
  for thread, filename, idx in threads:
    if thread.result() != 0:
      print('A problem failed. Ignoring it... {}'.format(all_args[idx]))
      continue
    thread_results = read_results(filename, delete=True)
    all_results[all_args[idx]].append(thread_results)

  all_results = dict(all_results)  # Remove defaultdict.
  store_results(all_results, final_filename)


if __name__ == '__main__':
  msgpack_numpy.patch()  # Magic.

  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--output_results', action='store', default=None, help='Where to store results.')
  defaults = Arguments()
  for field in Arguments._fields:
    v = getattr(defaults, field)
    parser.add_argument('--{}'.format(field), type=type(v), action='store', default=v)
  # Internal arguments.
  parser.add_argument('--internal_output', action='store', default=None, help='Intermediate results')
  args = parser.parse_args()

  # Get problem list.
  problems = []
  for filename in glob.iglob(args.problem, recursive=True):
    if os.path.isfile(filename):
      problems.append(filename)
  assert problems, 'No problem has been specified.'
  communication_radii = ast.literal_eval(args.communication_radius)
  if not isinstance(communication_radii, collections.Iterable):
    communication_radii = [communication_radii]
  for r in communication_radii:
    assert isinstance(r, (float, int)), 'Communication radius must be a float'
  communication_radii = [float(r) for r in communication_radii]
  try:
    modes = ast.literal_eval(args.mode)
  except ValueError:
    modes = args.mode
  if isinstance(modes, str):
    modes = [modes]
  for m in modes:
    assert m == 'dynamic' or m == 'static', 'Mode can only be "static" or "dynamic"'
  schemes = ast.literal_eval(args.scheme)
  assert isinstance(schemes, collections.Iterable), 'Scheme must be a tuple of list of tuples.'
  if isinstance(schemes[0], str):
    schemes = [schemes]
  all_problems = [Arguments(*v) for v in itertools.product(problems, communication_radii, modes, schemes)]
  assert all_problems

  if len(all_problems) == 1:
    run_problem(args.internal_output, all_problems[0])
  else:
    assert args.output_results, 'Must specify --output_results'
    run(args.output_results, all_problems)
