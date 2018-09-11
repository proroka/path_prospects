"""Shows a problem.

Usage:
python3 show_problem.py --problem=problems/berlin/problem0
"""

import argparse
import matplotlib.pylab as plt
import numpy as np

import run

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--problem', action='store', default=None, help='Path to problem.')
  args = parser.parse_args()

  robots, environment = run.read_problem(args.problem, verbose=False)
  for r in robots:
    r._draw(environment, r.current, value=2)
    r._draw(environment, r.goal, value=2)

  plt.figure()
  ax = plt.subplot(111)
  ax.matshow(environment.T)
  ax.set_title(args.problem)
  for r in robots:
    ax.text(r.current.x, r.current.y, 'S{}'.format(r.id), verticalalignment='top')
    ax.text(r.goal.x, r.goal.y, 'G{}'.format(r.id), verticalalignment='top')
  plt.show()
