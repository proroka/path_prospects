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

  grid = (1 - environment) * 255
  def robot_color(i, min_value=50, max_value=200):
    return int((max_value - min_value) * float(i) / len(robots) + min_value)
  for r in robots:
    r._draw(grid, r.current, value=robot_color(r.id))
    r._draw(grid, r.goal, value=robot_color(r.id))

  plt.figure()
  ax = plt.subplot(111)
  ax.matshow(grid.T, cmap='gray')
  ax.set_title(args.problem)
  for r in robots:
    ax.text(r.current.x, r.current.y, 'S{}'.format(r.id), verticalalignment='top')
    ax.text(r.goal.x, r.goal.y, 'G{}'.format(r.id), verticalalignment='top')
  plt.tick_params(axis='both', which='both', bottom=False, top=False,
                  labelbottom=False, labeltop=False, left=False, right=False,
                  labelleft=False, labelright=False)
  plt.show()
