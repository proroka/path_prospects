import matplotlib.pylab as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

from mpl_toolkits.mplot3d import proj3d
def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, a, b],
                     [0, 0, -0.0001, zback]])
proj3d.persp_transformation = orthogonal_proj

dz = 0.01

graph1 = {
    'a1': (0, 0),
    'a2': (0, 1),
    'a3': (0, 2),
    'a4': (0, 3),
    'a5': (0, 4),
    'a6': (0, 5),
    'b1': (1, 0),
    'b2': (1, 1),
    'b3': (1, 2),
    'b4': (1, 3),
    'b5': (1, 4),
    'b6': (1, 5),
    'c3': (2, 2),
    'c4': (2, 3),
    'd3': (3, 2),
    'd4': (3, 3),
    'e3': (4, 2),
    'e4': (4, 3),
    'f3': (5, 2),
    'f4': (5, 3),
    'g1': (6, 0),
    'g2': (6, 1),
    'g3': (6, 2),
    'g4': (6, 3),
    'g5': (6, 4),
    'g6': (6, 5),
    'h1': (7, 0),
    'h2': (7, 1),
    'h3': (7, 2),
    'h4': (7, 3),
    'h5': (7, 4),
    'h6': (7, 5),
}

graph2 = {
    'a1': (.5, .5),
    'a2': (.5, 1.5),
    'a3': (.5, 2.5),
    'a4': (.5, 3.5),
    'a5': (.5, 4.5),
    'b3': (1.5, 2.5),
    'c3': (2.5, 2.5),
    'd3': (3.5, 2.5),
    'e3': (4.5, 2.5),
    'f3': (5.5, 2.5),
    'g1': (6.5, .5),
    'g2': (6.5, 1.5),
    'g3': (6.5, 2.5),
    'g4': (6.5, 3.5),
    'g5': (6.5, 4.5),
}

path1 = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (5, 3),
    (6, 3),
    (6, 4),
    (6, 4),
    (6, 4),
    (6, 4),
    (6, 4),
    (6, 4),
    (6, 4),
    (6, 4),
    (6, 4),
    (6, 4),
]

path2 = [
    (6.5, .5),
    (6.5, .5),
    (6.5, .5),
    (6.5, .5),
    (6.5, .5),
    (6.5, .5),
    (6.5, .5),
    (6.5, .5),
    (6.5, .5),
    (6.5, 1.5),
    (6.5, 2.5),
    (5.5, 2.5),
    (4.5, 2.5),
    (3.5, 2.5),
    (2.5, 2.5),
    (1.5, 2.5),
    (.5, 2.5),
    (.5, 1.5),
    (.5, .5),
]

graphs = [graph1, graph2]
paths = [path1, path2]


def plot_arena(ax, x, y, z):
  # Walls.
  c = 'k'
  ax.plot([-.5 + x, -.5 + x], [-.5 + y, 5.5 + y], [z, z], c=c, lw=1)
  ax.plot([-.5 + x, 7.5 + x], [-.5 + y, -.5 + y], [z, z], c=c, lw=1)
  ax.plot([7.5 + x, -.5 + x], [5.5 + y, 5.5 + y], [z, z], c=c, lw=1)
  ax.plot([7.5 + x, 7.5 + x], [5.5 + y, -.5 + y], [z, z], c=c, lw=1)

  # Obstacles.
  collection = Poly3DCollection([[
      [1.5 + x, -.5 + y, z],
      [1.5 + x, 1.5 + y, z],
      [5.5 + x, 1.5 + y, z],
      [5.5 + x, -.5 + y, z],
  ]])
  collection.set_facecolor(c)
  ax.add_collection3d(collection)

  collection = Poly3DCollection([[
      [1.5 + x, 3.5 + y, z],
      [1.5 + x, 5.5 + y, z],
      [5.5 + x, 5.5 + y, z],
      [5.5 + x, 3.5 + y, z],
  ]])
  collection.set_facecolor(c)
  ax.add_collection3d(collection)

  collection = Poly3DCollection([[
      [-.5 + x, -.5 + y, z - .0001],
      [-.5 + x, 5.5 + y, z - .0001],
      [7.5 + x, 5.5 + y, z - .0001],
      [7.5 + x, -.5 + y, z - .0001],
  ]], alpha=0.)
  collection.set_facecolor('w')
  ax.add_collection3d(collection)


def plot_dot(ax, x, y, z, c='k'):
  d = .1
  ax.add_collection3d(Poly3DCollection([[
      [x - d, y - d, z],
      [x - d, y + d, z],
      [x + d, y + d, z],
      [x + d, y - d, z],
  ]], facecolor=c))


def plot_graph(ax, x, y, z, graph, c='k'):
  for point in graph.values():
    plot_dot(ax, point[0] + x, point[1] + y, z, c)


def plot_robot(ax, robot, z, offset=0, arena=True):
  if arena:
    plot_arena(ax, 9 * offset, 0., z)
  plot_graph(ax, 9 * offset, 0., z, graphs[robot], c=['r', 'b'][robot])


def draw_edges(ax, robot, z1, z2, offset=0):
  graph = graphs[robot]
  c = ['r', 'b'][robot]
  x = 9 * offset
  y = 0
  for node1 in graph.values():
    for node2 in graph.values():
      if node1 == node2:
        continue
      if ((node1[0] in (node2[0] - 1, node2[0] + 1) and node1[1] == node2[1]) or
          (node1[1] in (node2[1] - 1, node2[1] + 1) and node1[0] == node2[0])):
        ax.plot([node1[0] + x, node2[0] + x], [node1[1] + y, node2[1] + y], [z1, z2], c=c, lw=1)


def draw_path(ax, robot, offset=0, volume=False, squeeze=False):
  x = 9 * offset
  y = 0
  dz1 = 0.01 if not squeeze else 0.
  path = paths[robot]
  c = ['r', 'b'][robot]
  for z, ((x1, y1), (x2, y2)) in enumerate(zip(path[:-1], path[1:])):
    ax.plot([x1 + x, x2 + x], [y1 + y, y2 + y], [z * dz1, z * dz1 + dz1], c=c, lw=1)
  if volume:
    a = np.linspace(0., np.pi * 2., 20)
    r = (robot + 1) / 2.
    xs = np.cos(a) * r
    ys = np.sin(a) * r
    for z, (x1, y1) in enumerate(path):
      ax.plot(xs + x + x1, ys + y + y1, [z * dz1] * len(xs), lw=1, c=c, alpha=.5)
    for z, ((x1, y1), (x2, y2)) in enumerate(zip(path[:-1], path[1:])):
      for x3, y3 in zip(xs, ys):
        ax.plot([x1 + x + x3, x2 + x + x3], [y1 + y + y3, y2 + y + y3], [z * dz1, z * dz1 + dz1], c=c, lw=1, alpha=.5)




plot_robot(ax, 0, z=0., offset=0, arena=True)
plot_robot(ax, 1, z=0., offset=0, arena=True)
draw_path(ax, 0, offset=0, squeeze=True)
draw_path(ax, 1, offset=0, squeeze=True)

# plot_robot(ax, 0, z=0., offset=0, arena=True)
# plot_robot(ax, 0, z=dz * 18, offset=0, arena=True)
# plot_robot(ax, 0, z=dz * 9, offset=0, arena=True)
# draw_path(ax, 0, offset=0)

# plot_robot(ax, 1, z=0., offset=1, arena=True)
# plot_robot(ax, 1, z=dz * 18, offset=1, arena=True)
# draw_path(ax, 0, offset=1, volume=True)
# draw_path(ax, 1, offset=1, volume=True)


axis_length = 5
offset = 1.5
ax.plot([-offset, axis_length-offset], [-offset, -offset], [0, 0], 'grey')
ax.plot([-offset, -offset], [-offset, axis_length-offset], [0, 0], 'grey')
ax.plot([-offset, -offset], [-offset, -offset], [0, axis_length * dz], 'grey')
plt.axis('off')
plt.axis('equal')
plt.show()
