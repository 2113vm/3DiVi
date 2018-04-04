# def get_RT_lstsq(pxs, end_pxs):
#     X = np.array([[pxs[0,1], - pxs[0,0], 1, 0],
#                   [pxs[0,0], pxs[0,1], 0, 1],
#                   [pxs[1,1], - pxs[1,0], 1, 0],
#                   [pxs[1,0], pxs[1,1], 0, 1],
#                   [pxs[2,1], - pxs[2,0], 1, 0],
#                   [pxs[2,0], pxs[2,1], 0, 1],
#                   [pxs[3,1], - pxs[3,0], 1, 0],
#                   [pxs[3,0], pxs[3,1], 0, 1],
#                   [pxs[4,1], - pxs[4,0], 1, 0],
#                   [pxs[4,0], pxs[4,1], 0, 1],
#                   [pxs[5,1], - pxs[5,0], 1, 0],
#                   [pxs[5,0], pxs[5,1], 0, 1],
#                   [pxs[6,1], - pxs[6,0], 1, 0],
#                   [pxs[6,0], pxs[6,1], 0, 1]])
#
#     Y = np.array([end_pxs[0,1],
#                   end_pxs[0,0],
#                   end_pxs[1,1],
#                   end_pxs[1,0],
#                   end_pxs[2,1],
#                   end_pxs[2,0],
#                   end_pxs[3,1],
#                   end_pxs[3,0],
#                   end_pxs[4,1],
#                   end_pxs[4,0],
#                   end_pxs[5,1],
#                   end_pxs[5,0],
#                   end_pxs[6,1],
#                   end_pxs[6,0]])
#
#     w = np.linalg.lstsq(X, Y)
#     cos_a, sin_a, t1, t2 = w[0]
#
#     return np.array([[cos_a, -sin_a],[sin_a, cos_a]]), np.array([[t1],[t2]])

#
# for i in map(lambda x: sum(x**2), pxs):
#     print(i)
#
# for px in pxs:
#     print(np.argmin(np.array([x for x in map(lambda x: sum(x**2), (px - end_pxs))])))
#
#
# np.sum((pxs[3] - end_pxs)**2, axis=1)
