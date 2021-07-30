import numpy as np
from math import sqrt



def my_cubic_interp1d(x0, x, y):

    def diff(lst):
        """
        numpy.diff with default settings
        """
        size = len(lst) - 1
        r = [0] * size
        for i in range(size):
            r[i] = lst[i + 1] - lst[i]
        return r

    def list_searchsorted(listToInsert, insertInto):
        """
        numpy.searchsorted with default settings
        """

        def float_searchsorted(floatToInsert, insertInto):
            for i in range(len(insertInto)):
                if floatToInsert <= insertInto[i]:
                    return i
            return len(insertInto)

        return [float_searchsorted(i, insertInto) for i in listToInsert]

    def clip(lst, min_val, max_val, inPlace=False):
        """
        numpy.clip
        """
        if not inPlace:
            lst = lst[:]
        for i in range(len(lst)):
            if lst[i] < min_val:
                lst[i] = min_val
            elif lst[i] > max_val:
                lst[i] = max_val
        return lst

    def subtract(a, b):
        """
        returns a - b
        """
        return a - b

    size = len(x)

    xdiff = diff(x)
    ydiff = diff(y)

    # allocate buffer matrices
    Li = [0] * size
    Li_1 = [0] * (size - 1)
    z = [0] * (size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = sqrt(2 * xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0  # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size - 1, 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bi = 0.0  # natural boundary
    z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    # solve [L.T][x] = [y]
    i = size - 1
    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    # find index
    index = list_searchsorted(x0, x)
    index = clip(index, 1, size - 1)

    xi1 = [x[num] for num in index]
    xi0 = [x[num - 1] for num in index]
    yi1 = [y[num] for num in index]
    yi0 = [y[num - 1] for num in index]
    zi1 = [z[num] for num in index]
    zi0 = [z[num - 1] for num in index]
    hi1 = list(map(subtract, xi1, xi0))

    # calculate cubic - all element-wise multiplication
    f0 = [0] * len(hi1)
    for j in range(len(f0)):
        f0[j] = zi0[j] / (6 * hi1[j]) * (xi1[j] - x0[j]) ** 3 + \
                zi1[j] / (6 * hi1[j]) * (x0[j] - xi0[j]) ** 3 + \
                (yi1[j] / hi1[j] - zi1[j] * hi1[j] / 6) * (x0[j] - xi0[j]) + \
                (yi0[j] / hi1[j] - zi0[j] * hi1[j] / 6) * (xi1[j] - x0[j])

    return f0


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    points = np.array([(131, 115), (134, 271), (181, 355), (319, 390), (422, 389), (511, 356),
                       (562, 301)])
    x = points[:, 0]
    y = points[:, 1]

    plt.scatter(x, y)
    x_new = np.linspace(min(x), max(x), 100000)
    #print(my_cubic_interp1d(x_new, x, y))
    plt.plot(x_new, my_cubic_interp1d(x_new, x, y))

    plt.show()