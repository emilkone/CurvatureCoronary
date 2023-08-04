import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
# from math import sqrt
import sympy as sym

# circle = [(156, 112), (180, 141), (198, 156), (213, 179), (222, 209), (230, 233), (237, 255), (245, 271), (256, 290), (270, 304), (287, 312), (308, 315), (329, 314), (341, 310), (352, 306), (367, 314), (372, 327), (380, 344), (403, 343), (404, 359), (413, 364), (429, 357)]
CRITICAL_CURVATURE = 1


def CS(circle):
    points = np.array(
        circle)
    x = points[:, 0]
    y = points[:, 1]
    arr = np.arange(np.amin(x), np.amax(x), 0.1)
    s = interpolate.CubicSpline(x, y)

    check_list = []
    cl = []
    smooth_d2 = np.gradient(np.gradient(s(arr)))

    flag = smooth_d2[0]
    for i in range(len(smooth_d2)):
        if np.sign(flag) != np.sign(smooth_d2[i]):
            flag = smooth_d2[i]
            check_list.append(arr[i])
            cl.append(s(arr)[i])
            # print(smooth_d2[i-1], arr[i-1])
            # print(smooth_d2[i], arr[i])

    # Curvature for inflection points
    def curvIP():
        forcurvx = np.array(check_list)
        forcurvy = np.array(cl)

        x_t = np.gradient(forcurvx)
        y_t = np.gradient(forcurvy)

        speed = np.sqrt(x_t * x_t + y_t * y_t)

        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)

        curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / speed ** 1.5

        crit_val_i = []
        crit_val_j = []
        for i, j, k in zip(check_list, cl, curvature_val):
            plt.text(i, j + 0.2, "%.4f" % k, fontsize=10)
            if k > CRITICAL_CURVATURE:
                crit_val_i.append(i)
                crit_val_j.append(j)
        plt.plot(crit_val_i, crit_val_j, 'yo', label='CRITICAL CURVATURE')

    # max, min
    peaks = np.where((s(arr)[1:-1] > s(arr)[0:-2]) * (s(arr)[1:-1] > s(arr)[2:]))[0] + 1
    dips = np.where((s(arr)[1:-1] < s(arr)[0:-2]) * (s(arr)[1:-1] < s(arr)[2:]))[0] + 1

    # print (arr[dips], s(arr)[dips])

    fig, ax = plt.subplots(1, 1)

    # # peaks
    # crit_peaks_arr = [arr[peaks] for i in peaks if s(arr)[i] > CRITICAL_CURVATURE]
    # crit_peaks_s = [s(arr)[i] for i in peaks if s(arr)[i] > CRITICAL_CURVATURE]
    #
    # # dips
    # crit_dips_arr = [arr[dips] for i in dips if s(arr)[i] > CRITICAL_CURVATURE]
    # crit_dips_s = [s(arr)[i] for i in dips if s(arr)[i] > CRITICAL_CURVATURE]
    #
    # # cl
    # crit_dips_check_list = [i for i in check_list if i > CRITICAL_CURVATURE]
    # crit_dips_cl = [cl[i] for i in range(len(check_list)) if check_list[i] > CRITICAL_CURVATURE]

    ax.plot(x, y, 'bo', label='Data Point')
    # ax.plot(b, c, 'ro', label='maxmin')
    ax.plot(arr, s(arr), 'k-', label='Cubic Spline', lw=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(arr[peaks], s(arr)[peaks], 'rx', label='Min point')
    plt.plot(arr[dips], s(arr)[dips], 'rx', label='Max point')
    plt.plot(check_list, cl, 'm*', label='Inflection point')

    # plt.plot(crit_peaks_arr, crit_peaks_s, 'r', label[])

    plt.gca().invert_yaxis()

    # Curvature for extrema points
    def curvEP():
        xm = sym.Symbol('x')

        def polynom():
            poly = np.polyfit(x, y, 5)
            a = round(poly[0], 20)
            b = round(poly[1], 20)
            c = round(poly[2], 20)
            d = round(poly[3], 20)
            e = round(poly[4], 20)
            f = round(poly[5], 20)
            k = a * xm ** 5 + b * xm ** 4 + c * xm ** 3 + d * xm ** 2 + e * xm + f
            return k

        def print_polynom():
            poly = np.polyfit(x, y, 5)
            a = poly[0]
            b = poly[1]
            c = poly[2]
            d = poly[3]
            e = poly[4]
            f = poly[5]
            print(f"{a:+.3e} * x**5 {b:+.3e} * x**4 {c:+.3e} * x**3 {d:+.3e} * x**2 {e:+.3e} * x {f:+.3e}")

        def curv():
            p = polynom()
            k = abs(sym.diff(p, xm, 2)) / sym.sqrt(1 + sym.diff(p, xm, 1) ** 2) ** 3
            # k = k(g)
            return k

        def curvm(k, g):
            a = k
            b = g
            c = []
            d = []

            print_polynom()

            for i in a:
                c.append(curv().subs(xm, i))
            for i in b:
                d.append(curv().subs(xm, i))
            return c, d

        c, d = curvm(arr[peaks], arr[dips])
        for i, j, k in zip(arr[peaks], s(arr)[peaks], c):
            plt.text(i, j + 0.5, "%.4f" % k, fontsize=10)
        for i, j, k in zip(arr[dips], s(arr)[dips], d):
            plt.text(i, j - 1, "%.4f" % k, fontsize=10)

    # for i in range(x.shape[0] - 1):
    #     segment_x = np.linspace(x[i], x[i + 1], 100)
    #     # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
    #     exp_x = (segment_x - x[i])[None, :] ** np.arange(4)[::-1, None]
    #     # Sum over the rows of exp_x weighted by coefficients in the ith column of s.c
    #     segment_y = s.c[:, i].dot(exp_x)
    #     ax.plot(segment_x, segment_y, label='Segment {}'.format(i), ls='--', lw=3)
    # for i, check_list1 in enumerate(check_list, 1):
    #     plt.axvline(x=check_list1, color='k', label=f'Inflection Point {i}')
    curvIP()
    curvEP()
    ax.legend()
    plt.show()

# CS(circle)
