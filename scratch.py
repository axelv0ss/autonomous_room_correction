from matplotlib import pyplot as plt
import numpy as np

# fu = 20000
# fl = 40
#
# x = np.arange(0, 1, 0.01)
# y = fu * (fu / fl) ** x
# f_oct = x * np.log2(fu/fl)
# y = fu * 2 ** f_oct
#
# plt.semilogy(x, y, basey=2)
# plt.show()


# def b_oct(Q):
#     num = np.sqrt(1 + 4 * Q ** 2) + 1
#     den = np.sqrt(1 + 4 * Q ** 2) - 1
#     return np.log2(num/den)
#
#
# Q = np.arange(0.5, 0.7, 0.01)
#
#
# plt.plot(Q, 1/b_oct(Q))
# plt.xlabel("Q")
# plt.ylabel("1/b_oct")
#
# plt.show()

x = np.random.normal(0, 0, 1000000)


def f_new(x, f_old=1):
    if x >= 0:
        return f_old * (1 + x)
    elif x < 0:
        return f_old / (1 - x)


# plt.hist([f_new(x) for x in x], bins=np.logspace(np.log10(0.1), np.log10(10.0), 100))
plt.hist(2 ** x, bins=np.logspace(np.log10(0.1), np.log10(10.0), 100))
plt.gca().set_xscale("log")
plt.show()
