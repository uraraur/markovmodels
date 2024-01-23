import matplotlib.pyplot as plt
from scipy.stats import norm, zipf, uniform, beta, poisson
from scipy.stats import binom as binomo
from math import sqrt, exp
import random
import numpy as np

#--------------------------------------------------------------------------------------------
#1.Моделювання степеневого розподілу

N = 10 ** 6

X = [2]

for i in range(N):
    if X[i] == 1: 
        if random.uniform(0, 1) < (1 / 2) ** (5 / 2):
            X.append(2)
            continue
        X.append(1)
        continue

    u = random.uniform(0, 1)
    if u < 1 / 2: 
        X.append(X[i] - 1)
    else:
        p = (X[i] / (X[i] + 1)) ** (3 / 2)
        v = random.uniform(0, 1)
        if v < p:
            X.append(X[i] + 1)
        else: 
            X.append(X[i])

count = []
for i in range(1, max(X)):
  count.append(X.count(i) / 10 ** 6)

x = np.arange(1, 1000)
#pmf = zipf.pmf(x, 1.5)

#plt.bar(range(1, len(count)+1), count, align='center', label='Block')
#plt.plot(x[:len(count)], pmf[:len(count)], 'r-', lw=2, label='Zipf PMF')
#plt.xlabel('Значення')
#plt.ylabel('Ймовірність')
#plt.legend()
#plt.xlim(0, 40)
#plt.show()

#--------------------------------------------------------------------------------------------
#2.Декодування тексту   
    

alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
X = "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections."
X = X.lower()

text_n = ""
for l in X:
        if l in alp:
            text_n += l
        else:
           text_n += ' '
text_n =  " ".join(text_n.split())

count_bigr = {}
p = text_n[0]
for i in alp:
    for j in alp:
        count_bigr[i + j] = 0
for l in text_n[1:]:
    count_bigr[p + l] += 1
    p = l

print(count_bigr)

#--------------------------------------------------------------------------------------------
#3.Градка  

L = 100
n_iterations = 10**5

beta = 0.75

grid = np.random.choice([-1, 1], size=(L, L))

def energy_at_point(spin, neighbors):
    return -spin * np.sum(neighbors)

for i in range(n_iterations):
    row, col = random.randint(0, L - 1), random.randint(0, L - 1)

    neighbors = []
    if 0 <= row - 1 < L:
        neighbors.append(grid[(row - 1) % L, col])
    if 0 <= row + 1 < L:
        neighbors.append(grid[(row + 1) % L, col])
    if 0 <= col - 1 < L:
        neighbors.append(grid[row, (col - 1) % L])
    if 0 <= col + 1 < L:
        neighbors.append(grid[row, (col + 1) % L])

    energy_before = energy_at_point(grid[row, col], neighbors)
    grid[row, col] *= -1
    energy_after = energy_at_point(grid[row, col], neighbors)
    p = 1/(1+np.exp(-2*beta * energy_after))
    # print(i, p)
    U = uniform(loc=0, scale=1).rvs()
    if U < p:
        grid[row, col] *= -1

b = -1.5
plt.imshow(grid, cmap='Greens', interpolation='nearest')
plt.show()

