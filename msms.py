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

""" #--------------------------------------------------------------------------------------------
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
 """
#--------------------------------------------------------------------------------------------
#3.Градка  

L = 100                                  
T = 10**5

beta = [-1, 0, 0.2, 0.441, 0.7, 1]

lat = np.random.choice([-1, 1], size=(L, L))
lat_extend = np.zeros((L + 2, L + 2), dtype = int)
#lat_extend[1:(L+1), 1:(L+1)] = lat

def energy(i, j, lat):
    i = i + 1
    j = j + 1
    lat_extend[1:(L+1), 1:(L+1)] = lat
    return np.roll(lat_extend, 1, axis=0)[i, j] + np.roll(lat_extend, -1, axis=0)[i, j] + np.roll(lat_extend, 1, axis=1)[i, j] + np.roll(lat_extend, -1, axis=1)[i, j]

for i in range(T):
    row, col = random.randint(0, L - 1), random.randint(0, L - 1)
    p = 1 / ( 1 + np.exp(2 * beta[0] * energy(row, col, lat)))
    u = uniform(loc=0, scale=1).rvs()
    if u < p:
        lat[row, col] = -1
    else:
        lat[row, col] = 1
plt.imshow(lat, cmap = 'Greys', interpolation='nearest')
plt.show()
 
""" #--------------------------------------------------------------------------------------------
#3.Розподіли 

a = 5
b = 2

uniform_dist = uniform(loc=0, scale=1)

def my_beta(a, b):
    x = [uniform_dist.rvs()]

    for i in range(N):
        U = uniform_dist.rvs()
        V = uniform_dist.rvs()
        if V <= ((U / X[i])**(a - 1) * ((1 - U)/(1 - X[i]))**(b - 1)):
            x.append(U)
        else:
            x.append(x[i])
    return x


beta_samples = my_beta(a, b)

x = np.linspace(0, 1, N)
pdf_values = beta.pdf(x, a, b)

plt.figure(figsize=(8, 6))
plt.hist(beta_samples, bins=50, density=True, alpha=0.7)
plt.plot(x, pdf_values, label=f'Beta({a}, {b}) PDF')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()
plt.show() """