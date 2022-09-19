from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, fmin


def med_err(loc, scale, median_obs):
    dist = stats.norm(loc, scale).rvs(10000)
    dist = dist**2
    dist = dist[dist > 0.0005]
    median_est = np.median(dist)
    err = (median_est - median_obs)**2

    return err


def std_err(scale, loc, d16, d84):
    if scale <= 0:
        scale = abs(scale)
    dist = stats.norm(loc, scale).rvs(10000)
    dist = dist**2
    dist = dist[dist > 0.0005]
    err = (np.percentile(dist, 84)-d84)**2 + (np.percentile(dist, 16)-d16)**2

    return err


def shape_err(a, scale, loc, dif):
    dist = stats.skewnorm(a, loc, scale).rvs(10000)
    dist = dist[dist > 0.0005]
    err = (np.percentile(dist, 84) - np.percentile(dist, 16))**2

    return err


a, loc, scale = 5, 0.015, 0.05

data = pd.read_csv('./Input_data/Woods_D.csv')
data = np.sqrt(data['D']/1000)

params = stats.norm.fit(data)
print(params)

k2, pval = stats.normaltest(data)
if pval < 1e-3:
    print('p-value: ', pval, ' suggests distribution may not be normal')

# plt.figure()
# plt.hist(data, bins=15, density=True, alpha=0.6, color='g')
# p = stats.norm.pdf(data)
# plt.plot(data, stats.norm.pdf(data), 'k', linewidth=2)
# plt.show()

median_d = 0.061
d_16 = 0.031
d_84 = 0.111

res = fmin(med_err, params[0], args=(params[1], median_d))
loc_opt = res[0]
#print(res.message)

res2 = fmin(std_err, params[1], args=(loc_opt, d_16, d_84))
scale_opt = res2[0]
#print(res2.message)

#res3 = fmin(shape_err, ae, args=(scale_opt, loc_opt, d_84-d_16))
#a_opt = res3[0]
#print(res3.message)

print(loc_opt, scale_opt)
new_data = stats.norm(loc_opt, scale_opt).rvs(10000)
new_data = new_data**2
new_data = new_data[new_data >= 0.005]
print(len(new_data))
plt.figure()
plt.hist(new_data, bins=15, density=True, alpha=0.6, color='g')
#xmin, xmax = plt.xlim()
#x = np.linspace(xmin, xmax, 100)
#p = stats.skewnorm.pdf(x, ae, loc_opt, scale_opt)
#plt.plot(x, p, 'k', linewidth=2)
plt.show()

print(np.percentile(new_data, 50), np.percentile(new_data, 16), np.percentile(new_data, 84))

# use alpha to maintain shape then optimize loc until the mean matches the estimated mean
# and scale until d16 and d84 match estimated - change function to include mean D and flow
# required to move it