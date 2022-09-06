from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, fmin


def mean_err(loc, a, scale, median_obs):
    dist = stats.skewnorm(a, loc, scale).rvs(1000)
    dist = dist[dist > 0.0005]
    median_est = np.median(dist)
    err = abs(median_est - median_obs)

    return err


def std_err(scale, a, loc, std_orig):
    if scale <= 0:
        scale = abs(scale)
    dist = stats.skewnorm(a, loc, scale).rvs(1000)
    dist = dist[dist > 0.0005]
    err = abs(np.std(dist) - std_orig)

    return err


def shape_err(a, scale, loc, d16, d84):
    dist = stats.skewnorm(a, loc, scale).rvs(1000)
    dist = dist[dist > 0.0005]
    err = abs(np.percentile(dist, 16) - d16)**2 + abs(np.percentile(dist, 84) - d84)**2

    return err



a, loc, scale = 5, 0.015, 0.05

data = pd.read_csv('./Input_data/Woods_D.csv')
data = data['D']/1000

ae, loce, scalee = stats.skewnorm.fit(data)
print(ae, loce, scalee)

plt.figure()
plt.hist(data, bins=15, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.skewnorm.pdf(x, ae, loce, scalee)
plt.plot(x, p, 'k', linewidth=2)
plt.show()

median_d = 0.065
d_16 = 0.021
d_84 = 0.138

res = fmin(mean_err, loce, args=(ae, scalee, median_d))
loc_opt = res[0]
#print(res.message)

res2 = fmin(std_err, scalee, args=(ae, loc_opt, np.std(data)))
scale_opt = res2[0]
#print(res2.message)

res3 = fmin(shape_err, ae, args=(scale_opt, loc_opt, d_16, d_84))
a_opt = res3[0]
#print(res3.message)

new_data = stats.skewnorm(a_opt, loc_opt, scale_opt).rvs(1000)
new_data = new_data[new_data >= 0.005]
print(len(new_data))
plt.figure()
plt.hist(new_data, bins=15, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.skewnorm.pdf(x, ae, loc_opt, scale_opt)
plt.plot(x, p, 'k', linewidth=2)
plt.show()

print(np.percentile(new_data, 50), np.percentile(new_data, 16), np.percentile(new_data, 84))

# use alpha to maintain shape then optimize loc until the mean matches the estimated mean
# and scale until d16 and d84 match estimated - change function to include mean D and flow
# required to move it