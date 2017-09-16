import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
d1 = rs.normal(size=200) * 100 + 450
d2 = rs.normal(size=200) * 100 + 500

# Plot a kernel density estimate and rug plot
sns.distplot(d1, hist=False, color="r", ax=axes[0, 1])
# Plot a kernel density estimate and rug plot
sns.distplot(d2, hist=False, color="g", ax=axes[0, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()

plt.plot()
plt.show()
