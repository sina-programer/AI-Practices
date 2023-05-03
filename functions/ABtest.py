import pandas as pd
import numpy as np


def ABtest(x1, x2, n=1000):
    diff = x2.mean() - x1.mean()

    data = pd.DataFrame(columns=['label', 'value'])
    for y, x in enumerate([x1, x2]):
        data = pd.concat([data, pd.DataFrame({'label': y, 'value': x})], ignore_index=True)

    result = np.empty(n)
    for i in np.arange(n):
        data['label'] = np.random.randint(0, 2, len(data))
        result[i] = data.groupby('label').mean().diff().loc[1, 'value']

    return len(result[result >= diff]) / n  # probability of being a random swing between x1-x2


a = np.array([23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23])
b = np.array([31, 28, 19, 24, 32, 27, 16, 41, 23, 32, 29, 33])

print("ABtest score (the probability of being random): ", ABtest(a, b))
