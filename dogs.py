import numpy as np
import matplotlib.pyplot as plt

data_amount_per_type = 500

grey_height = 28 + 4 * np.random.randn(data_amount_per_type)
lab_height = 24 + 4 * np.random.randn(data_amount_per_type)
pudel_height = 24 + 4 * np.random.randn(data_amount_per_type)

plt.hist([grey_height, lab_height, pudel_height], stacked=False, color=['r', 'b', 'g'])
plt.show()
