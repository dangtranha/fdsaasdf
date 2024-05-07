from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

class Outlier_detector:

    # Gaussian Mixture Model
    def gmm(df, col, n_components):
        data = df[df[col].notnull()][col]
        g = GaussianMixture(n_components, max_iter=100, n_init=1)
        reshaped_data = np.array(data.values.reshape(-1,1))
        g.fit(reshaped_data)
        probs = g.score_samples(reshaped_data)
        data_probs = pd.DataFrame(np.power(10, probs), index = data.index, columns = [col+'_mixture'])
        df = pd.concat([df, data_probs],axis=1)
        return df