from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)
    # Write your code below:

    sample_weight = np.zeros_like(Y, dtype=float)

    probs_y = Counter(Y)
    probs_y = {y: probs_y[y]/len(Y) for y in probs_y.keys()}

    probs_a = {a: Counter(X[a]) for a in A}

    for a in probs_a:
        for key in probs_a[a].keys():
            probs_a[a][key] /= len(X)

    probs_ay = {a: {0: Counter(), 1: Counter()} for a in A}
    for i, row in X.iterrows():
        y = Y[i]
        for a in A:
            probs_ay[a][y][row[a]] += 1
    for a in probs_ay:
        for y in probs_ay[a]:
            for key in probs_ay[a][y].keys():
                probs_ay[a][y][key] /= len(X)
    
    for i, row in X.iterrows():
        weights = []
        for a in A:
            p_a = probs_a[a][row[a]]
            p_y = probs_y[Y[i]]
            p_ay = probs_ay[a][Y[i]][row[a]]
            weight = (p_y * p_a) / p_ay if p_ay > 0 else 0
            weights.append(weight)    
        
        sample_weight[i] = np.product(weights)

    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight


