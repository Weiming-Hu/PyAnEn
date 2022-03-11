import numpy as np
import matplotlib.pyplot as plt

from ranky import rankz
from PyAnEn.utils_verify import rank_histogram

def test_RankHist():
    
    # This is the package I'm verifying against
    # https://github.com/oliverangelil/rankhistogram
    #
    
    obs = np.random.randint(5, 95, (10, 20, 30)) + 0.1
    ensemble = np.random.randint(1, 100, (40, 10, 20, 30))

    # Feed into rankz function
    mask = np.full((10, 20, 30), 1)
    theirs = rankz(obs, ensemble, mask)
    
    for ens_axis in [0, 1, 2, 3]:
        
        # Calculate my rank histograms
        ranks = rank_histogram(np.moveaxis(ensemble, 0, ens_axis), obs, ens_axis)
        mine = plt.hist(ranks.ravel(), bins=theirs[1])

        assert np.all(mine[0] == theirs[0]), 'Failed with ensemble axis {}'.format(ens_axis)
