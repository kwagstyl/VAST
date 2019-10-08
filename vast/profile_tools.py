

import scipy.ndimage as nd



#taubin smoothing.
def taubin(profile, lam=0.5, n_iter=10):
        profile = profile.copy()
        for ii in range(n_iter):
            displ = nd.laplace(profile)
            profile += lam * displ
        return profile

