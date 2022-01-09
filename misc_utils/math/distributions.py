import numpy as np

def normal(x,mu,sigma):
    return ( 2.*np.pi*sigma**2.)**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )


def normal2(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * (np.e * -0.5 * ((x-mu)**2) / (sigma**2))
