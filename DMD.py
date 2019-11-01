#-----
# DMD algorithms
# @author Andy Goldschmidt
#
# TODO
# - Should we create an ABC interface for DMD?
# - Should we store the DMD result as a scipy state space model (dlsim)?
# - Else, should simulate have options 'continuous' and 'euler'? (at interface)
# - Predict from arbitrary initial conditions
# - Better zero control behavior (string 'Zero'?)
#-----
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

# ------------------------------------------------------
# -- Helper methods
# ------------------------------------------------------
color = plt.rcParams['axes.prop_cycle'].by_key()['color'] # store color array

def delay_embed(X, shift):
    '''
    Delay-embed the matrix X with measurements from future times.
    
    parameters:)
        X: Data matrix with columns storing states at sequential time measurements
        shift: Number of future times copies to augment to the current time state       
    '''
    _,T = X.shape
    return np.vstack([X[:,i:(T-shift)+i] for i in range(shift+1)])

def dag(X):
    return X.conj().T # Conjugate transpose (dagger) shorthand

def plot_eigs(eigs, **kwargs):
    ''' 
    Plot the provided eigenvalues (of the dynamics operator A).
    '''
    xlim = kwargs.pop('xlim', [-1.1,1.1])
    ylim = kwargs.pop('xlim', [-1.1,1.1])

    fig, ax = plt.subplots(1, **kwargs)
    ax.set_aspect('equal'), ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.scatter(eigs.real, eigs.imag)
    ax.add_artist(plt.Circle((0,0), 1, color='k', linestyle='--', fill=False))
    return fig, ax

# ------------------------------------------------------
# -- Main class
# ------------------------------------------------------
class DMD:
    def __init__(self, X, sample_times, dmd_modes='exact', threshold=None):
        self.X1 = X[:, :-1]
        self.X2 = X[:, 1:]
        
        self.t0 = sample_times[0]
        self.dt = sample_times[1] - sample_times[0]
        self.orig_timesteps = sample_times[:-1]
        
        # I. X2 = A X1 and Atilde = U*AU
        U, S, Vt = svd(self.X1, full_matrices=False)
        if threshold:
            r = np.sum(S > threshold)
            U = U[:,:r]
            S = S[:r]
            Vt = Vt[:r,:]
        self.Atilde = dag(U)@self.X2@dag(Vt)@np.diag(1/S)
        self.A = self.X2@dag(Vt)@np.diag(1/S)@dag(U)

        # II. Atilde W = W Y (Eigendecomposition)
        self.eigs, W = eig(self.Atilde)

        # III. Two versions (eigenvectors of A)
        #      DMD_exact = X2 V S^-1 W 
        #      DMD_proj = U W
        if dmd_modes == 'exact':
            self.modes = self.X2@dag(Vt)@np.diag(1/S)@W
        elif dmd_modes == 'projected':
            self.modes = U@W
        else:
            raise ValueError('In DMD initialization, unknown dmd_mode type.')

    def time_spectrum(self, t):
        '''
        Returns a continous approximation to the time dynamics of A, with dimensions
        according to (eigenvalues)x(times).

        Note that A = e^(curlyA dt) so we have for operator,eigs pairs of (A,Y) 
        and (curlyA,Omega), e^log(Y)/dt = Omega 
        '''
        if np.isscalar(t):
            # Cast eigs to complex numbers for logarithm
            return np.exp(np.log(self.eigs + 0j)*(t-self.t0)/self.dt)
        else:
            return np.array([self.time_spectrum(it) for it in t]).T
        
    def predict(self, t=None):
        ''' 
        Return the predicted future state according to the continous approximation to A.

        Default is to predict along the original timesteps.
        '''
        t = self.orig_timesteps if t is None else t 
        left = self.modes
        right = pinv(self.modes)@self.X1[:,0]
        if np.isscalar(t):
            return left@np.diag(self.time_spectrum(t))@right
        else:
            return np.array([left@np.diag(self.time_spectrum(it))@right for it in t]).T
        

# ------------------------------------------------------
# -- DMD with control (DMDc)
# ------------------------------------------------------
class DMDc:   
    def __init__(self, X, Ups, sample_times, threshold=None):
        # prelim
        self.color = plt.rcParams['axes.prop_cycle'].by_key()['color'] # set default color array
        dag = lambda X: X.conj().T # Matrix shorthand
        
        self.X1 = X[:, :-1]
        self.X2 = X[:, 1:]
        self.Ups = Ups[:,:-1] if Ups.shape[1]==len(sample_times) else Ups # ONLY these 2 options
        
        self.t0 = sample_times[0]
        self.dt = sample_times[1] - sample_times[0]
        self.orig_timesteps = sample_times[:-1]
        
        # I. Compute SVDs
        Omega = np.vstack([self.X1, self.Ups])
        Ug,Sg,Vgt = svd(Omega, full_matrices=False)
        U,S,Vt = svd(self.X2, full_matrices=False)
        if np.any(threshold):
            t1,t2 = 2*[threshold] if np.isscalar(threshold) else threshold
            r1 = np.sum(Sg > t1)
            Ug = Ug[:,:r1]
            Sg = Sg[:r1]
            Vgt = Vgt[:r1,:]
            r2 = np.sum(S > t2)
            U = U[:,:r2]
            S = S[:r2]
            Vt = Vt[:r2,:]
        # II. Compute operators
        n,_ = self.X2.shape
        self.Atilde = dag(U)@self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[:n,:])@U
        self.Btilde = dag(U)@self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[n:,:])

        # III. DMD modes        
        self.eigs, W = eig(self.Atilde)
        self.A = self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[:n,:])
        self.modes = self.A@U@W
        
        # Also need Btilde -> B -> continuous B operator 
        self.B = self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[n:,:])
#         self.Bcurly = # TODO

    def time_spectrum(self, t):
        '''
        Returns a continous approximation to the time dynamics of A, with dimensions
        according to (eigenvalues)x(times).

        Note that A = e^(curlyA dt) so we have for operator,eigs pairs of (A,Y) 
        and (curlyA,Omega), e^log(Y)/dt = Omega 
        '''
        if np.isscalar(t):
            return np.exp(np.log(self.eigs)*(t-self.t0)/self.dt)
        else:
            return np.array([self.time_spectrum(it) for it in t]).T

    def predict(self, control=None):
        '''
        Predict the future state from A and B using steps from X0 as long as a control signal is available.
            Default behavior (control=None) is to use the original control. (If the underlying A is desired, 
            format zeros_like u that runs for the desired time.)

        TODO: Continuous time, en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        '''
        Ups = self.Ups if control is None else control
        xt = self.X1[:,0]
        res = [xt]
        # Add initial point and ignore last point to match orig_timesteps
        for ut in Ups.T[:-1]:
            xt_1 = self.A@xt + self.B@ut
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])