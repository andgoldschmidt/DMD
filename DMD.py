#-----
# DMD algorithms
# @author Andy Goldschmidt
#
# TODO
# - Should we create an ABC interface for DMD?
# - Should we store the DMD result as a scipy state space model (dlsim)?
# - Else, should simulate have options 'continuous' and 'euler'? (at interface)
#-----
import numpy as np
from numpy.linalg import svd, pinv, eig
import matplotlib.pyplot as plt

# ------------------------------------------------------
# -- Helper methods
# ------------------------------------------------------
def delay_embed(X, shift):
    '''
    Delay-embed the matrix X with measurements from future times.
    
    parameters:
        X: Data matrix with columns storing states at sequential time measurements
        shift: Number of future times copies to augment to the current time state       

    returns:
        shifted X: the function maps (d, t) to (shift+1, d, t-shft) which
                   is stacked into ((shift+1)*d, t-shift)
    '''
    if X.ndim != 2:
        raise ValueError('In delay_embed, invalid X matrix shape of ' + str(X.shape))
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
    def __init__(self, X2, X1, ts, **kwargs):
        '''
        Parameters:
            X2, X1:
                Data matrices with columns storing states at sequential time measurements,
                such that X2 = A X1.
            ts:
                Sequential time series at which the X (or X1) measurements were taken.

        Optional parameters:
            dmd_modes: default 'exact'
            threshold: default None

        Updates:
            self.X2,self.X1: data
            self.t0: initial time
            self.dt: timestep
            self.orig_timesteps: list of times for X1
            self.A: discrete time dynamical operator
            self.eigs: eigenvalues of Atilde
            self.modes: eigenvectors of A and Atilde
        '''
        self.X2 = X2
        self.X1 = X1

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == len(self.X1) else ts[:-1]

        dmd_modes = kwargs.get('dmd_modes', 'exact')
        threshold = kwargs.get('threshold')
        
        # I. X2 = A X1 and Atilde = U*AU
        U, S, Vt = svd(self.X1, full_matrices=False)
        if threshold:
            r = np.sum(S > threshold)
            U = U[:,:r]
            S = S[:r]
            Vt = Vt[:r,:]
        Atilde = dag(U)@self.X2@dag(Vt)@np.diag(1/S)
        self.A = self.X2@dag(Vt)@np.diag(1/S)@dag(U)

        # II. Atilde W = W Y (Eigendecomposition)
        self.eigs, W = eig(Atilde)

        # III. Two versions (eigenvectors of A)
        #      DMD_exact = X2 V S^-1 W 
        #      DMD_proj = U W
        if dmd_modes == 'exact':
            self.modes = self.X2@dag(Vt)@np.diag(1/S)@W
        elif dmd_modes == 'projected':
            self.modes = U@W
        else:
            raise ValueError('In DMD initialization, unknown dmd_mode type.')        

    @classmethod
    def from_full(cls, X, ts, **kwargs):
        '''
        Construct the LHS and RHS of the DMD model using the full data matrix.
        '''
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        return cls(X2, X1, ts, **kwargs)

    def time_spectrum(self, t):
        '''
        Returns a continous approximation to the time dynamics of A (discrete time), 
        with dimensions of (eigenvalues)x(times).

        Note that A = e^(ct_A dt) so we have for (operator,eigs) pairs of (A, Y) 
        and (ct_A, Omega), such that e^log(Y)/dt = Omega 
        '''
        if np.isscalar(t):
            # Cast eigs to complex numbers for logarithm
            return np.exp(np.log(self.eigs + 0j)*(t-self.t0)/self.dt)
        else:
            return np.array([self.time_spectrum(it) for it in t]).T
        
    def predict(self, t=None, x0=None):
        ''' 
        Return the predicted future state according to the continous approximation to A.

        Default is to predict along the original timesteps.
        '''
        x0 = self.X1[:,0] if x0 is None else x0
        t = self.orig_timesteps if t is None else t
        left = self.modes
        right = pinv(self.modes)@x0
        if np.isscalar(t):
            return left@np.diag(self.time_spectrum(t))@right
        else:
            return np.array([left@np.diag(self.time_spectrum(it))@right for it in t]).T
    
# ------------------------------------------------------
# -- DMD with control (DMDc)
# ------------------------------------------------------
class DMDc:   
    def __init__(self, X2, X1, Ups, ts, **kwargs):
        '''
        Parameters:
            X2, X1:
                Data matrices with columns storing states at sequential time measurements,
                such that X2 = A X1 + B U
            Ups:
                Control matrix with columns storing control signals at sequential times.
            ts:
                Sequential time series at which the X (or X1) measurements were taken.

        Optional parameters:
            threshold: default None

        Updates:
            self.X2,self.X1: data
            self.Ups: control
            self.t0: initial time
            self.dt: timestep
            self.orig_timesteps: list of times for X1
            self.A (self.Atilde): discrete time dynamical operator (projected)
            self.B (self.Btilde): discrete time control operator (projected)
            self.eigs: eigenvalues of Atilde
            self.modes: eigenvectors of A and Atilde
        '''
        self.X1 = X1
        self.X2 = X2
        self.Ups = Ups[:,:-1] if Ups.shape[1]==len(ts) else Ups # ONLY these 2 options
        
        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == len(self.X1) else ts[:-1]
        
        # I. Compute SVDs
        Omega = np.vstack([self.X1, self.Ups])
        Ug,Sg,Vgt = svd(Omega, full_matrices=False)
        U,S,Vt = svd(self.X2, full_matrices=False)

        threshold = kwargs.get('threshold')
        if np.any(threshold):
            # Allow for independent thresholding
            t1,t2 = 2*[threshold] if np.isscalar(threshold) else threshold
            # Threshold right hand side
            r1 = np.sum(Sg > t1)
            Ug = Ug[:,:r1]
            Sg = Sg[:r1]
            Vgt = Vgt[:r1,:]
            # Threshold left hand side
            r2 = np.sum(S > t2)
            U = U[:,:r2]
            S = S[:r2]
            Vt = Vt[:r2,:]

        # II. Compute operators
        n,_ = self.X2.shape
        self.Atilde = dag(U)@self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[:n,:])@U
        self.Btilde = dag(U)@self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[n:,:]) 

        # III. DMD modes   
        self.A = self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[:n,:])     
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A@U@W
        
        # IV. Control (TODO: Btilde -> B -> continuous B operator) 
        self.B = self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[n:,:])

    @classmethod
    def from_full(cls, X, Ups, ts, **kwargs):
        X2 = X[:, 1:]
        X1 = X[:, :-1]
        return cls(X2, X1, Ups, ts, **kwargs)

    def predict(self, control=None, x0=None):
        '''
        Predict the future state from A and B using steps from X0 as long as a control signal is available.
            Default behavior (control=None) is to use the original control. (If the underlying A is desired, 
            format zeros_like u that runs for the desired time.)

        TODO: Continuous time, en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        '''
        Ups = self.Ups if control is None else control
        xt = self.X1[:,0] if x0 is None else x0
        res = []
        for ut in Ups.T:
            xt_1 = self.A@xt + self.B@ut
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])

# ------------------------------------------------------
# -- DMDc with separate access to lhs and rhs.
# ------------------------------------------------------
class bilinear_DMDc:
    def __init__(self, X2, X1, U, ts, **kwargs):
        '''
        X2 = A X1 + U B X1

        Parameters:
            X2, X1: 
                Offset data matrices with columns storing states at sequential times
            U: 
                The bilinear control signal(s) acting on X1 in the RHS term U B X1.
            ts:
                Sequential time series at which the X (or X1) measurements were taken.

        Optional Parameters:
            shift: default 0 
                Number of time delays (necessary to match times in the u B X control term)
            threshold: default None

        Updates:
            self.X2,self.X1: data
            self.U: bilinear control signal
            self.Ups: control signal U.X1
            self.t0: initial time
            self.dt: timestep
            self.orig_timesteps: list of times for X1
            self.A (self.Atilde): discrete time dynamical operator (projected)
            self.B (self.Btilde): discrete time control operator (projected)
            self.eigs: eigenvalues of Atilde
            self.modes: eigenvectors of A and Atilde
        ''' 
        self.U = U
        self.X1 = X1
        self.X2 = X2

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == len(self.X1) else ts[:-1]

        # Partially unwrap delay embedding to make sure the correct control signals
        # are combined with the correct data times.
        self.shift = kwargs.get('shift', 0)
        self.Ups = np.einsum('sit, sjt->sijt',
                             self.U.reshape(self.shift+1, -1, len(self.orig_timesteps)),
                             self.X1.reshape(self.shift+1, -1, len(self.orig_timesteps))
                            ).reshape(-1, len(self.orig_timesteps))
        
        # I. Compute SVDs
        Omega = np.vstack([self.X1, self.Ups])
        Ug,Sg,Vgt = svd(Omega, full_matrices=False)
        U,S,Vt = svd(self.X2, full_matrices=False)

        threshold = kwargs.get('threshold')
        if np.any(threshold):
            # Allow for independent thresholding
            t1,t2 = 2*[threshold] if np.isscalar(threshold) else threshold
            # Threshold right hand side
            r1 = np.sum(Sg > t1)
            Ug = Ug[:,:r1]
            Sg = Sg[:r1]
            Vgt = Vgt[:r1,:]
            # Threshold left hand side
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
        
        # IV. Control (TODO: Btilde -> B -> continuous B operator) 
        self.B = self.X2@dag(Vgt)@np.diag(1/Sg)@dag(Ug[n:,:])

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

    def predict(self, control, x0=None):
        '''
        Predict the future state from A and B using steps from X0 as long as a control is available.

        We must use the current state of X for future state prediction! This is different than using
        the control signal from the initial fit.

        Parameters:
            control: the time-delayed control signal.
        '''
        control = self.U if control is None else control
        xt = self.X1[:,0] if x0 is None else x0 # Flat array
        res = []
        # predict one fewer because we've appended the original x0
        for t in range(control.shape[1]):
            # Careful with time delays! Outer product and flatten.
            _ct = control[:, t].reshape(self.shift+1,-1)
            _xt = xt.reshape(self.shift+1,-1)
            ups_t = np.einsum('si,sj->sij', _ct, _xt).flatten()
            xt_1 = self.A@xt + self.B@ups_t
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])



