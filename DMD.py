#-----
# DMD algorithms
# @author Andy Goldschmidt
#
# TODO
# - classic DMD needs predict_dst / predict_cts
# - Should we create an ABC interface for DMD?
# - __init__.py and separate files
#-----
import numpy as np
from numpy.linalg import svd, pinv, eig
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt

# ------------------------------------------------------
# -- Helper methods
# ------------------------------------------------------
def delay_embed(X, shift):
    '''
    Delay-embed the matrix X with measurements from future times.
    
    parameters:
        X: 
            Data matrix with columns storing states at sequential time measurements
        shift: 
            Number of future times copies to augment to the current time state       

    returns:
        shifted X: 
            the function maps (d, t) to (shift+1, d, t-shft) which
            is stacked into ((shift+1)*d, t-shift)
    '''
    if X.ndim != 2:
        raise ValueError('In delay_embed, invalid X matrix shape of ' + str(X.shape))
    _,T = X.shape
    return np.vstack([X[:,i:(T-shift)+i] for i in range(shift+1)])

def dag(X):
    return X.conj().T # Conjugate transpose (dagger) shorthand

def dst_from_cts(cA, cB, dt):
    '''
    Convert constant continuous state space matrices to discrete
    matrices with time step dt using:
        exp(dt*[[cA, cB],  = [[dA, dB],
                [0,  0 ]])    [0,  1 ]]
    
    Returns:
        dA, dB

    Require cA \in R^(na x na) and cB \in R^(na x nb). The zero and 
    identity components make the matrix square.
    '''
    na,_ = cA.shape
    _,nb = cB.shape
    cM = np.block([[cA, cB],
                   [np.zeros([nb,na]), np.zeros([nb,nb])]])
    dM = expm(cM*dt)
    return dM[:na,:na], dM[:na, na:]

def cts_from_dst(dA, dB, dt):
    '''
    Convert discrete state space matrices with time step dt to 
    continuous matrices by inverting:
        exp(dt*[[cA, cB],  = [[dA, dB],
                [0,  0 ]])    [0,  1 ]]

    Returns:
        cA, cB

    Require dA \in R^(na x na) and dB \in R^(na x nb). The zero and 
    identity components make the matrix square.
    '''
    na,_ = dA.shape
    _,nb = dB.shape
    dM = np.block([[dA, dB],
                   [np.zeros([nb,na]), np.identity(nb)]])
    cM = logm(dM)/dt
    return cM[:na,:na], cM[:na, na:]

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

def threshold_svd(X, threshold, threshold_type):
    '''
    Parameters:
        X: 
            Matrix for SVD
        threshold: Pos. real, int, or None
            Truncation value for SVD results
        threshold_type: 'percent', 'count'
            Type of truncation, ignored if threshold=None
    '''
    U, S, Vt = svd(X, full_matrices=False)
    if threshold_type == 'percent':
        r = np.sum(S/np.max(S) > threshold)
    elif threshold_type == 'count':
        r = threshold
    else:
        raise ValueError('Invalid threshold_type.')
    return U[:,:r], S[:r], Vt[:r,:]

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
            dmd_modes: {'exact', 'projected'}. default 'exact'
            threshold: Real or int. default None
                Truncate the singular values associated with DMD modes
            threshold_type: (requires param threshold) {'number', 'percent'}. default 'percent'

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
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]
        
        # I. Compute SVD
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            U, S, Vt = svd(self.X1, full_matrices=False)
        else:
            threshold_type = kwargs.get('threshold_type', 'percent')
            U, S, Vt = threshold_svd(self.X1, threshold, threshold_type)

        # II: Compute operators: X2 = A X1 and Atilde = U*AU
        Atilde = dag(U)@self.X2@dag(Vt)@np.diag(1/S)
        self.A = self.X2@dag(Vt)@np.diag(1/S)@dag(U)

        # III. DMD Modes
        #       Atilde W = W Y (Eigendecomposition)
        self.eigs, W = eig(Atilde)

        # Two versions (eigenvectors of A)
        #       (i)  DMD_exact = X2 V S^-1 W 
        #       (ii) DMD_proj = U W
        dmd_modes = kwargs.get('dmd_modes', 'exact')
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
    def __init__(self, X2, X1, U, ts, **kwargs):
        '''
        Parameters:
            X2, X1:
                Data matrices with columns storing states at sequential time measurements,
                such that X2 = A X1 + B U
            U:
                Control matrix with columns storing control signals at sequential times.
            ts:
                Sequential time series at which the X (or X1) measurements were taken.

        Optional parameters:
            threshold: Real or int. default None
                Truncate the singular values associated with DMD modes
            threshold_type: (requires param threshold) {'number', 'percent'}. default 'percent'

        Updates:
            self.X2,self.X1: data
            self.U: control
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
        self.U = U if U.shape[1]==self.X1.shape[1] else U[:,:-1] # ONLY these 2 options
        Omega = np.vstack([self.X1, self.U])

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]
        
        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1,t2 = 2*[threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug,Sg,Vgt = threshold_svd(Omega, t1, threshold_type)
            U,S,Vt = threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        n,_ = self.X2.shape
        left = self.X2@dag(Vgt)@np.diag(1/Sg)
        self.A = left@dag(Ug[:n,:])     
        self.B = left@dag(Ug[n:,:])

        # III. DMD modes   
        self.Atilde = dag(U)@self.A@U
        self.Btilde = dag(U)@self.B
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A@U@W
        
    @classmethod
    def from_full(cls, X, U, ts, **kwargs):
        X2 = X[:, 1:]
        X1 = X[:, :-1]
        return cls(X2, X1, U, ts, **kwargs)

    def predict_dst(self, control=None, x0=None):
        '''
        Predict the future state from A and B using steps from X0 as long as a control signal is available
        using the discrete system equation,
                    X_2 = A X_1 + B u_1

        Default behavior (control=None) is to use the original control. (If the underlying A is desired, 
        format zeros_like u that runs for the desired time.)
        '''
        U = self.U if control is None else control
        xt = self.X1[:,0] if x0 is None else x0
        res = [xt]
        for ut in U[:,:-1].T:
            xt_1 = self.A@xt + self.B@ut
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        '''
        Predict the future state from A and B using steps from X0 as long as a control signal is available
        using the continuous system equation,
                    X_dot = A X + B u
                 => x(t+dt) = e^{dt A}(x(t) + dt B u(t))

        Default behavior (control=None) is to use the original control. (If the underlying A is desired, 
        format zeros_like u that runs for the desired time.) Be sure that dt matches the train dt if
        using delay embeddings.

        Parameters:
            control: 
                the time-delayed control signal. Must match the 
                dimensions of the training control signal.
            x0:
                the initial value
            dt:
                the time-step along which the control changes
        '''
        U = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:,0] if x0 is None else x0
        res = [xt]
        for ut in U[:,:-1].T:
            xt_1 = expm(dt*self.A)@(xt + dt*self.B@ut)
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.U.shape[0], n_steps])

# ------------------------------------------------------
# -- bilinear DMD
# ------------------------------------------------------
class biDMD:
    def __init__(self, X2, X1, U, ts, **kwargs):
        '''
        X2 = A X1 + U B X1

        Parameters:
            X2, X1: 
                Offset data matrices with columns storing states at sequential times.
                Altenatively, X2 = X_dot and X1 = X.
            U: 
                The bilinear control signal(s) acting on X1 in the RHS term U B X1.
            ts:
                Sequential time series at which the X (or X1) measurements were taken.

        Optional Parameters:
            shift: default 0 
                Number of time delays (necessary to match times in the u B X control term)
            threshold: Real or int. default None
                Truncate the singular values associated with DMD modes
            threshold_type: (requires param threshold) {'number', 'percent'}. default 'percent'

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
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # store useful dimension
        n_time = len(self.orig_timesteps)

        # Partially unwrap delay embedding to make sure the correct control signals
        #   are combined with the correct data times. The unwrapped (=>) operators:
        #     X1  => (delays+1) x (measured dimensions) x (measurement times)
        #     U   => (delays+1) x (number of controls)  x (measurement times)
        #     Ups => (delays+1) x (controls) x (measured dimensions) x (measurement times)
        #         => (delays+1 x controls x measured dimensions) x (measurement times)
        #   Re-flatten all but the time dimension of Ups to set the structure of the
        #   data matrix. This will set the strucutre of the B operator to match our
        #   time-delay function.
        self.shift = kwargs.get('shift', 0)
        self.Ups = np.einsum('sct, smt->scmt',
                             self.U.reshape(self.shift+1, -1, n_time),
                             self.X1.reshape(self.shift+1, -1, n_time)
                            ).reshape(-1, n_time)
        Omega = np.vstack([self.X1, self.Ups])
        
        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1,t2 = 2*[threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug,Sg,Vgt = threshold_svd(Omega, t1, threshold_type)
            U,S,Vt = threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        n,_ = self.X2.shape
        left = self.X2@dag(Vgt)@np.diag(1/Sg)
        self.A = left@dag(Ug[:n,:])
        self.B = left@dag(Ug[n:,:])

        # III. DMD modes
        self.Atilde = dag(U)@self.A@U
        self.Btilde = dag(U)@self.B   
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A@U@W        

    def predict_dst(self, control=None, x0=None):
        '''
        Predict the future state from A and B using steps from X0 as long as 
        a control is available, solving the discrete evolution
            x_1 = A x_0 + B (u.x_0) = [A B] [x_0,
                                             u.x_0]

        We must use the current state of X for future state prediction! This is
        different than using the control signal from the initial fit.

        Parameters:
            control: 
                the time-delayed control signal
            x0:
                the initial value
        '''
        control = self.U if control is None else control
        xt = self.X1[:,0] if x0 is None else x0 # Flat array
        res = [xt]
        for t in range(control.shape[1]-1):
            # Outer product then flatten to correctly combine the different
            #   times present due to time-delays. That is, make sure that
            #   u(t)'s multiply x(t)'s
            #     _ct    => (time-delays + 1) x (number of controls)
            #     _xt    => (time-delays + 1) x (measured dimensions)
            #     _ups_t => (time-delays + 1) x (controls) x (measurements)
            #   Flatten to get the desired vector.
            _ct = control[:, t].reshape(self.shift+1, -1)
            _xt = xt.reshape(self.shift+1, -1)
            ups_t = np.einsum('sc,sm->scm', _ct, _xt).flatten()

            xt_1 = self.A@xt + self.B@ups_t
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        '''
        Continuous control predicts X_dot = (A + uB)X for a control signal
        over time-steps of dt, applying
            x_1 = e^{A dt + u B dt } x_0
        across each time-step dt where u is constant. Be sure that dt matches
        the train dt if using delay embeddings.
    
        Parameters:
            control: 
                the time-delayed control signal. Must match the 
                dimensions of the training control signal.
            x0:
                the initial value
            dt:
                the time-step along which the control changes
        '''
        control = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:,0] if x0 is None else x0 # Flat array

        # store useful dimensions
        delay_dim = self.shift + 1
        control_dim = self.U.shape[0]//delay_dim
        measure_1_dim = self.X1.shape[0]//delay_dim
        to_dim = self.X2.shape[0]

        res = [xt]
        for t in range(control.shape[1]-1):
            # Correctly combine u(t) and B(t)
            #   Initial:
            #     B      <= (time-delays+1 x measurements_2) x (time-delays+1 x controls x measurements_1)
            #   Reshape:
            #     B      => (time-delays+1 x measurements_2) x (time-delays+1) x (controls) x (measurements_1)
            #     _ct    => (time-delays+1) x (controls) 
            #     _uBt   => (time-delays+1 x measurements_2) x (time-delays+1) x (measurements_1)
            #            => (time-delays+1 x measurements_2) x (time-delays+1 x measurements_1)
            #   Notice that _uBt is formed by a sum over all controls in order to act on the
            #   state xt which has dimensions of (delays x measurements_1).
            _uBt = np.einsum('ascm,sc->asm',
                             self.B.reshape(to_dim, delay_dim, control_dim, measure_1_dim), 
                             control[:, t].reshape(delay_dim, control_dim)
                            ).reshape(to_dim, delay_dim*measure_1_dim)

            xt_1 = expm((self.A + _uBt)*dt)@xt
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])

# ------------------------------------------------------
# -- bilinear DMD with control
# ------------------------------------------------------
class biDMDc:
    def __init__(self, X2, X1, U, ts, **kwargs):
        '''
        X2 = A X1 + U B X1 + D U

        Parameters:
            X2, X1: 
                Offset data matrices with columns storing states at sequential times.
                Altenatively, X2 = X_dot and X1 = X.
            U: 
                The bilinear control signal(s) acting on X1 in the RHS term U B X1.
            ts:
                Sequential time series at which the X (or X1) measurements were taken.

        Optional Parameters:
            shift: default 0 
                Number of time delays (necessary to match times in the u B X control term)
            threshold: Real or int. default None
                Truncate the singular values associated with DMD modes
            threshold_type: (requires param threshold) {'number', 'percent'}. default 'percent'

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
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # store useful dimension
        n_time = len(self.orig_timesteps)
        self.shift = kwargs.get('shift', 0)
        delay_dim = self.shift + 1

        # Partially unwrap delay embedding to make sure the correct control signals
        #   are combined with the correct data times. The unwrapped (=>) operators:
        #     X1  => (delays+1) x (measured dimensions) x (measurement times)
        #     U   => (delays+1) x (number of controls)  x (measurement times)
        #     Ups => (delays+1) x (controls) x (measured dimensions) x (measurement times)
        #         => (delays+1 x controls x measured dimensions) x (measurement times)
        #   Re-flatten all but the time dimension of Ups to set the structure of the
        #   data matrix. This will set the strucutre of the B operator to match our
        #   time-delay function.
        self.Ups = np.einsum('sct, smt->scmt',
                             self.U.reshape(delay_dim, -1, n_time),
                             self.X1.reshape(delay_dim, -1, n_time)
                            ).reshape(-1, n_time)
        Omega = np.vstack([self.X1, self.Ups, self.U])
        
        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1,t2 = 2*[threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug,Sg,Vgt = threshold_svd(Omega, t1, threshold_type)
            U,S,Vt = threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        c = self.U.shape[0]//delay_dim
        n = self.X1.shape[0]
        left = self.X2@dag(Vgt)@np.diag(1/Sg)
        # Omega = X + uX + u => dim'ns: n + c*n + c
        self.A = left@dag(Ug[:n,:])
        self.B = left@dag(Ug[n:(c+1)*n,:])
        self.D = left@dag(Ug[(c+1)*n:, :])

        # III. DMD modes
        self.Atilde = dag(U)@self.A@U
        self.Btilde = dag(U)@self.B
        self.Dtilde = dag(U)@self.D
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A@U@W

    def predict_dst(self, control=None, x0=None):
        '''
        Predict the future state from A and B using steps from X0 as long as 
        a control is available, solving the discrete evolution
            x_1 = A x_0 + B (u.x_0) = [A B] [x_0,
                                             u.x_0]

        We must use the current state of X for future state prediction! This is
        different than using the control signal from the initial fit.

        Parameters:
            control: 
                the time-delayed control signal
            x0:
                the initial value
        '''
        control = self.U if control is None else control
        xt = self.X1[:,0] if x0 is None else x0 # Flat array
        res = [xt]
        for t in range(control.shape[1]-1):
            # Outer product then flatten to correctly combine the different
            #   times present due to time-delays. That is, make sure that
            #   u(t)'s multiply x(t)'s
            #     _ct    => (time-delays + 1) x (number of controls)
            #     _xt    => (time-delays + 1) x (measured dimensions)
            #     _ups_t => (time-delays + 1) x (controls) x (measurements)
            #   Flatten to get the desired vector.
            _ct = control[:, t].reshape(self.shift+1, -1)
            _xt = xt.reshape(self.shift+1, -1)
            ups_t = np.einsum('sc,sm->scm', _ct, _xt).flatten()

            xt_1 = self.A@xt + self.B@ups_t + self.D@control[:,t]
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        '''
        Continuous control predicts X_dot = (A + uB)X for a control
        signal over time-steps of dt, applying
            x_1 = e^{A dt + u B dt } x_0
        across each time-step dt where u is constant.
    
        Parameters:
            control: 
                the time-delayed control signal. Must match the 
                dimensions of the training control signal.
            dt:
                the time-step along which the control changes
            x0:
                the initial value
        '''
        control = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:,0] if x0 is None else x0 # Flat array

        # store useful dimensions
        delay_dim = self.shift + 1
        control_dim = self.U.shape[0]//delay_dim
        measure_1_dim = self.X1.shape[0]//delay_dim
        to_dim = self.X2.shape[0]

        res = [xt]
        for t in range(control.shape[1]-1):
            # Correctly combine u(t) and B(t)
            #   Initial:
            #     B      <= (time-delays+1 x measurements_2) x (time-delays+1 x controls x measurements_1)
            #   Reshape:
            #     B      => (time-delays+1 x measurements_2) x (time-delays+1) x (controls) x (measurements_1)
            #     _ct    => (time-delays+1) x (controls) 
            #     _uBt   => (time-delays+1 x measurements_2) x (time-delays+1) x (measurements_1)
            #            => (time-delays+1 x measurements_2) x (time-delays+1 x measurements_1)
            #   Notice that _uBt is formed by a sum over all controls in order to act on the
            #   state xt which has dimensions of (delays x measurements_1).
            _uBt = np.einsum('ascm,sc->asm',
                             self.B.reshape(to_dim, delay_dim, control_dim, measure_1_dim), 
                             control[:, t].reshape(delay_dim, control_dim)
                            ).reshape(to_dim, delay_dim*measure_1_dim)

            xt_1 = expm(dt*(self.A + _uBt))@(xt + dt*self.D@control[:, t])
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])

