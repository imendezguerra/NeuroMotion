
import numpy as np

class MNPoolDenSoma:
    
    def __init__(self, N, ms_name) -> None:
        
        # Number of MNs and muscle they belong to
        self.N = N
        self.ms_name = ms_name
        
        # Constants and parameters
        self.Cm = 1.0 * 1e-2 # membrane capacitance (F/m^2)
        self.Ri = 70 * 1e-2 # cytoplasmic resistivity (ohm * m)
        
        # Eequilibrium potentials 
        self.E_L = 0 # leakage potential, equal to the membrane resting potential (V)
        self.E_Na = 120.0 * 1e-3 # Na+ equilibrium potential (V)
        self.E_K = -77.0 * 1e-3  # K+ equilibrium potential (V)
        self.E_Ca = 140.0 * 1e-3  # Ca2+ equilibrium potential (V)

        # Maximal ionic conductances (S/m^2)
        self.gNa = 30.0 * 1e1 # Na+ channels - depolarisation (S/m^2)
        self.gKf = 4.0 * 1e1 # Fast K+ channels - repolarisation (S/m^2)

        # Coupling conductance between soma and dendrite
        self.gc = 0.1 * 1e1  # (S/m^2)

        # State variables
        self.pulse_width = 0.6 * 1e-3 # (s) Duration of the somatic channel opening/closing

        self.alphaM = 22 * 1e-3 # (s^-1)
        self.betaM = 13 * 1e-3 # (s^-1)

        self.alphaH = 0.5 * 1e-3 # (s^-1)
        self.betaH = 4 * 1e-3 # (s^-1)

        self.alphaN = 1.5 * 1e-3 # (s^-1)
        self.betaN = 0.1 * 1e-3 # (s^-1)

        self.alphaQ = 1.5 * 1e-3 # (s^-1)

        self.alphaP = 0.008 * 1e-3 # (s^-1)

        self.m0 = 0 # (n.u.)
        self.h0 = 1 # (n.u.)
        self.n0 = 0 # (n.u.)
        self.q0 = 0 # (n.u.)
        self.p0 = 0 # (n.u.)

        # Neuromodulation level (0 to 1)
        self.gamma = 1.0

        self._init_pool()

    def _init_pool(self):

        # Size properties of the pool
        self.r_soma = np.linspace(38.75, 41.25, self.N) * 1e-6 # (m)
        self.l_soma = np.linspace(70.5, 82.5, self.N) * 1e-6 # (m)

        self.r_dend = np.linspace(20.75, 31.25, self.N) * 1e-6 # (m)
        self.l_dend = np.linspace(5.5, 6.8, self.N) * 1e-3 # (m)

        # Resistance and capacitance of the pool
        self.Rm_soma = np.linspace(1.15, 1.05, self.N) * 1e-1 # (ohm * m^2)
        self.Rm_dend = np.linspace(14.4, 10.7, self.N) * 1e-1 # (ohm * m^2)

        # Capacitance of the pool
        self.Cm_soma = 2 * np.pi * self.r_soma * self.l_soma * self.Cm
        self.Cm_dend = 2 * np.pi * self.r_dend * self.l_dend * self.Cm

        # Conductances
        aux_soma = self.Ri * self.l_soma / (np.pi * self.r_soma**2)
        aux_dend = self.Ri * self.l_dend / (np.pi * self.r_dend**2)
        self.gc =  2 / (aux_soma + aux_dend)

        self.gL_soma = (2 * np.pi * self.r_soma * self.l_soma) / self.Rm_soma
        self.gL_dend = (2 * np.pi * self.r_dend * self.l_dend) / self.Rm_dend

        # Rheobase current and voltage
        self.I_rheo = np.linspace(3.5, 6.5, self.N) * 1e-9 # (A)
        self.Rn = 1/(self.gL_soma + (self.gL_dend * self.gc)/(self.gL_dend + self.gc)) # (ohm)
        self.Vth = self.Rn * self.I_rheo # (V)

        # Axon conduction velocity
        self.cv = np.linspace(44, 47, self.N) # (m/s)

        # Ionic conductances and state variables
        self.gKs = np.linspace(16, 25, self.N) * 1e1 # (S/m^2) Slow K+ channels - hyperpolarisation
        self.gCa = np.linspace(0.038, 0.029, self.N) * 1e1 # (S/m^2) for L-type Ca2+ channels

        self.betaQ = np.linspace(0.025, 0.038, self.N) * 1e-3 # (s^-1)
        self.betaP = np.linspace(0.014, 0.016, self.N) * 1e-3 # (s^-1)
        
        self.Vth_Ca = np.linspace(2.5, 3.0, self.N) * 1e-3 # (V)

    def _init_vars(self, t_samples):
            
        # Initial conditions
        self.V_soma = np.zeros(self.N, t_samples) # (V)
        self.V_dend = np.zeros(self.N, t_samples) # (V)

        self.m = np.zeros(self.N, t_samples) # (n.u.)
        self.h = np.ones(self.N, t_samples) # (n.u.)
        self.n = np.zeros(self.N, t_samples) # (n.u.)
        self.q = np.zeros(self.N, t_samples) # (n.u.)
        self.p = np.zeros(self.N, t_samples) # (n.u.)
         
        # Initialise currents
        self.I_Na = np.zeros(self.N, t_samples) # (A)
        self.I_Kf = np.zeros(self.N, t_samples) # (A)
        self.I_Ks = np.zeros(self.N, t_samples) # (A)
        self.I_Ca = np.zeros(self.N, t_samples) # (A)
        self.I_L_soma = np.zeros(self.N, t_samples) # (A)
        self.I_L_dend = np.zeros(self.N, t_samples) # (A)
        # self.I_syn = np.zeros(self.N, t_samples) # (A) Inputs
        # self.I_inj = np.zeros(self.N, t_samples) # (A) Inputs

        # Spike train
        self.spike_train = np.empty((self.N,), dytpe=object)

    def _eq_solver(self, n, I_syn, I_inj, timestamps):

        self.spike_train[n] = []

        for t, timestamp in enumerate(timestamps):
            
            if t == 0:
                t_spike = t
                continue

            dt = timestamps[t] - timestamps[t-1]

            # Check for spike condition
            if t - t_spike < self.pulse_width:
                # Update gating variables
                self.m[n,t] = 1 + (self.m0 - 1) * np.exp(-self.alphaM * (t - t_spike))
                self.h[n,t] = self.h0 * np.exp(-self.betaH * (t - t_spike))
                self.n[n,t] = 1 + (self.n0 - 1) * np.exp(-self.alphaN * (t - t_spike))
                self.q[n,t] = 1 + (self.q0 - 1) * np.exp(-self.alphaQ * (t - t_spike))
                self.p[n,t] = 1 + (self.p0 - 1) * np.exp(-self.alphaP * (t - t_spike))
            else:
                # Update gating variables
                self.m[n,t] = self.m0 * np.exp(-self.betaM * (t - t_spike))
                self.h[n,t] = 1 + (self.h0 - 1) * np.exp(-self.alphaH * (t - t_spike))
                self.n[n,t] = 1 + (self.m0 - 1) * np.exp(-self.alphaM * (t - t_spike))
                self.q[n,t] = self.q0 * np.exp(-self.betaQ * (t - t_spike))
                self.p[n,t] = self.p0 * np.exp(-self.betaP * (t - t_spike))

            # Ionic currents
            self.I_Na[n,t] = self.gNa * (self.m[n,t]**3) * self.h[n,t] * (self.V_soma[n,t-1] - self.E_Na)
            self.I_Kf[n,t] = self.gKf * (self.n[n,t]**4) * (self.V_soma[n,t-1] - self.E_K)
            self.I_Ks[n,t] = self.gKs * (self.q[n,t]**2) * (self.V_soma[n,t-1] - self.E_K)
            self.I_Ca[n,t] = self.gamma * self.gCa * self.p[n,t] * (self.V_dend[n,t-1] - self.E_Ca)

            # Update membrane potentials
            self.V_soma[n,t] = self.V_soma[t-1] + dt * (1/self.Cm_soma) * (-self.gL_soma * (self.V_soma[n,t-1] - self.E_L) + self.gc * (self.V_soma[n,t-1] - self.V_dend[n,t-1]) - self.I_Na[n,t] - self.I_Kf[n,t] - self.I_Ks[n,t] + I_inj[n,t])
            self.V_dend[n,t] = self.V_dend[t-1] + dt * (1/self.Cm_dend) * (-self.gL_dend * (self.V_dend[n,t-1] - self.E_L) + self.gc * (self.V_dend[n,t-1] - self.V_soma[n,t-1]) - self.I_Ca[n,t] - I_syn[n,t])

            # Check for spikes
            if self.V_soma[n,t] > self.Vth[n]:
                t_spike = t
                self.spike_train[n].append(timestamp)


    def get_spike_train(self, timestamps, Isyn, Iinj=None):

        # Initialise variables based on time and number of MNs
        t_samples = len(timestamps)
        self._init_vars(t_samples)

        # Loop through each MN
        for n in range(self.N):
            self._eq_solver(n, Isyn, Iinj, timestamps)

        return self.spike_train
    




