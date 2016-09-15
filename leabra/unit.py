"""
Implementation of a Leabra Unit, reproducing the behavior of emergent 8.0.

We implement only the rate-coded version. The code is intended to be as simple
as possible to understand. It is not in any way optimized for performance.
"""
import copy

import numpy as np
import scipy.interpolate



class Unit:
    """Leabra Unit (as implemented in emergent 8.0)"""

    def __init__(self, spec=None, log_names=('net', 'I_net', 'v_m', 'act', 'v_m_eq', 'adapt')):
        """
        spec:  UnitSpec instance with custom values for the unit parameters.
               If None, default values will be used.
        """
        self.spec = spec
        if self.spec is None:
            self.spec = UnitSpec()

        self.ex_inputs  = []    # excitatory inputs for the next cycle
        self.forced     = False # Is activity directly set?
        self.forced_act = None  # if forced, act == forced_act

        self.g_e     = 0
        self.I_net   = 0
        self.I_net_r = self.I_net
        self.v_m     = 0.4
        self.v_m_eq  = self.v_m
        self.act     = 0         # current activity
        self.act_m   = self.act  # activity at the end of the minus phase
        self.act_nd  = self.act  # non-depressed activity # FIXME: not implemented yet

        self.adapt   = 0     # adaptation current: causes the rate of activation
                              # to decrease over time

        # averages
        self.avg_ss    = 0.15 # FIXME: why 0.15?
        self.avg_s     = 0.15
        self.avg_m     = 0.15
        self.avg_l     = 0.15 # FIXME: may be different, investigate `avg_l.init`
        self.avg_l_lrn = 1.0
        self.avg_s_eff = 0.0

        self.logs  = {name: [] for name in log_names}


    def cycle(self, g_i=0.0, dt_integ=1):
        """Cycle the unit"""
        return self.spec.cycle(self, g_i=g_i, dt_integ=dt_integ)

    def calculate_net_in(self):
        return self.spec.calculate_net_in(self)


    @property
    def net(self):
        """Excitatory conductance."""
        return self.spec.g_bar_e * self.g_e


    def add_excitatory(self, inp_act, forced=False):
        """Add an input for the next cycle.

        If the activity is directly set (`forced==True`), then at most one call
        to add_excitatory should be done per cycle. If no call is made and the
        last call was forced, then the subsequent cycles continue to force this
        value on the unit activity.
        """
        self.forced = forced
        if self.forced: # the activity is set directly
            assert len(self.ex_inputs) == 0
            self.forced_act = inp_act
        else:
            self.ex_inputs.append(inp_act)


    def update_logs(self):
        """Record current state. Called after each cycle."""
        for name in self.logs.keys():
            self.logs[name].append(getattr(self, name))


    def show_config(self):
        """Display the value of constants and state variables."""
        print('Parameters:')
        for name in ['dt_vm', 'dt_net', 'g_l', 'g_bar_e', 'g_bar_l', 'g_bar_i',
                     'e_rev_e', 'e_rev_l', 'e_rev_i', 'act_thr', 'act_gain']:
            print('   {}: {:.2f}'.format(name, getattr(self.spec, name)))
        print('State:')
        for name in ['g_e', 'I_net', 'v_m', 'act', 'v_m_eq']:
            print('   {}: {:.2f}'.format(name, getattr(self, name)))



class UnitSpec:
    """Units specification.

    Each unit can have different parameters values. They don't change during
    cycles, and unless you know what you're doing, you should not change them
    after the Unit creation. The best way to proceed is to create the UnitSpec,
    modify it, and provide the spec when instantiating a Unit:

    >>> spec = UnitSpec(act_thr=0.35) # specifying parameters at instantiation
    >>> spec.bias = 0.5               # you can also do it afterward
    >>> u = Unit(spec=spec)           # creating a Unit instance

    """

    def __init__(self, **kwargs):
        # time step constants
        self.dt_net     = 1/1.4   # for net update (net = g_e * g_bar_e)
        self.dt_vm      = 1/3.3   # for vm update
        # input channels parameters
        self.g_l        = 1.0     # leak current (constant)
        self.g_bar_e    = 0.3     # excitatory maximum conductance
        self.g_bar_i    = 1.0     # inhibitory maximum conductance
        self.g_bar_l    = 0.3     # leak maximum conductance
        # reversal potential
        self.e_rev_e    = 1.0     # excitatory
        self.e_rev_i    = 0.25    # inhibitory
        self.e_rev_l    = 0.3     # leak
        # activation function parameters
        self.act_thr    = 0.5     # threshold
        self.act_gain   = 40      # gain
        self.noisy_act  = True    # If True, uses the noisy activation function
        self.act_sd     = 0.01    # standard deviation of the noisy gaussian
        # spiking behavior
        self.spk_thr    = 1.2     # spike threshold for resetting v_m # FIXME: actually used?
        self.v_m_r      = 0.3     # reset value for v_m
        # adapt behavior
        self.adapt_on   = False   # if True, enable the adapt behavior
        self.dt_adapt   = 1/144.  # time-step constant for adapt update
        self.vm_gain    = 0.04    # FIXME: desc
        self.spike_gain = 0.00805 # FIXME: desc
        # bias #FIXME: not implemented.
        self.bias       = 0.0
        # average parameters
        self.avg_ss_dt  = 0.5
        self.avg_s_dt   = 0.5
        self.avg_m_dt   = 0.1
        self.avg_l_dt   = 0.1 # computed once every trial
        self.avg_l_min  = 0.1
        self.avg_l_max  = 1.5
        self.avg_m_in_s = 0.1

        for key, value in kwargs.items():
            assert hasattr(self, key) # making sure the parameter exists.
            setattr(self, key, value)

        self._nxx1_conv = None # precomputed convolution for the noisy xx1 function


    def copy(self):
        """Return a copy of the spec"""
        return copy.deepcopy(self)


    def xx1(self, v_m):
        """Compute the x/(x+1) function."""
        X = self.act_gain * max(v_m, 0.0)
        return X / (X + 1)


    def noisy_xx1(self, v_m):
        """Compute the noisy x/(x+1) function.

        The noisy x/(x+1) function is the convolution of the x/(x+1) function
        with a Gaussian with a `self.spec.act_sd` standard deviation. Here, we
        precompute the convolution as a look-up table, and interpolate it with
        the desired point every time the function is called.
        """
        if self._nxx1_conv is None:  # convolution not precomputed yet
            res = 0.001 # resolution of the precomputed array

            # computing the gaussian
            ns_rng = max(3.0 * self.act_sd, res)
            xs = np.arange(-ns_rng, ns_rng+res, res)  # x represents self.v_m
            var = max(self.act_sd, 1.0e-6)**2
            gaussian = np.exp(-xs**2 / var)   # computing unscaled guassian
            gaussian = gaussian/sum(gaussian) # normalization

            # computing xx1 function
            xs = np.arange(-2*ns_rng, 1.0 + ns_rng + res, res)  # x represents self.v_m
            X  = self.act_gain * np.maximum(xs, 0)
            xx1 = X / (X + 1)  # regular x/(x+1) function over xs

            # convolution
            conv = np.convolve(xx1, gaussian, mode='same')

            # cutting to valid range
            xs_valid = np.arange(-ns_rng, 1.0 + res, res)  # x represents self.v_m
            conv = conv[np.searchsorted(xs, xs_valid[0],  side='left'):
                        np.searchsorted(xs, xs_valid[-1], side='left')+1]
            assert len(xs_valid) == len(conv)

            self._nxx1_conv = xs_valid, conv

        xs, conv = self._nxx1_conv
        if v_m < xs[0]:
            return 0.0
        elif v_m > xs[-1]:
            return self.xx1(xs)
        else:
            return float(scipy.interpolate.interp1d(xs, conv, kind='linear',
                                                    fill_value='extrapolate')(v_m))


    def calculate_net_in(self, unit, dt_integ=1):
        """Calculate the net input for the unit. To execute before cycle()"""
        if unit.forced:
            unit.act = unit.forced_act
            return # done!

        # computing net_raw, the total, instantaneous, excitatory input for the neuron
        net_raw = sum(unit.ex_inputs) # / max(1, len(self.ex_inputs))
        unit.ex_inputs = []

        # updating net
        unit.g_e += dt_integ * self.dt_net * (net_raw - unit.g_e)  # eq 2.16


    def cycle(self, unit, g_i=0.0, dt_integ=1):
        """Update activity

        unit    :  the unit to cycle
        g_i     :  inhibitory input
        dt_integ:  integration time step, in ms.
        """

        if unit.forced:
            return # done!

        # computing I_net and I_net_r
        unit.I_net   = self.integrate_I_net(unit, g_i, dt_integ, ratecoded=False, steps=2) # half-step integration
        unit.I_net_r = self.integrate_I_net(unit, g_i, dt_integ, ratecoded=True,  steps=1) # one-step integration

        # updating v_m and v_m_eq
        unit.v_m    += dt_integ * self.dt_vm * unit.I_net   # - unit.adapt is done on the I_net value.
        unit.v_m_eq += dt_integ * self.dt_vm * unit.I_net_r

        # reseting v_m if over the threshold (spike-like behavior)
        if unit.v_m > self.act_thr:
            unit.spike = 1
            unit.v_m = self.v_m_r
        else:
            unit.spike = 0

        # selecting the activation function, noisy or not.
        act_fun = self.noisy_xx1 if self.noisy_act else self.xx1

        # computing new_act, from v_m_eq (because rate-coded neuron)
        if unit.v_m_eq <= self.act_thr:
            new_act = act_fun(unit.v_m_eq - self.act_thr)
        else:
            gc_e = self.g_bar_e * unit.g_e
            gc_i = self.g_bar_i * g_i
            gc_l = self.g_bar_l * self.g_l
            g_e_thr = (  gc_i * (self.e_rev_i - self.act_thr)
                       + gc_l * (self.e_rev_l - self.act_thr)
                       - unit.adapt) / (self.act_thr - self.e_rev_e)
            new_act = act_fun(gc_e - g_e_thr)  # gc_e == unit.net

        # updating activity
        unit.act_nd += dt_integ * self.dt_vm * (new_act - unit.act_nd)
        unit.act = unit.act_nd # FIXME: implement stp

        # updating adaptation
        if self.adapt_on:
            unit.adapt += dt_integ * (
                            self.dt_adapt * (self.vm_gain * (unit.v_m - self.e_rev_l) - unit.adapt)
                            + unit.spike * self.spike_gain
                          )

        self.update_avgs(unit, dt_integ)
        unit.update_logs()

    def update_avgs(self, unit, dt_integ):
        """Update all averages except long-term, at the end of every cycle."""
        unit.avg_ss += dt_integ * self.avg_ss_dt * (unit.act_nd - unit.avg_ss)
        unit.avg_s  += dt_integ * self.avg_s_dt  * (unit.avg_ss - unit.avg_s )
        unit.avg_m  += dt_integ * self.avg_m_dt  * (unit.avg_s  - unit.avg_m )
        unit.avg_s_eff = self.avg_m_in_s * unit.avg_m + (1 - self.avg_m_in_s) * unit.avg_s

    def update_avg_l(self, dt_integ):
        """Update the long-term average.

        Called at the end of every trial (*not every cycle*).
        """
        if unit_avg_m > 0.2: # FIXME: 0.2 is a magic number here
            unit.avg_l += self.avg_l_dt * (self.avg_l_max - unit.avg_m)
        else:
            unit.avg_l += self.avg_l_dt * (self.avg_l_min - unit.avg_m)

    def integrate_I_net(self, unit, g_i, dt_integ, ratecoded=True, steps=1):
        """Integrate and returns I_net for the provided v_m

        :param steps:  number of intermediary integration steps.
        """
        assert steps >= 1

        gc_e = self.g_bar_e * unit.g_e
        gc_i = self.g_bar_i * g_i
        gc_l = self.g_bar_l * self.g_l
        v_m_eff = unit.v_m_eq if ratecoded else unit.v_m
        adapt   = 0.0         if ratecoded else unit.adapt

        for _ in range(steps):
            I_net = (  gc_e * (self.e_rev_e - v_m_eff)
                     + gc_i * (self.e_rev_i - v_m_eff)
                     + gc_l * (self.e_rev_l - v_m_eff)
                     - unit.adapt)
            v_m_eff += dt_integ/steps * self.dt_vm * I_net

        return I_net
