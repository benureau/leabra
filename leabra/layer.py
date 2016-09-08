from .unit import Unit


class Layer:
    """Leabra Layer class"""

    def __init__(self, size, spec=None, unit_spec=None):
        """
        size     :  Number of units in the layer.
        spec     :  LayerSpec instance with custom values for the parameter of
                    the layer. If None, default values will be used.
        unit_spec:  UnitSpec instance with custom values for the parameters of
                    the units of the layer. If None, default values will be used.
        """
        self.spec = spec
        if self.spec is None:
            self.spec = LayerSpec()
        #!#assert self.spec.inhib.lower() in self.spec.legal_inhib

        self.size = size
        self.units = [Unit(spec=unit_spec) for _ in range(self.size)]

        self.g_i = 0.0
        self.fbi = 0.0

        self.connections = []

    @property
    def activities(self):
        """Return the matrix of the units's activities"""
        return [u.act for u in self.units]

    def set_activities(self, inputs):
        """Set the units's activities equal to the inputs"""
        assert len(inputs) == self.size
        for u, inp in zip(self.units, inputs):
            u.act = inp

    
    def add_excitatory(self, inputs, forced_act=False):
        assert len(inputs) == self.size
        for u, net_raw in zip(self.units, inputs):
            u.add_excitatory(net_raw, forced_act=forced_act)

    def cycle(self):
        self.spec.cycle(self)




class LayerSpec:
    """Layer parameters"""

    #!#legal_inhib = 'kwta', 'kwta_avg'  # available values for self.inhib

    def __init__(self, **kwargs):
        #!#self.inhib    = 'kwta'   # inhibition rule: 'kwta' or 'kwta_avg'
        #!#self.k        = None     # number of active units.
        #!#self.kwta_pct = 0.25     # proportion of active units; used to compute k
                                 # only if self.k is None.
        #!#self.q        = 0.25     # see eq. A6
        #

        # time step constants:
        self.fb_dt = 1/1.4          # Integration constant for feed back inhibition
        
        # weighting constants
        self.fb = 1                 # feedback scaling of inhibition

        # thresholds:
        slef.ff0 = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _inhibition(self, layer):
        """Compute inhibition"""

        
        return 0.0

    
    def cycle(self, layer):
        """Calculate net inputs for this layer"""
        for connection in layer.connections:
            connection.cycle()

        for u in layer.units:
            u.calculate_net_in()

        """Update the state of the layer"""
        self.g_i = self._inhibition(layer)
        for u in layer.units:
            u.cycle(g_i=self.g_i)
