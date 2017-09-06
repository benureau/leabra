from .unit import Unit
import statistics


class Layer:
    """Leabra Layer class"""

    def __init__(self, size, spec=None, unit_spec=None, name=None):
        """
        size     :  Number of units in the layer.
        spec     :  LayerSpec instance with custom values for the parameter of
                    the layer. If None, default values will be used.
        unit_spec:  UnitSpec instance with custom values for the parameters of
                    the units of the layer. If None, default values will be used.
        """
        self.name = name
        self.spec = spec
        if self.spec is None:
            self.spec = LayerSpec()
        #!#assert self.spec.inhib.lower() in self.spec.legal_inhib

        self.size = size
        self.units = [Unit(spec=unit_spec) for _ in range(self.size)]

        self.gc_i = 0.0  # inhibitory conductance
        self.ffi  = 0.0  # feedforward component of inhibition
        self.fbi  = 0.0  # feedback component of inhibition

        self.connections = []

    def reset(self):
        """Reset all the units in the layer"""
        for u in self.units:
            u.reset()

    @property
    def activities(self):
        """Return the matrix of the units's activities"""
        return [u.act for u in self.units]

    @property
    def g_e(self):
        """Return the matrix of the units's net exitatory input"""
        return [u.g_e for u in self.units]

    def force_activity(self, activities):
        """Set the units's activities equal to the inputs."""
        assert len(activities) == self.size
        for u, act in zip(self.units, activities):
            u.force_activity(act)

    def add_excitatory(self, inputs):
        """Add excitatory inputs to the layer's units."""
        assert len(inputs) == self.size
        for u, net_raw in zip(self.units, inputs):
            u.add_excitatory(net_raw)

    def cycle(self):
        self.spec.cycle(self)

    def show_config(self):
        """Display the value of constants and state variables."""
        print('Parameters:')
        for name in ['fb_dt', 'ff0', 'ff', 'fb', 'g_i']:
            print('   {}: {:.2f}'.format(name, getattr(self.spec, name)))
        print('State:')
        for name in ['gc_i', 'fbi', 'ffi']:
            print('   {}: {:.2f}'.format(name, getattr(self, name)))




class LayerSpec:
    """Layer parameters"""

    def __init__(self, **kwargs):
        """Initialize a LayerSpec"""
        self.lay_inhib = True # activate inhibition?

        # time step constants:
        self.fb_dt = 1/1.4  # Integration constant for feed back inhibition

        # weighting constants
        self.fb    = 1.0    # feedback scaling of inhibition
        self.ff    = 1.0    # feedforward scaling of inhibition
        self.g_i   = 1.8    # inhibition multiplier

        # thresholds:
        self.ff0 = 0.1

        for key, value in kwargs.items():
            assert hasattr(self, key) # making sure the parameter exists.
            setattr(self, key, value)

    def _inhibition(self, layer):
        """Compute the layer inhibition"""
        if self.lay_inhib:
            # Calculate feed forward inhibition
            netin = [u.g_e for u in layer.units]
            layer.ffi = self.ff * max(0, statistics.mean(netin) - self.ff0)

            # Calculate feed back inhibition
            layer.fbi += self.fb_dt * (self.fb * statistics.mean(layer.activities) - layer.fbi)

            return self.g_i * (layer.ffi + layer.fbi)
        else:
            return 0.0

    def cycle(self, layer):
        """Cycle the layer, and all the units in it."""
        # calculate net inputs for this layer
        for u in layer.units:
            u.calculate_net_in()

        # update the state of the layer
        layer.gc_i = self._inhibition(layer)
        for u in layer.units:
            u.cycle(g_i=layer.gc_i)
