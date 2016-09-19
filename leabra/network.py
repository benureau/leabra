
class NetworkSpec:
    """Network parameters"""

    def __init__(self, **kwargs):
        # time step constants
        self.quarter_size = 15  # number of cycles in a settle period

        for key, value in kwargs.items():
            assert hasattr(self, key) # making sure the parameter exists.
            setattr(self, key, value)


class Network:
    """Leabra Network class"""

    def __init__(self, spec=None, layers=(), connections=()):
        self.spec = spec
        if self.spec is None:
            self.spec = NetworkSpec()

        self.quarter_nb  = 1 # next quarter to execute (1, 2, 3 or 4)
        self.cycles      = 0 # how many cycles happened (total)
        self.layers      = list(layers)
        self.connections = list(connections)

        self._inputs, self._outputs = {}, {}


    def add_connection(self, connection):
        self.connections.append(connection)

    def add_layer(self, layer):
        self.layers.append(layer)


    def _get_layer(self, name):
        """Get a layer from its name.

        If layers share the name, return the first one added to the network.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError("layer '{}' not found.".format(name))

    def set_inputs(self, act_map):
        """Set inputs activities, set at the beginning of all quarters.

        :param act_map:  a dict with layer names as keys, and activities arrays
                         as values.
        """
        self._inputs = act_map

    def set_outputs(self, act_map):
        """Set inputs activities, set at the beginning of all quarters.

        :param act_map:  a dict with layer names as keys, and activities arrays
                         as values
        """
        self._outputs = act_map

    def cycle(self):
        """Execute a quarter"""
        for conn in self.connections:
            conn.cycle()
        for layer in self.layers:
            layer.cycle()
        self.cycles += 1


    def quarter(self): # FIXME:
        """Execute a quarter"""

        # reset all layers
        if self.quarter_nb == 1:
            for layer in self.layers:
                layer.reset()

        # force activities for inputs and outputs
        for name, activities in self._inputs.items():
            self._get_layer(name).force_activity(activities)
        if self.quarter_nb == 4:
            for name, activities in self._outputs.items():
                self._get_layer(name).force_activity(activities)

        # cycle the network
        for t in range(self.spec.quarter_size):
            self.cycle()

        # end of minus or minus phase
        if self.quarter_nb == 3:
            self.end_minus_phase()

        if self.quarter_nb == 4:
            self.end_plus_phase()
            self.quarter_nb = 0

        self.quarter_nb += 1


    def trial(self):
        """Execute a trial. Will execute up until the end of the plus phase."""
        self.quarter()
        while self.quarter_nb != 1:
            self.quarter()

    def end_minus_phase(self):
        """End of the minus phase. Current unit activity is stored."""
        for layer in self.layers:
            for unit in layer.units:
                unit.act_m = unit.act

    def end_plus_phase(self):
        """End of the plus phase. Connections change weights."""
        for conn in self.connections:
            conn.learn()
