
class NetworkSpec:
    """Network parameters"""

    def __init__(self, **kwargs):
        self.quarter_size = 25  # number of cycles in a settle period

        for key, value in kwargs.items():
            assert hasattr(self, key) # making sure the parameter exists.
            setattr(self, key, value)


class Network:
    """Leabra Network class"""

    def __init__(self, spec=None, layers=(), connections=()):
        self.spec = spec
        if self.spec is None:
            self.spec = NetworkSpec()

        self.cycle_count = 0 # number of cycles finished in the current trial
        self.cycle_tot   = 0 # total number of cycles executed (not reset at end of trial)
        self.quarter_nb  = 1 # current quarter number (1, 2, 3 or 4)
        self.trial_count = 0 # number of trial finished
        self.phase       = 'minus'

        self.layers      = list(layers)
        self.connections = list(connections)

        self._inputs, self._outputs = {}, {}
        self.build()

    def add_connection(self, connection):
        self.connections.append(connection)
        self.build()

    def add_layer(self, layer):
        self.layers.append(layer)

    def build(self):
        """Precompute necessary network datastructures.

        This needs to be run every time a layer or connection is added or removed from the network,
        or if the value of a connection's `wt_scale_rel` is changed. This automatically run when
        using the `add_connection()` method.
        """
        for layer in self.layers:
            rel_sum = sum(connection.spec.wt_scale_rel for connection in layer.to_connections)
            for connection in layer.to_connections:
                connection.wt_scale_rel_eff = connection.spec.wt_scale_rel / rel_sum

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


    def _pre_cycle(self):
        """Check if some action needs to be done before starting the cycle.

        Checks if the network is at a special moment (beginning of trial, start
        or end of phase, etc.), and if some action needs to be done.
        """
        if self.cycle_count == self.spec.quarter_size: # a quarter just ended
            self.quarter_nb += 1
            if self.quarter_nb == 5: # a trial just ended
                self.trial_count += 1
                self.quarter_nb = 1
            self.cycle_count = 0

        if self.cycle_count == 0: # start of a quarter
            for connection in self.connections:
               connection.compute_netin_scaling()

            if self.quarter_nb == 1: # start of trial
                # reset all layers
                if self.quarter_nb == 1:
                    for layer in self.layers:
                        layer.trial_init()
                # force activities for inputs
                for name, activities in self._inputs.items():
                    self._get_layer(name).force_activity(activities)

            elif self.quarter_nb == 4: # start of plus phase
                # force activities for outputs
                for name, activities in self._outputs.items():
                    self._get_layer(name).force_activity(activities)


    def _post_cycle(self):
        """Same as _pre_cycle, but after the cycle has executed"""
        if self.cycle_count == self.spec.quarter_size: # end of a quarter
            if self.quarter_nb == 3: # end of minus phase
                self.end_minus_phase()

            if self.quarter_nb == 4: # end of plus phase
                self.end_plus_phase()


    def cycle(self):
        """Execute a cycle"""
        self._pre_cycle()

        for conn in self.connections:
            conn.cycle()
        for layer in self.layers:
            layer.cycle(self.phase)
        self.cycle_count += 1
        self.cycle_tot   += 1

        self._post_cycle()


    def quarter(self): # FIXME:
        """Execute a quarter"""
        self.cycle()
        while self.cycle_count < self.spec.quarter_size:
            self.cycle()


    def trial(self):
        """Execute a trial. Will execute up until the end of the plus phase."""
        self.quarter()
        while self.quarter_nb != 4:
            assert self.cycle_count == self.spec.quarter_size
            self.quarter()
        return self.compute_sse()

    def compute_sse(self):
        """Compute the sum of squared error in prediction (SSE).

        Should be run only after the minus phase is finished.
        """
        sse = 0
        for name, activities in self._outputs.items():
            for act, unit in zip(activities, self._get_layer(name).units):
                sse += (act - unit.act_m)**2
        return sse

    def end_minus_phase(self):
        """End of the minus phase. Current unit activity is stored."""
        for layer in self.layers:
            for unit in layer.units:
                unit.act_m = unit.act
        self.phase = 'plus'

    def end_plus_phase(self):
        """End of the plus phase. Connections change weights."""
        for conn in self.connections:
            conn.learn()
        for layer in self.layers:
            for unit in layer.units:
                unit.update_avg_l()

        self.phase = 'minus'
