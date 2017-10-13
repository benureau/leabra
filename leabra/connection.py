import random

import numpy as np



class Link:
    """A link between two units. Simple, non active class."""

    def __init__(self, pre_unit, post_unit, w0, fw0, index=None):
        """
        Parameters:
            pre_unit   the unit sending its activity
            post_unit  the unit receiving the activity
            w0         the initial weight value
            fw0        initial value of the fast weight parameter
            index      position in the weight matrix
        """
        self.pre  = pre_unit
        self.post = post_unit
        self.wt   = w0
        self.fwt  = fw0
        self.dwt  = 0.0
        self.key  = None


class Connection:
    """Connection between layers"""

    def __init__(self, pre_layer, post_layer, spec=None):
        """
        Parameters:
            pre_layer   the layer sending its activity.
            post_layer  the layer receiving the activity.
        """
        self.pre   = pre_layer
        self.post  = post_layer
        self.links = []
        self.spec  = spec
        if self.spec is None:
            self.spec = ConnectionSpec()

        self.wt_scale_act = 1.0  # scaling relative to activity.
        self.wt_scale_rel_eff = None  # effective relative scaling weight, once other connections
                                      # are taken into account (computed by the network).

        self.spec.projection_init(self)

        pre_layer.from_connections.append(self)
        post_layer.to_connections.append(self)

    @property
    def wt_scale(self):
        try:
            return self.wt_scale_act * self.wt_scale_rel_eff
        except TypeError as e:
            print('Error: did you correctly run the network.build() method?')
            raise e

    @property
    def weights(self):
        """Return a matrix of the links weights"""
        if self.spec.proj.lower() == '1to1':
            return np.array([[link.wt for link in self.links]])
        else:  # proj == 'full'
            W = np.zeros((len(self.pre.units), len(self.post.units)))  # weight matrix
            link_it = iter(self.links)  # link iterator
            for i, pre_u in enumerate(self.pre.units):
                for j, post_u in enumerate(self.post.units):
                    W[i, j] = next(link_it).wt
            return W

    @weights.setter
    def weights(self, value):
        """Override the links weights"""
        if self.spec.proj.lower() == '1to1':
            for wt, link in zip(value, self.links):
                link.wt = wt
        else:  # proj == 'full'
            link_it = iter(self.links)  # link iterator
            for i, pre_u in enumerate(self.pre.units):
                for j, post_u in enumerate(self.post.units):
                    next(link_it).wt = value[i][j]

    def learn(self):
        self.spec.learn(self)

    def cycle(self):
        self.spec.cycle(self)

    def compute_netin_scaling(self):
        self.spec.compute_netin_scaling(self)

class ConnectionSpec:

    legal_proj  = 'full', '1to1'        #              ... for self.proj

    def __init__(self, **kwargs):
        """Connnection parameters"""
        # self.force    = False   # activity are set directly in the post_layer
        self.inhib    = False   # if True, inhibitory connection
        self.proj     = 'full'  # connection pattern between units.
                                # Can be 'Full' or '1to1'. In the latter case,
                                # the layers must have the same size.

        # random initialization
        self.rnd_type = 'uniform' # shape of the weight initialization
        self.rnd_mean = 0.5       # mean of the random variable for weights init.
        self.rnd_var  = 0.25      # variance (or ±range for uniform)

        # learning
        self.lrule    = None    # the learning rule to use (None or 'leabra')
        self.lrate    = 0.01    # learning rate

        # xcal learning
        self.m_lrn    = 1.0     # weighting of the error driven learning

        self.d_thr    = 0.0001  # threshold value for XCAL check-mark function
        self.d_rev    = 0.1     # reversal value for XCAL check-mark function

        self.sig_off  = 1.0
        self.sig_gain = 6.0

        # netin scaling
        self.wt_scale_abs = 1.0  # absolute scaling weight: direct multiplier, strength of the connection
        self.wt_scale_rel = 1.0  # relative scaling weight, relative to other connections.

        for key, value in kwargs.items():
            assert hasattr(self, key) # making sure the parameter exists.
            setattr(self, key, value)

    def cycle(self, connection):
        """Transmit activity."""
        for link in connection.links:
            if link.post.act_ext is None: # activity not forced
                scaled_act = self.wt_scale_abs * connection.wt_scale * link.wt * link.pre.act
                link.post.add_excitatory(scaled_act)

    def _rnd_wt(self):
        """Return a random weight, according to the specified distribution"""
        if self.rnd_type == 'uniform':
            return random.uniform(self.rnd_mean - self.rnd_var,
                                  self.rnd_mean + self.rnd_var)
        elif self.rnd_type == 'gaussian':
            return random.gauss(self.rnd_mean, np.sqrt(self.rnd_var))
        raise NotImplementedError

    def _full_projection(self, connection):
        # creating unit-to-unit links
        connection.links = []
        for i, pre_u in enumerate(connection.pre.units):
            for j, post_u in enumerate(connection.post.units):
                w0 = self._rnd_wt()
                fw0 = self.sig_inv(w0)
                connection.links.append(Link(pre_u, post_u, w0, fw0, index=(i, j)))

    def _1to1_projection(self, connection):
        # creating unit-to-unit links
        connection.links = []
        assert len(connection.pre.units) == len(connection.post.units)
        for i, (pre_u, post_u) in enumerate(zip(connection.pre.units, connection.post.units)):
            w0 = self._rnd_wt()
            fw0 = self.sig_inv(w0)
            connection.links.append(Link(pre_u, post_u, w0, fw0, index=(i, i)))

    def compute_netin_scaling(self, connection):
        """Compute Netin Scaling

        See https://grey.colorado.edu/emergent/index.php/Leabra_Netin_Scaling for details.
        """
        pre_act_avg = connection.pre.avg_act_p_eff
        pre_size = len(connection.pre.units)
        n_links = len(connection.links)

        sem_extra = 2.0 # constant
        pre_act_n = max(1, int(pre_act_avg * pre_size + 0.5)) # estimated number of active units

        if (n_links == pre_size):
            connection.wt_scale_act = 1.0 / pre_act_n
        else:
            post_act_n_max = min(n_links, pre_act_n)
            post_act_n_avg = max(1, pre_act_avg * n_links + 0.5)
            post_act_n_exp = min(post_act_n_max, post_act_n_avg + sem_extra)
            connection.wt_scale_act = 1.0 / post_act_n_exp

    def projection_init(self, connection):
        if self.proj == 'full':
            self._full_projection(connection)
        if self.proj == '1to1':
            self._1to1_projection(connection)


    def learn(self, connection):
        if self.lrule is not None:
            self.learning_rule(connection)
            self.apply_dwt(connection)
        for link in connection.links:
            link.wt = max(0.0, min(1.0, link.wt)) # clipping weights after change

    def apply_dwt(self, connection):

        for link in connection.links:
            link.dwt *= (1 - link.fwt) if (link.dwt > 0) else link.fwt
            # print('before wt={} fwt={}, '.format(link.wt, link.fwt))
            link.fwt += link.dwt
            link.wt = self.sig(link.fwt)
            # print('after wt={} fwt={}'.format(link.wt, link.fwt))

            link.dwt = 0.0

    def learning_rule(self, connection):
        """Leabra learning rule."""

        for link in connection.links:
            srs = link.post.avg_s_eff * link.pre.avg_s_eff
            srm = link.post.avg_m * link.pre.avg_m
            # print('erro', self.m_lrn  * self.xcal(srs, srm))
            # print('hebb', link.post.avg_l_lrn * self.xcal(srs, link.post.avg_l), '  link.post.avg_l_lrn=', link.post.avg_l_lrn)
            link.dwt += (  self.lrate * ( self.m_lrn * self.xcal(srs, srm)
                         + link.post.avg_l_lrn * self.xcal(srs, link.post.avg_l)))

    def xcal(self, x, th):
        if (x < self.d_thr):
            return 0
        elif (x > th * self.d_rev):
            return (x - th)
        else:
            return (-x * ((1 - self.d_rev)/self.d_rev))

    def sig(self, w):
        return 1 / (1 + (self.sig_off * (1 - w) / w) ** self.sig_gain)

    def sig_inv(self, w):
        if   w <= 0.0: return 0.0
        elif w >= 1.0: return 1.0
        return 1 / (1 + ((1 - w) / w) ** (1 / self.sig_gain) / self.sig_off)
