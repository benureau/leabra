import random

import numpy as np



class Link:
    """A link between two units. Simple, non active class."""

    def __init__(self, pre_unit, post_unit, w0, fw0):
        self.pre  = pre_unit
        self.post = post_unit
        self.wt   = w0
        self.fwt  = fw0
        self.dwt  = 0.0


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

        self.spec.projection_init(self)

        post_layer.connections.append(self)

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

    def learn(self):
        self.spec.learn(self)

    def cycle(self):
        self.spec.cycle(self)



class ConnectionSpec:

    legal_proj  = 'full', '1to1'        #              ... for self.proj

    def __init__(self, **kwargs):
        """Connnection parameters"""
        self.st       = 1.0     # connection strength
        # self.force    = False   # activity are set directly in the post_layer
        self.inhib    = False   # if True, inhibitory connection
        self.proj     = 'full'  # connection pattern between units.
                                # Can be 'Full' or '1to1'. In the latter case,
                                # the layers must have the same size.
        self.lrule    = None    # the learning rule to use (None or 'leabra')

        # random initialization
        self.rnd_type = 'uniform' # shape of the weight initialization
        self.rnd_mean = 0.5       # mean of the random variable for weights init.
        self.rnd_var  = 0.25      # variance (or ±range for uniform)

        self.lrate    = 0.01    # learning rate

        self.m_lrn    = 1.0     # weighting of the error driven learning

        self.d_thr    = 0.0001
        self.d_rev    = 0.1

        self.sig_off  = 1.0
        self.sig_gain = 6.0

        for key, value in kwargs.items():
            assert hasattr(self, key) # making sure the parameter exists.
            setattr(self, key, value)

    def cycle(self, connection):
        """Transmit activity."""
        for link in connection.links:
            if not link.post.forced:
                #print(self.st, link.wt, link.pre.act)
                scaled_act = self.st * link.wt * link.pre.act
                link.post.add_excitatory(scaled_act)

    def _rnd_wt(self):
        """Return a random weight, according to the specified distribution"""
        if self.rnd_type == 'uniform':
            return random.uniform(self.rnd_mean - self.rnd_var,
                                  self.rnd_mean + self.rnd_var)
        raise NotImplementedError

    def _full_projection(self, connection):
        # creating unit-to-unit links
        connection.links = []
        for pre_u in connection.pre.units:
            for post_u in connection.post.units:
                w0 = self._rnd_wt()
                fw0 = self.sig_inv(w0)
                connection.links.append(Link(pre_u, post_u, w0, fw0))


    def _1to1_projection(self, connection):
        # creating unit-to-unit links
        connection.links = []
        assert connection.pre.size == connection.post.size
        for pre_u, post_u in zip(connection.pre.units, connection.post.units):
            w0 = self._rnd_wt()
            fw0 = self.sig_inv(w0)
            connection.links.append(Link(pre_u, post_u, w0, fw0))


    def projection_init(self, connection):
        if self.proj == 'full':
            self._full_projection(connection)
        if self.proj == '1to1':
            self._1to1_projection(connection)


    def learn(self, connection):
        if self.learn is not None:
            self.learning_rule(connection)
            self.apply_dwt(connection)
        for link in connection.links:
            link.wt = max(0.0, min(1.0, link.wt)) # clipping weights after change

    def apply_dwt(self,connection):

        for link in connection.links:
            link.dwt *= (1 - link.fwt) if (link.dwt > 0) else link.fwt
            #print('dwt', link.dwt)
            link.fwt += link.dwt
            #print('fwt', link.fwt, 'wt', self.sig(link.fwt))
            link.wt = self.sig(link.fwt)

            link.dwt = 0.0

    def learning_rule(self, connection):
        """Leabra learning rule."""

        for link in connection.links:
            srs = link.post.avg_s_eff * link.pre.avg_s_eff
            srm = link.post.avg_m * link.pre.avg_m
            #print('srs = {:.6f} [avg_s_eff {:.6f} {:.6f}'.format(srs, link.pre.avg_s_eff, link.post.avg_s_eff))
            #print('srm = {:.6f} [avg_m     {:.6f} {:.6f}'.format(srm, link.pre.avg_m, link.post.avg_m))
            #print('diff = {:.8f}'.format(srm-srs))
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
        return 1 / (1 + ((1 - w) / w) ** (1 / self.sig_gain) / self.sig_off)
