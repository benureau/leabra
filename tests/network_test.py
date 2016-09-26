import unittest

import numpy as np

import data

import dotdot  # pylint: disable=unused-import
import leabra



class NetworkTestAPI(unittest.TestCase):

    def test_simple_usage(self):
        """Test the basic Network API"""
        input_layer  = leabra.Layer(4, name='input_layer')
        output_spec  = leabra.LayerSpec(g_i=1.5, ff=1, fb=0.5, fb_dt=1/1.4, ff0=0.1)
        output_layer = leabra.Layer(2, spec=output_spec, name='output_layer')

        conspec = leabra.ConnectionSpec(proj="full", lrule='leabra')
        conn    = leabra.Connection(input_layer, output_layer, spec=conspec)

        network = leabra.Network(layers=[input_layer, output_layer], connections=[conn])

        network.set_inputs({'input_layer': [1.0, 1.0, 0.0, 0.0]})
        network.set_outputs({'output_layer': [1.0, 0.0]})

        for _ in range(20):
            network.trial()

        self.assertTrue(True)


class NetworkTestBehavior(unittest.TestCase):
    """Check that the Network behaves as it should.

    The checks are concerned with consistency with the equations that define the
    network's and overall learning behavior. These test should pass on the
    emergent implementation as well.
    """

    def test_simple_pattern_learning(self):
        """Quantitative test on the pair of neurons scenario"""
        check = True

        for inhib in [False, True]:
            if inhib:
                emergent_data = data.parse_weights('neuron_pair_inhib.txt')
            else:
                emergent_data = data.parse_weights('neuron_pair.txt')

            u_spec = leabra.UnitSpec(act_thr=0.5, act_gain=100, act_sd=0.01,
                                     g_bar_e=1.0, g_bar_i=1.0, g_bar_l=0.1,
                                     e_rev_e=1.0, e_rev_i=0.25, e_rev_l=0.3,
                                     avg_l_min=0.2, avg_l_init=0.155, avg_l_max=1.5,
                                     adapt_on=False)
            input_layer  = leabra.Layer(1, unit_spec=u_spec, name='input_layer')
            g_i = 1.5 if inhib else 0.0
            output_spec  = leabra.LayerSpec(g_i=g_i, ff=1.0, fb=0.5, fb_dt=1/1.4, ff0=0.1)
            output_layer = leabra.Layer(1, spec=output_spec, unit_spec=u_spec, name='output_layer')
            for u in output_layer.units:
                u.avg_l_lrn = 1.0
            conspec = leabra.ConnectionSpec(proj='full', lrule='leabra', lrate=0.04,
                                            m_lrn=0.0, rnd_mean=0.5, rnd_var=0.0)
            conn    = leabra.Connection(input_layer, output_layer, spec=conspec)

            network = leabra.Network(layers=[input_layer, output_layer], connections=[conn])
            network.set_inputs({'input_layer': [0.95]})
            network.set_outputs({'output_layer': [0.95]})

            logs = {'wt': [], 'sse': []}
            for t in range(50):
                logs['wt'].append(conn.links[0].wt)
                sse = network.trial()
                logs['sse'].append(sse)

            for name in ['wt', 'sse']:
                for t, (py, em) in enumerate(zip(logs[name], emergent_data[name])):
                    if not np.allclose(py, em, rtol=0, atol=1e-05):
                        print('{}:{:2d} [py] {:.10f} != {:.10f} [emergent] ({}inhib) diff={:g}'.format(
                                name, t,   py,        em, '' if inhib else 'no ', py-em))
                        check = False

        self.assertTrue(check)


if __name__ == '__main__':
    unittest.main()
