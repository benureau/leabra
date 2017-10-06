import unittest
import os

import numpy as np

import data

import dotdot  # pylint: disable=unused-import
import leabra

from read_weight_file import read_weights
from utils import quantitative_match


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

        def build_network(inhib):
            u_spec = leabra.UnitSpec(act_thr=0.5, act_gain=100, act_sd=0.005,
                                     g_bar_e=1.0, g_bar_i=1.0, g_bar_l=0.1,
                                     e_rev_e=1.0, e_rev_i=0.25, e_rev_l=0.3,
                                     avg_l_min=0.2, avg_l_init=0.4, avg_l_gain=2.5,
                                     adapt_on=False)
            input_layer  = leabra.Layer(1, unit_spec=u_spec, name='input_layer')
            g_i = 1.5 if inhib else 0.0
            output_spec  = leabra.LayerSpec(g_i=g_i, ff=0.0, fb=0.5, fb_dt=1/1.4, ff0=0.1)
            output_layer = leabra.Layer(1, spec=output_spec, unit_spec=u_spec, name='output_layer')
            for u in output_layer.units:
                u.avg_l_lrn = 1.0
            conspec = leabra.ConnectionSpec(proj='full', lrule='leabra', lrate=0.04,
                                            m_lrn=0.0, rnd_mean=0.5, rnd_var=0.0)
            conn    = leabra.Connection(input_layer, output_layer, spec=conspec)

            network = leabra.Network(layers=[input_layer, output_layer], connections=[conn])
            network.set_inputs({'input_layer': [0.95]})
            network.set_outputs({'output_layer': [0.95]})

            return network

        def compute_logs(network):
            logs = {'wt': [], 'sse': [], 'output_act_m': []}
            for t in range(50):
                logs['wt'].append(network.connections[0].links[0].wt)
                sse = network.trial()
                logs['sse'].append(sse)
                logs['output_act_m'].append(network.layers[-1].units[0].act_m)
            return logs

        for inhib in [True, False]:
            suffix = '_inhib' if inhib else ''
            emergent_data = data.parse_weights('neuron_pair{}.dat'.format(suffix))
            cycle_data    = data.parse_unit('neuron_pair{}_cycle.dat'.format(suffix))

            network = build_network(inhib)
            logs = compute_logs(network)
            print(logs['wt'])
            cycle_logs = network.layers[-1].units[0].logs
            check = quantitative_match(cycle_logs, cycle_data, limit=100,
                                       rtol=2e-05, atol=0, check=check, verbose=1)

        self.assertTrue(check)


    def test_netin_scaling(self):
        """Quantitative test on the netin scaling scenario"""
        cycle_data = data.parse_unit('netin.dat')

        def build_network():
            u_spec = leabra.UnitSpec(act_thr=0.5, act_gain=100, act_sd=0.005,
                                     g_bar_e=1.0, g_bar_l=0.1, g_bar_i=1.0,
                                     e_rev_e=1.0, e_rev_l=0.3, e_rev_i=0.25,
                                     avg_l_min=0.2, avg_l_init=0.4, avg_l_gain=2.5,
                                     adapt_on=False)
            # layers
            input0_layer  = leabra.Layer(4, unit_spec=u_spec, name='input0_layer')
            input1_layer  = leabra.Layer(4, unit_spec=u_spec, name='input1_layer')
            output_spec  = leabra.LayerSpec(lay_inhib=False)
            output_layer = leabra.Layer(4, spec=output_spec, unit_spec=u_spec, name='output_layer')
            # connections
            conn_spec0 = leabra.ConnectionSpec(proj='1to1', lrule=None, rnd_mean=0.5, rnd_var=0.0,
                                               wt_scale_abs=1.0, wt_scale_rel=1.0)
            conn_spec1 = leabra.ConnectionSpec(proj='1to1', lrule=None, rnd_mean=0.5, rnd_var=0.0,
                                               wt_scale_abs=2.0, wt_scale_rel=1.0)
            conn0 = leabra.Connection(input0_layer, output_layer, spec=conn_spec0)
            conn1 = leabra.Connection(input1_layer, output_layer, spec=conn_spec1)
            # network
            network = leabra.Network(layers=[input0_layer, input1_layer, output_layer],
                                     connections=[conn0, conn1])
            network.set_inputs({'input0_layer': [0.5, 0.95, 0.0, 0.25],
                                'input1_layer': [0.0, 0.5, 0.95, 0.75]})
            return network

        def compute_logs(network):
            logs = {'net': [], 'act': []}
            output_layer = network.layers[2]
            for t in range(200):
                network.cycle()
                logs['net'].append(np.array([u.net for u in output_layer.units]))
                logs['act'].append(np.array([u.act for u in output_layer.units]))
            return logs

        network = build_network()
        logs = compute_logs(network)

        self.assertTrue(quantitative_match(logs, cycle_data, rtol=2e-05, atol=0))


    def test_std_project(self):
        """Quantitative test on the template Leabra project"""

        def build_network(n):
            u_spec = leabra.UnitSpec(act_thr=0.5, act_gain=100, act_sd=0.005,
                                     g_bar_e=1.0, g_bar_l=0.1, g_bar_i=1.0,
                                     e_rev_e=1.0, e_rev_l=0.3, e_rev_i=0.25,
                                     avg_l_min=0.2, avg_l_init=0.4, avg_l_gain=2.5,
                                     adapt_on=False)

            # layers
            layer_spec = leabra.LayerSpec(lay_inhib=False)
            input_layer  = leabra.Layer(n, spec=layer_spec, unit_spec=u_spec, name='input_layer')
            hidden_layer = leabra.Layer(n, spec=layer_spec, unit_spec=u_spec, name='hidden_layer')
            output_layer = leabra.Layer(n, spec=layer_spec, unit_spec=u_spec, name='output_layer')

            # connections
            weights = read_weights(os.path.join(os.path.dirname(__file__), 'emergent_projects/leabra_std{}.wts'.format(n)))
            inphid_conn_spec = leabra.ConnectionSpec(proj='full', lrule=None, rnd_mean=0.5, rnd_var=0.0,
                                                     wt_scale_abs=1.0, wt_scale_rel=1.0)
            hidout_conn_spec = leabra.ConnectionSpec(proj='full', lrule=None, rnd_mean=0.5, rnd_var=0.0,
                                                     wt_scale_abs=1.0, wt_scale_rel=1.0)
            inphid_conn = leabra.Connection(input_layer,  hidden_layer, spec=inphid_conn_spec)
            inphid_conn.weights = weights[('Input', 'Hidden')]
            hidout_conn = leabra.Connection(hidden_layer, output_layer, spec=hidout_conn_spec)
            hidout_conn.weights = weights[('Hidden', 'Output')]

            # network
            network = leabra.Network(layers=[input_layer, hidden_layer, output_layer],
                                     connections=[inphid_conn, hidout_conn])
            n_sqrt = int(round(np.sqrt(n)))
            network.set_inputs ({'input_layer' : [0.95]*n_sqrt + [0.0]*(n-n_sqrt)})
            network.set_outputs({'output_layer': [0.0]*(n-n_sqrt) + [0.95]*n_sqrt}) # FIXME 0.95 -> 1.0

            return network

        def compute_logs(network):
            logs = {'net': [], 'act': [], 'I_net': [], 'v_m': []}
            hidden_layer = network.layers[1]
            for t in range(200):
                network.cycle()
                logs['net'].append(np.array([u.net for u in hidden_layer.units]))
                logs['act'].append(np.array([u.act for u in hidden_layer.units]))
                logs['I_net'].append(np.array([u.I_net for u in hidden_layer.units]))
                logs['v_m'].append(np.array([u.v_m for u in hidden_layer.units]))
            return logs



        check = True
        for n in [4]:
            table_data = data.parse_unit('leabra_std{}_cycle.dat'.format(n))
            network = build_network(n)
            logs = compute_logs(network)
            for name in logs.keys():
                for t, (py, em) in enumerate(list(zip(logs[name], table_data[name]))[:-1]):
                    py = py[:4]
                    if not np.allclose(py, em, rtol=2e-05, atol=0):
                        print('{}:{:2d} [py] {} != {} [em] diff={} (n={})'.format(
                               name, t,      py,  em,        py-em,   n))
                        check = False

        self.assertTrue(check)


if __name__ == '__main__':
    unittest.main()
