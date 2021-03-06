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

        log_names = ('net', 'I_net', 'v_m', 'act', 'v_m_eq', 'adapt',
                     'avg_ss', 'avg_s', 'avg_s_eff', 'avg_m', 'avg_l')

        def build_network(inhib, fixed_lrn_factor=None):

            if fixed_lrn_factor is not None:

                class UnitSpecFixedLrnFactor(leabra.UnitSpec):

                    def avg_l_lrn(self, unit):
                        return fixed_lrn_factor

                unitspec_class = UnitSpecFixedLrnFactor
            else:
                unitspec_class = leabra.UnitSpec

            u_spec = unitspec_class(act_thr=0.5, act_gain=100, act_sd=0.005,
                                    g_bar_e=1.0, g_bar_i=1.0, g_bar_l=0.1,
                                    e_rev_e=1.0, e_rev_i=0.25, e_rev_l=0.3,
                                    avg_l_min=0.2, avg_l_init=0.4, avg_l_gain=2.5,
                                    adapt_on=False)
            input_layer  = leabra.Layer(1, unit_spec=u_spec, genre=leabra.INPUT, name='input_layer')
            for unit in input_layer.units:
                unit.log_names = log_names
                unit.logs      = {name: [] for name in unit.log_names}
            output_spec  = leabra.LayerSpec(lay_inhib=inhib, g_i=1.8, ff=1.0, fb=1.0, fb_dt=1/1.4, ff0=0.1) #FIXME fb_tau
            output_layer = leabra.Layer(1, spec=output_spec, unit_spec=u_spec,
                                           genre=leabra.OUTPUT, name='output_layer')
            for unit in output_layer.units:
                unit.log_names = log_names
                unit.logs      = {name: [] for name in unit.log_names}
            conspec = leabra.ConnectionSpec(proj='full', lrule='leabra', lrate=0.04,
                                            m_lrn=1.0, rnd_mean=0.5, rnd_var=0.0)
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
            trial_data = data.parse_weights('neuron_pair{}.dat'.format(suffix), flat=True)
            cycle_data = data.parse_unit('neuron_pair{}_cycle.dat'.format(suffix), flat=True)

            network = build_network(inhib)#, fixed_lrn_factor=0.0)
            trial_logs = compute_logs(network)
            #print(logs['wt'])
            cycle_logs = network.layers[-1].units[0].logs
            cycle_logs['gc_i'] = network.layers[-1].logs['gc_i']
            check = quantitative_match(cycle_logs, cycle_data, limit=-1,
                                       rtol=2e-05, atol=1e-08, check=check, verbose=1)
            check = quantitative_match(trial_logs, trial_data,
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
            log_names = ('net', 'I_net', 'v_m', 'act', 'v_m_eq', 'adapt',
                         'avg_ss', 'avg_s', 'avg_s_eff', 'avg_m', 'avg_l')

            u_spec = leabra.UnitSpec(act_thr=0.5, act_gain=100, act_sd=0.005,
                                     g_bar_e=1.0, g_bar_l=0.1, g_bar_i=1.0,
                                     e_rev_e=1.0, e_rev_l=0.3, e_rev_i=0.25,
                                     avg_l_min=0.2, avg_l_init=0.4, avg_l_gain=2.5,
                                     adapt_on=False)

            # layers
            layer_spec = leabra.LayerSpec(lay_inhib=False)
            input_layer  = leabra.Layer(n, spec=layer_spec, unit_spec=u_spec, genre=leabra.INPUT, name='input_layer')
            hidden_layer = leabra.Layer(n, spec=layer_spec, unit_spec=u_spec, genre=leabra.HIDDEN, name='hidden_layer')
            output_layer = leabra.Layer(n, spec=layer_spec, unit_spec=u_spec, genre=leabra.OUTPUT, name='output_layer')
            for layer in [input_layer, hidden_layer, output_layer]:
                for unit in layer.units:
                    unit.log_names = log_names
                    unit.logs      = {name: [] for name in unit.log_names}

            # connections
            weights = read_weights(os.path.join(os.path.dirname(__file__), 'emergent_projects/leabra_std{}.wts'.format(n)))
            inphid_conn_spec = leabra.ConnectionSpec(proj='full', lrule='leabra', lrate=0.04, rnd_mean=0.5, rnd_var=0.0,
                                                     wt_scale_abs=1.0, wt_scale_rel=1.0)
            hidout_conn_spec = leabra.ConnectionSpec(proj='full', lrule='leabra', lrate=0.04, rnd_mean=0.5, rnd_var=0.0,
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
            input_logs  = {'net': [], 'act': [], 'I_net': [], 'v_m': [], 'avg_s': [], 'avg_s_eff': [], 'avg_m': [], 'avg_l': []}
            hidden_logs = {'net': [], 'act': [], 'I_net': [], 'v_m': [], 'avg_s': [], 'avg_s_eff': [], 'avg_m': [], 'avg_l': []}
            output_logs = {'net': [], 'act': [], 'I_net': [], 'v_m': [], 'avg_s': [], 'avg_s_eff': [], 'avg_m': [], 'avg_l': []}
            input_layer, hidden_layer, output_layer = network.layers
            trial_logs = {'hidden_wts': [], 'output_wts': []}
            for t in range(5):
                trial_logs['hidden_wts'].append(network.connections[0].weights)
                trial_logs['output_wts'].append(network.connections[1].weights)
                network.trial()

            for i in range(network.cycle_tot):
                for logs, layer in [(input_logs, input_layer), (hidden_logs, hidden_layer), (output_logs, output_layer)]:
                    logs['net'].append(np.array(  [u.logs['net'][i]   for u in layer.units]))
                    logs['act'].append(np.array(  [u.logs['act'][i]   for u in layer.units]))
                    logs['I_net'].append(np.array([u.logs['I_net'][i] for u in layer.units]))
                    logs['v_m'].append(np.array(  [u.logs['v_m'][i]   for u in layer.units]))
                    logs['avg_s'].append(np.array([u.logs['avg_s'][i] for u in layer.units]))
                    logs['avg_s_eff'].append(np.array([u.logs['avg_s_eff'][i] for u in layer.units]))
                    logs['avg_m'].append(np.array([u.logs['avg_m'][i] for u in layer.units]))
                    logs['avg_l'].append(np.array([u.logs['avg_l'][i] for u in layer.units]))
            return input_logs, hidden_logs, output_logs, trial_logs

        def postprocessing(py, em):
            return py[:4], em

        check = True
        for n in [4]:
            trial_data = data.parse_file('leabra_std{}_trial.dat'.format(n))

            cycle_input_data = data.parse_unit('leabra_std{}_cycle_input.dat'.format(n))
            cycle_hidden_data = data.parse_unit('leabra_std{}_cycle_hidden.dat'.format(n))
            cycle_output_data = data.parse_unit('leabra_std{}_cycle_output.dat'.format(n))
            network = build_network(n)

            # wt for wt_balancing
            # for i in range(4):
            #     wts = [network.connections[0].weights[j][i] for j in range(4)]
            #     print(np.mean(wts), wts)
            #
            # for i in range(4):
            #     wts = [network.connections[1].weights[j][i] for j in range(4)]
            #     print(np.mean(wts), wts)

            input_logs, hidden_logs, output_logs, trial_logs = compute_logs(network)
            # print([u.act_m for u in network.layers[2].units])

            for title, logs, em_data in [('input',  input_logs,  cycle_input_data),
                                         ('hidden', hidden_logs, cycle_hidden_data),
                                         ('output', output_logs, cycle_output_data)]:
                check = quantitative_match(logs, em_data, limit=-1, rtol=2e-05, atol=2e-07,
                                           check=check, verbose=1, postprocessing=postprocessing,
                                           prefix='{}:{} '.format(title, n))
            check = quantitative_match(trial_logs, trial_data, limit=-1, rtol=2e-05, atol=2e-07,
                                       check=check, verbose=1, prefix='{}:{} '.format('trial', n))

        self.assertTrue(check)


if __name__ == '__main__':
    unittest.main()
