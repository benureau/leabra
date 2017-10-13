"""An example of a standard input/output learning network, with a variable number of hidden layers"""

import leabra


def build_network(n_input, n_output, n_hidden):

    # specifications
    unit_spec  = leabra.UnitSpec(adapt_on=True, noisy_act=True)
    inpout_layer_spec = leabra.LayerSpec(lay_inhib=True, g_i=2.0, ff=1, fb=0.5)
    hidden_layer_spec = leabra.LayerSpec(lay_inhib=True, g_i=1.8, ff=1, fb=1)
    conn_spec  = leabra.ConnectionSpec(proj='full', lrule='leabra', lrate=0.04,
                                       rnd_type='uniform',  rnd_mean=0.50, rnd_var=0.25)

    # input/outputs
    input_layer  = leabra.Layer(n_input,  spec=inpout_layer_spec, unit_spec=unit_spec, genre=leabra.INPUT,  name='input_layer')
    output_layer = leabra.Layer(n_output, spec=inpout_layer_spec, unit_spec=unit_spec, genre=leabra.OUTPUT, name='output_layer')

    # creating the required numbers of hidden layers and connections
    layers = [input_layer]
    connections = []
    for i in range(n_hidden):
        hidden_layer = leabra.Layer(n_input, spec=hidden_layer_spec, unit_spec=unit_spec,
                                             genre=leabra.HIDDEN, name='hidden_layer_{}'.format(i))
        hidden_conn  = leabra.Connection(layers[-1],  hidden_layer, spec=conn_spec)
        layers.append(hidden_layer)
        connections.append(hidden_conn)

    last_conn  = leabra.Connection(layers[-1],  output_layer, spec=conn_spec)
    connections.append(last_conn)
    layers.append(output_layer)

    network_spec = leabra.NetworkSpec(quarter_size=25)
    network = leabra.Network(layers=layers, connections=connections)

    return network

def test_network(network, input_pattern):
    assert len(network.layers[0].units) == len(input_pattern)
    network.set_inputs({'input_layer': input_pattern})

    network.trial()
    return [unit.act_m for unit in network.layers[-1].units]

def train_network(network, input_pattern, output_pattern):
    """Run one trial on the network"""
    assert len(network.layers[0].units) == len(input_pattern)

    assert len(network.layers[-1].units) == len(output_pattern)
    network.set_inputs({'input_layer': input_pattern})
    network.set_outputs({'output_layer': output_pattern})

    sse = network.trial()
    print('{} sse={}'.format(network.trial_count, sse))
    return [unit.act_m for unit in network.layers[-1].units]


if __name__ == '__main__':

    network = build_network(4, 4, 1)
    print(test_network(network, [1.0, 1.0, 0.0, 0.0]))

    for i in range(500):
        train_network(network, [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0])

    print(test_network(network, [1.0, 1.0, 0.0, 0.0]))
    #
    # network = build_network(25, 25, 2)
    # horizontal = 10*[0.0] + 5*[1.0] + 10*[0.0]
    # vertical   = 5*[0.0, 0.0, 1.0, 0.0, 0.0]
    # leftdiag   = [1.0, 0.0, 0.0, 0.0, 0.0,
    #               0.0, 1.0, 0.0, 0.0, 0.0,
    #               0.0, 0.0, 1.0, 0.0, 0.0,
    #               0.0, 0.0, 0.0, 1.0, 0.0,
    #               0.0, 0.0, 0.0, 0.0, 1.0]
    # rightdiag  = [0.0, 0.0, 0.0, 0.0, 1.0,
    #               0.0, 0.0, 0.0, 1.0, 0.0,
    #               0.0, 0.0, 1.0, 0.0, 0.0,
    #               0.0, 1.0, 0.0, 0.0, 0.0,
    #               1.0, 0.0, 0.0, 0.0, 0.0]
    #
    # for i in range(50):
    #     train_network(network, horizontal, horizontal)
    #     train_network(network, vertical, vertical)
    #     train_network(network, leftdiag, leftdiag)
    #     train_network(network, rightdiag, rightdiag)
    #
    # print(test_network(network, horizontal))
