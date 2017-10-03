"""An example of a standard input/output learning network, with a variable number of hidden layers"""

import leabra


def build_network(n_input, n_output, n_hidden):

    # specifications
    unit_spec  = leabra.UnitSpec(adapt_on=True, noisy_act=True)
    layer_spec = leabra.LayerSpec(lay_inhib=True)
    conn_spec  = leabra.ConnectionSpec(proj='full', rnd_type='uniform',  rnd_mean=0.75, rnd_var=0.2)

    # input/outputs
    input_layer  = leabra.Layer(n_input, spec=layer_spec, unit_spec=unit_spec, name='input_layer')
    output_layer = leabra.Layer(n_output, spec=layer_spec, unit_spec=unit_spec, name='output_layer')

    # creating the required numbers of hidden layers and connections
    layers = [input_layer]
    connections = []
    for i in range(n_hidden):
        hidden_layer = leabra.Layer(n_input, spec=layer_spec, unit_spec=unit_spec, name='hidden_layer_{}'.format(i))
        hidden_conn  = leabra.Connection(layers[-1],  hidden_layer, spec=conn_spec)
        layers.append(hidden_layer)
        connections.append(hidden_conn)

    last_conn  = leabra.Connection(layers[-1],  output_layer, spec=conn_spec)
    layers.append(output_layer)

    network_spec = leabra.NetworkSpec(quarter_size=50)
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

    network.trial()
    return [unit.act_m for unit in network.layers[-1].units]


if __name__ == '__main__':
    network = build_network(36, 24, 1)
    print(test_network(network, 18*[1.0, 0.0]))
    print(train_network(network, 18*[1.0, 0.0], 12*[0.0, 1.0]))
    print(train_network(network, 18*[1.0, 0.0], 12*[0.0, 1.0]))
    print(train_network(network, 18*[1.0, 0.0], 12*[0.0, 1.0]))
