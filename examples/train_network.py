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
    for i in range(3):
        network.quarter()
    acts = network.layers[-1].activities
    network.quarter()
    return acts


if __name__ == '__main__':
    network = build_network(100, 25, 5)
    print(test_network(network, 50*[1.0, 0.0]))
