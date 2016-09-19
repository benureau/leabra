import unittest

import numpy as np

import dotdot  # pylint: disable=unused-import
import leabra



class LayerTestAPI(unittest.TestCase):

    def simple_usage_test(self):
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


if __name__ == '__main__':
    unittest.main()
