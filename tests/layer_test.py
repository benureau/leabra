import unittest
import copy

import numpy as np

import data

import dotdot  # pylint: disable=unused-import
import leabra


class LayerTestsBehavior(unittest.TestCase):
    """Check that the Layer behaves as they should.

    The checks are concerned with consistency with the equation that define the
    layer's behavior. These test should pass on the emergent implementation as
    well.
    """

    def test_emergent_layer(self):
        """Test quantitative equivalence with emergent on a basic layer inhibition project."""
        emergent_data = data.parse_unit('layer_1.txt')

        unit_spec0 = leabra.UnitSpec(adapt_on=True, noisy_act=True)
        layer_spec0 = leabra.LayerSpec(gi=0.4, ff=1.0,fb=0.5)
        connection_spec0 = leabra.ConnectionSpec(proj='1to1')

        src_layer = leabra.Layer(10,spec=layer_spec0, unit_spec=unit_spec0)
        dst_layer = leabra.Layer(10,spec=layer_spec0, unit_spec= unit_spec0)

        connection0 = leabra.Connection(src_layer,dst_layer, spec=connection_spec0)

        input_pattern = 5 * [1.0, 0.0]

        for i in range(200):
            if ((i >= 10) and (i < 160)):
                src_layer.add_excitatory(input_pattern, forced=True)
            else:
                src_layer.add_excitatory(10 * [0.0], forced=True)
            src_layer.cycle()
            dst_layer.cycle()

        check = True
        for name in dst_layer.units[0].logs.keys():
            for t, (py, em) in enumerate(zip(dst_layer.units[0].logs[name], emergent_data[name])):
                if not np.allclose(py, em, rtol=1e-05, atol=1e-07):
                    print('{}:{} [py] {:.10f} != {:.10f} [emergent] ({}adapt)'.format(
                            name, t,   py,        em,                 '' if False else 'no '))
                    check = False

        self.assertTrue(check)


if __name__ == '__main__':
    unittest.main()
