import unittest

import dotdot  # pylint: disable=unused-import
import leabra


class UnitTests(unittest.TestCase):
    """Diverse checks on Unit behavior."""

    def test_xx1(self):
        u_spec = leabra.UnitSpec()
        u_spec.act_thr = 0.25
        u = leabra.Unit(spec=u_spec)
        self.assertEqual(u_spec.xx1(0.20), 0.0)
        self.assertTrue(u_spec.xx1(0.30) > 0.0)
        self.assertTrue(u_spec.noisy_xx1(0.20) > 0.0)
        self.assertTrue(u_spec.noisy_xx1(0.30) > 0.0)

    def test_instantiate(self):
        u_spec = leabra.UnitSpec()
        u_spec.act_thr = 0.25
        u  = leabra.Unit(spec=u_spec)
        u2 = leabra.Unit(spec=u_spec)
        self.assertEqual(u.spec.act_thr, 0.25)
        self.assertEqual(u2.spec.act_thr, 0.25)
        u.spec.act_thr = 0.30
        self.assertEqual(u.spec.act_thr, 0.30)
        self.assertEqual(u2.spec.act_thr, 0.30)

    def test_forced_act(self):
        u = leabra.Unit()
        u.add_excitatory(0.5, forced=True)
        u.calculate_net_in()
        u.cycle()
        self.assertEqual(u.act, 0.5)
        for _ in range(10):
            u.calculate_net_in()
            u.cycle()
            self.assertEqual(u.act, 0.5)

        u.add_excitatory(0.5, forced=False)
        u.calculate_net_in()
        u.cycle()
        self.assertFalse(u.act == 0.5)



if __name__ == '__main__':
    unittest.main()
