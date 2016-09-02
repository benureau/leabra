import unittest

import dotdot  # pylint: disable=unused-import
import leabra


class UnitTests(unittest.TestCase):
    """Diverse checks on Unit behavior."""

    def test_xx1(self):
        u_spec = leabra.UnitSpec()
        u_spec.act_thr = 0.25
        u = leabra.Unit(spec=u_spec)
        self.assertEqual(u.xx1(0.20), 0.0)
        self.assertTrue(u.xx1(0.30) > 0.0)
        self.assertTrue(u.noisy_xx1(0.20) > 0.0)
        self.assertTrue(u.noisy_xx1(0.30) > 0.0)

    def test_instantiate(self):
        u_spec = leabra.UnitSpec()
        u_spec.act_thr = 0.25
        u = u_spec.instantiate()
        u2 = u_spec.instantiate()
        self.assertEqual(u.spec.act_thr, 0.25)
        self.assertEqual(u2.spec.act_thr, 0.25)
        u.spec.act_thr = 0.30
        self.assertEqual(u.spec.act_thr, 0.30)
        self.assertEqual(u2.spec.act_thr, 0.30)



if __name__ == '__main__':
    unittest.main()
