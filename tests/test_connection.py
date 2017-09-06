import dotdot
import leabra

def test_sig_inv():
    conn_spec = leabra.ConnectionSpec()
    assert conn_spec.sig_inv(-1.0) == 0.0
    assert conn_spec.sig_inv( 0.0) == 0.0
    assert conn_spec.sig_inv( 0.5) == 0.5
    assert conn_spec.sig_inv( 1.0) == 1.0
    assert conn_spec.sig_inv( 2.0) == 1.0
