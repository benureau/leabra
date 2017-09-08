import data


def test_parser():
    """Test that data files are well parsed"""
    record = data.parse_unit('neuron.dat')

    #_H:	|cycle	%net	%I_net	%v_m	%act	%act_eq	%avg_ss	%avg_s	%avg_m	%avg_s_eff	%avg_l	%spike	%adapt	%syn_tr	%syn_nr	%syn_pr	%syn_kre	%vm_eq
    #_D:	1	0	-0.0286364	0.391322	0	0	0.075	0.1125	0.14625	0.115875	0.4	0	0	1	1	0.2	0	0.390909

    assert len(record['net']) == 200
    assert type(record['cycle'][0]) == int
    assert record['cycle'][0] == 1
    assert type(record['I_net'][0]) == float
    assert record['I_net'][0] == -0.0286364
    assert record['net'][-1] == 5.18082e-23


def test_parser_matrix():
    """Test that data files with matrices are well parsed"""
    record = data.parse_unit('netin.dat')

    assert record['I_net'][0][0, 0] == -0.0074677
    assert record['I_net'][0][0, 1] ==  0.129545
    assert record['I_net'][0][0, 2] ==  0.125024
    assert record['I_net'][0][0, 3] ==  0.111372


if __name__ == '__main__':
    test_parser()
    test_parser_matrix()
