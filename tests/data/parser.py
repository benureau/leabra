import os


unit_fmt = {'cycle': int, 'net': float, 'I_net': float, 'v_m': float,
            'act': float, 'act_eq': float, 'spike': int, 'adapt': float,
            'syn_tr': float, 'syn_pr': float, 'syn_nr': float, 'syn_kre': float,
            'vm_eq': float}

def parse_unit(filename):
    return parse_file(filename, unit_fmt)

def parse_xy(filename):
    return parse_file(filename, {'x': int, 'y': int})


def parse_file(filename, fmt):
    """Return the data file as a dict

    `vm_eq` is renamed in `v_m_eq`.
    """

    filepath = os.path.join(os.path.abspath(os.path.join(__file__, '..')), filename)
    with open(filepath, 'r') as fd:
        lines = fd.readlines()

    # removing the newline characters
    for i, line in enumerate(lines):
        if line.endswith('\n'):
            lines[i] = line[:-1]

    # checking some assumptions about the format
    assert lines[0].startswith('_H:'), 'Unrecognized format {}'.format(filepath)
    for line in lines[1:]:
        assert len(line) == 0 or line.startswith('_D:'), 'Unrecognized format {}'.format(filepath)

    header = lines[0].split('\t')
    header = [name[1:] for name in header[1:]]

    assert set(list(fmt.keys())) == set(header)

    data = {name: [] for name in header}

    for line in lines[1:]:
        if len(line) > 0:
            values = line.split('\t')[1:]
            assert len(values) == len(header)

            for name, v in zip(header, values):
                data[name].append(fmt[name](v))

    if 'vm_eq' in data:
        data['v_m_eq'] = data.pop('vm_eq')
    return data

if __name__ == '__main__':
    print(parse_file('neuron.txt'))
