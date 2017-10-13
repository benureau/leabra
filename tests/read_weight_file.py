"""Code to read emergent weights files (*.wts), so we can set the connection weights exactly like the emergent project.

Note: This code is properly horrible. And there are ways to write it properly, beautifully. Should you be motivated to do so, go write a patch for emergent to dump weights as JSON instead.
"""


def read_weights(filename):
    with open(filename, 'r') as fd:
        lines = fd.read().split('\n')

    assert lines[0] == '<Fmt TEXT>'
    assert lines[2] == '<Epoch 0>'

    weights = {}

    line_cursor = 3
    while line_cursor < len(lines):
        line = lines[line_cursor]
        if line.startswith('<Lay '):
            line_cursor = _read_layer(lines, line_cursor, weights)
        else:
            line_cursor += 1

    weights = _post_processing(weights)

    return weights

def _read_layer(lines, line_cursor, weights):
    line = lines[line_cursor]
    to_layer_name = line[5:-1]
    while line != '</Lay>':
        if line.startswith('<UgUn '):
            to_index = int(line[6:-1])
            line_cursor += 1
        elif line.startswith('<Cg '):
            line_cursor, data = _read_Cg(lines, line_cursor)
            from_layer_name, values = data
            weights.setdefault((from_layer_name, to_layer_name), [])
            weights[(from_layer_name, to_layer_name)].append((to_index, values))
        else:
            line_cursor += 1
        line = lines[line_cursor]

    return line_cursor + 1

def _read_Cg(lines, line_cursor):
    line = lines[line_cursor]
    from_layer_name = line.split(':')[1][:-1]
    line_cursor += 1
    n = int(lines[line_cursor][4:-1])
    line_cursor += 1
    values = []
    for i in range(n):
        index, value = lines[line_cursor].split(' ')
        index, value = int(index), float(value)
        assert i == index
        values.append(value)
        line_cursor += 1

    return line_cursor, (from_layer_name, values)

def _post_processing(weights):
    post_weights = {}
    for key, wts_values in weights.items():
        indexes = [to_index for to_index, wts in wts_values]
        assert set(range(len(indexes))) == set(indexes)
        from_n = None
        for to_index, wts in wts_values:
            if from_n is None:
                from_n = len(wts)
                weight_matrix = [[] for _ in range(from_n)]
            else:
                assert from_n == len(wts)
            for i, w in enumerate(wts):
                weight_matrix[i].append(w)
        post_weights[key] = weight_matrix

    return post_weights


if __name__ == '__main__':
    print(read_weights('leabra_std.wts'))
