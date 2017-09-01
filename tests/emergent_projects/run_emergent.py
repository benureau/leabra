import time
import subprocess
import socket
import sys
import json


# start emergent
filename = '/Users/fabien/research/renc/projects/leabra/leabra/tests/emergent_projects/neuron.proj'
p = subprocess.Popen(['emergent', '-nogui', '-server', '-p', filename])
time.sleep(10.0)

def connect(tries=5):
    while tries > 0:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('localhost', 5360))
            return s
        except ConnectionRefusedError:
            print('refused ({})'.format(tries))
            time.sleep(1)
            tries -= 1
    raise ConnectionRefusedError

def send(cmd):
    print(cmd)
    s.send(('{}\n'.format(json.dumps(cmd))).encode('utf-8'))
    #check status
    response = read_socket(s)
    assert json.loads(response)['status'] == 'OK'

def read_socket(s):
    #return s.recv(4096)
    data = b''
    part = s.recv(4096)
    data += part
    while len(part) == 4096: # QUESTION: what if it's *exactly* 4096?
        part = s.recv(4096)
        data += part
    return data

try:

    # connect to emergent
    #s.connect(('localhost', 5360))
    s = connect()

    output_filename = '/Users/fabien/research/renc/projects/leabra/leabra/tests/data/neuron_adapt'
    cmds = {'set_adapt_off': {'command': 'SetVar', 'program': 'SetDefaults', 'var_name': 'adapt_on', 'var_value': False},
            'set_defaults': {'command': 'RunProgram', 'program': 'SetDefaults'},
            'run':  {'command': 'RunProgram', 'program': 'LeabraSettle'},
            'set_filename': {'command': 'SetVar', 'program': 'SaveOutput', 'var_name': 'file_name', 'var_value': output_filename},
            'save': {'command': 'RunProgram', 'program': 'SaveOutput'},}

    print(read_socket(s))
    # send(cmds['set_adapt_off'])
    send(cmds['set_defaults'])
    send(cmds['run'])
    send(cmds['set_filename'])
    send(cmds['save'])

finally:
    p.kill()
