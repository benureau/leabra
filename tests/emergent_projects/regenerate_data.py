import os
import sys
import time
import json
import socket
import subprocess


here = os.path.dirname(__file__)
datadir = os.path.join(here, '../data')


class Emergent:

    def __init__(self, project_filename, verbose=True):
        """Open a project with a headless emergent"""
        self.project_filename = os.path.join(here, project_filename)
        self.verbose = verbose

        self.p = subprocess.Popen(['emergent', '-nogui', '-server', '-p', self.project_filename])
        self.connect()

    def connect(self, tries=10):
        """Create a socket connected to the emergent instance"""
        while tries > 0:
            try:
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.connect(('localhost', 5360))
                assert b'Emergent Server' in self.read_socket()
                return
            except ConnectionRefusedError:
                #print('refused ({})'.format(tries))
                time.sleep(1)
                tries -= 1
        raise ConnectionRefusedError

    def send(self, cmd, verbose=True):
        """Send a command to emergent a verify it went well."""
        if verbose:
            print('send: {}'.format(cmd))
        self.s.send(('{}\n'.format(json.dumps(cmd))).encode('utf-8'))
        #check status
        response = self.read_socket()
        response_json = json.loads(response)
        if verbose:
            'recv: {}'.format(response_json)
        assert response_json['status'] == 'OK'

    def read_socket(self):
        """Read all the data available on the socket"""
        data = b''
        part = self.s.recv(4096)
        data += part
        while len(part) == 4096: # QUESTION: what if it's *exactly* 4096?
            part = self.s.recv(4096)
            data += part
        return data

    def close(self):
        try:
            self.s.close()
        finally:
            self.p.kill() # using kill rather than terminate to avoid recovery file creation.


if __name__ == '__main__':
    run_cmd      = {'command': 'RunProgram', 'program': 'LeabraSettle'}
    filename_cmd = {'command': 'SetVar', 'program': 'SaveOutput', 'var_name': 'file_name', 'var_value': None}
    save_cmd     = {'command': 'RunProgram', 'program': 'SaveOutput'}

    for adapt, output_filename in [(True, 'neuron_adapt'), (False, 'neuron')]:
        try:
            print('# Generating {}.dat'.format(output_filename))
            em = Emergent('neuron.proj')
            filename_cmd['var_value'] = os.path.join(datadir, output_filename)

            em.send({'command': 'SetVar', 'program': 'SetAdapt', 'var_name': 'adapt_on', 'var_value': adapt})
            em.send({'command': 'RunProgram', 'program': 'SetAdapt'})
            em.send(run_cmd)
            em.send(filename_cmd)
            em.send(save_cmd)

            print('Generated {}.dat\n'.format(output_filename))

        finally:
            em.close()
            time.sleep(2.0)
