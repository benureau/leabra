import os
import subprocess

import nbformat


def _notebook_run(filepath):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    _, name = os.path.split(filepath)
    name = os.path.splitext(name)[0]
    outdir = os.path.abspath(os.path.join(__file__, '../outs'))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    outfilename = os.path.join(outdir, '{}.ipynb'.format(name))
    args = ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            '--ExecutePreprocessor.timeout=120',
            '--output', outfilename, filepath]
    subprocess.check_call(args)

    with open(outfilename, 'r') as f:
        nb = nbformat.read(f, nbformat.current_nbformat)
        errors = [output for cell in nb.cells if "outputs" in cell
                         for output in cell["outputs"]\
                         if output.output_type == "error"]

    return nb, errors


def test_ipynb():
    notebook_dir = os.path.abspath(os.path.join(__file__, '../../notebooks'))
    for filename in os.listdir(notebook_dir):
        if os.path.splitext(filename)[1] == '.ipynb':
            nb, errors = _notebook_run(os.path.join(notebook_dir, filename))
            assert len(errors) == 0, '{}'.format(errors)

if __name__ == '__main__':
    test_ipynb()
