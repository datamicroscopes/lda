import os

# NOTE: _version_base must be of the form
# _version_base = '...', since setup.py depends on it
_version_base = '0.2.0'

try:
    # read git hash from file
    with open(os.path.join(os.path.dirname(__file__), 'githash.txt')) as fp:
        _githash = fp.read().strip()
    __version__ = '{base}-{githash}'.format(
        base=_version_base, githash=_githash)
except IOError:
    __version__ = _version_base
