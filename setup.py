#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from distutils.version import LooseVersion
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.fail_fast = True
from cython import __version__ as cython_version
from pkg_resources import parse_version as V

import numpy
import sys
import os
import json
import re

from subprocess import Popen, PIPE


def get_git_sha1():
    try:
        import git
        required_version = '0.3.7'
        if V(git.__version__) < V(required_version):
            raise ImportError('could not import gitpython>=%s' % required_version)
    except ImportError as e:
        print >>sys.stderr, e
        return None
    repo = git.Repo(os.path.dirname(__file__))
    sha1 = repo.iter_commits().next().hexsha
    return sha1


def find_dependency(soname, incname):
    def test(prefix):
        sofile = os.path.join(prefix, 'lib/{}'.format(soname))
        incdir = os.path.join(prefix, 'include/{}'.format(incname))
        if os.path.isfile(sofile) and os.path.isdir(incdir):
            return (os.path.join(prefix, 'lib'),
                    os.path.join(prefix, 'include'))
        return None
    if 'VIRTUAL_ENV' in os.environ:
        ret = test(os.environ['VIRTUAL_ENV'])
        if ret is not None:
            return ret[0], ret[1]
    if 'CONDA_BUILD' in os.environ:
        d = os.environ.get('PREFIX', None)
        if d:
            ret = test(d)
            if ret is not None:
                return ret[0], ret[1]
    if 'CONDA_DEFAULT_ENV' in os.environ:
        # shell out to conda to get info
        cmd = ['conda', 'info', '--json']
        s = Popen(cmd, shell=False, stdout=PIPE).stdout.read()
        s = json.loads(s)
        if 'default_prefix' in s:
            ret = test(str(s['default_prefix']))
            if ret is not None:
                return ret[0], ret[1]
    return None, None


def find_cython_dependency(dirname):
    def test(prefix):
        incdir = os.path.join(prefix, 'cython/{}'.format(dirname))
        if os.path.isdir(incdir):
            return os.path.join(prefix, 'cython')
        return None
    if 'VIRTUAL_ENV' in os.environ:
        ret = test(os.environ['VIRTUAL_ENV'])
        if ret is not None:
            return ret
    if 'CONDA_BUILD' in os.environ:
        d = os.environ.get('PREFIX', None)
        if d:
            ret = test(d)
            if ret is not None:
                return ret
    if 'CONDA_DEFAULT_ENV' in os.environ:
        # shell out to conda to get info
        cmd = ['conda', 'info', '--json']
        s = Popen(cmd, shell=False, stdout=PIPE).stdout.read()
        s = json.loads(s)
        if 'default_prefix' in s:
            ret = test(str(s['default_prefix']))
            if ret is not None:
                return ret
    return None

clang = False
if sys.platform.lower().startswith('darwin'):
    clang = True

so_ext = 'dylib' if clang else 'so'

min_cython_version = '0.20.2' if clang else '0.20.1'
if LooseVersion(cython_version) < LooseVersion(min_cython_version):
    raise ValueError(
        'cython support requires cython>={}'.format(min_cython_version))

cc = os.environ.get('CC', None)
cxx = os.environ.get('CXX', None)
debug_build = 'DEBUG' in os.environ

distributions_lib, distributions_inc = find_dependency(
    'libdistributions_shared.{}'.format(so_ext), 'distributions')
microscopes_common_lib, microscopes_common_inc = find_dependency(
    'libmicroscopes_common.{}'.format(so_ext), 'microscopes')
microscopes_common_cython_inc = find_cython_dependency('microscopes')
microscopes_lda_lib, microscopes_lda_inc = find_dependency(
    'libmicroscopes_lda.{}'.format(so_ext), 'microscopes')

join = os.path.join
dirname = os.path.dirname
basedir = join(dirname(__file__), 'microscopes', 'lda')

if 'OFFICIAL_BUILD' not in os.environ:
    sha1 = get_git_sha1()
    if sha1 is None:
        sha1 = 'unknown'
    print 'writing git hash:', sha1
    githashfile = join(basedir, 'githash.txt')
    with open(githashfile, 'w') as fp:
        print >>fp, sha1
elif debug_build:
    raise RuntimeError("OFFICIAL_BUILD and DEBUG both set")

if distributions_inc is not None:
    print 'Using distributions_inc:', distributions_inc
if distributions_lib is not None:
    print 'Using distributions_lib:', distributions_lib
if microscopes_common_inc is not None:
    print 'Using microscopes_common_inc:', microscopes_common_inc
if microscopes_common_cython_inc is not None:
    print 'Using microscopes_common_cython_inc:', microscopes_common_cython_inc
if microscopes_common_lib is not None:
    print 'Using microscopes_common_lib:', microscopes_common_lib
if microscopes_lda_inc is not None:
    print 'Using microscopes_lda_inc:', microscopes_lda_inc
if microscopes_lda_lib is not None:
    print 'Using microscopes_lda_lib:', microscopes_lda_lib
if cc is not None:
    print 'Using CC={}'.format(cc)
if cxx is not None:
    print 'Using CXX={}'.format(cxx)
if debug_build:
    print 'Debug build'

extra_compile_args = [
    '-std=c++0x',
    '-Wno-unused-function',
]
# taken from distributions
math_opt_flags = [
    '-mfpmath=sse',
    '-msse4.1',
    #'-ffast-math',
    #'-funsafe-math-optimizations',
]
if not debug_build:
    extra_compile_args.extend(math_opt_flags)
if clang:
    extra_compile_args.extend([
        '-mmacosx-version-min=10.7',  # for anaconda
        '-stdlib=libc++',
        '-Wno-deprecated-register',
    ])
if debug_build:
    extra_compile_args.append('-DDEBUG_MODE')

include_dirs = [numpy.get_include()]
if 'EXTRA_INCLUDE_PATH' in os.environ:
    include_dirs.append(os.environ['EXTRA_INCLUDE_PATH'])
if distributions_inc is not None:
    include_dirs.append(distributions_inc)
if microscopes_common_inc is not None:
    include_dirs.append(microscopes_common_inc)
if microscopes_lda_inc is not None:
    include_dirs.append(microscopes_lda_inc)

library_dirs = []
if distributions_lib is not None:
    library_dirs.append(distributions_lib)
if microscopes_common_lib is not None:
    library_dirs.append(microscopes_common_lib)
if microscopes_lda_lib is not None:
    library_dirs.append(microscopes_lda_lib)

extra_link_args = []
if 'EXTRA_LINK_ARGS' in os.environ:
    extra_link_args.append(os.environ['EXTRA_LINK_ARGS'])


def make_extension(module_name):
    sources = [module_name.replace('.', '/') + '.pyx']
    return Extension(
        module_name,
        sources=sources,
        language="c++",
        include_dirs=include_dirs,
        libraries=["microscopes_common", "microscopes_lda",
                   "protobuf", "distributions_shared"],
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

extensions = cythonize([
    make_extension('microscopes.lda._model'),
    make_extension('microscopes.lda.definition'),
    make_extension('microscopes.lda.kernels'),
    make_extension('microscopes.lda.biology_data'),
], include_path=[microscopes_common_cython_inc])

with open('README.md') as f:
    long_description = f.read()


version = None
with open(join(basedir, '__init__.py')) as fp:
    for line in fp:
        if re.match("_version_base\s+=\s+'\S+'$", line):
            version = line.split()[-1].strip("'")
if not version:
    raise RuntimeError("could not determine version")

setup(version=version,
      name='microscopes-lda',
      description='Non-parametric bayesian inference',
      long_description=long_description,
      url='https://github.com/datamicroscopes/lda',
      author='Stephen Tu, Eric Jonas',
      maintainer='Stephen Tu',
      maintainer_email='tu.stephenl@gmail.com',
      packages=(
          'microscopes.lda',
      ),
      ext_modules=extensions)
