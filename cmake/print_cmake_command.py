import sys
import os
from subprocess import check_output

if __name__ == '__main__':
    build_type = "RelWithDebInfo"
    if len(sys.argv) > 1:
        build_type = sys.argv[1]
    if build_type not in ('Debug', 'Release', 'RelWithDebInfo'):
        raise ValueError("invalid build type: {}".format(build_type))
    ## XXX: handle virtualenv
    conda_full_path = check_output("which conda", shell=True).strip()
    if 'CONDA_DEFAULT_ENV' in os.environ:
        a, b = os.path.split(conda_full_path)
        assert b == 'conda'
        a, b = os.path.split(a)
        assert b == 'bin'
        conda_env_path = a
        a, b = os.path.split(a)
        assert b == os.environ['CONDA_DEFAULT_ENV']
    else:
        a, b = os.path.split(conda_full_path)
        assert b == 'conda'
        a, b = os.path.split(a)
        assert b == 'bin'
        conda_env_path = a
    print 'cmake -DCMAKE_BUILD_TYPE={} -DCMAKE_INSTALL_PREFIX={} -DCMAKE_PREFIX_PATH={} ..'.format(
            build_type,
            conda_env_path,
            conda_env_path)
