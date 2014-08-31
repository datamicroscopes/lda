#!/bin/sh

### currently, looks for either a virtualenv or anaconda install

LIBNAME=$1
INCNAME=$2

UNAME=`uname`
if [ "${UNAME}" = "Darwin" ]; then
    SOEXT=dylib
else
    SOEXT=so
fi

if [ -n "${VIRTUAL_ENV}" ]; then
    if [ -f "${VIRTUAL_ENV}/lib/lib${LIBNAME}.${SOEXT}" ] && [ -d "${VIRTUAL_ENV}/include/${INCNAME}" ]; then
        echo "${VIRTUAL_ENV}"
        exit 0
    fi
fi

if [ "${CONDA_BUILD}" = "1" ]; then
    if [ -f "${PREFIX}/lib/lib${LIBNAME}.${SOEXT}" ] && [ -d "${PREFIX}/include/${INCNAME}" ]; then
        echo "${PREFIX}"
        exit 0
    fi
fi

if [ -n "${CONDA_DEFAULT_ENV}" ]; then
    DIR=`conda info | grep 'default environment' | awk '{print $4}'`
    if [ -f "${DIR}/lib/lib${LIBNAME}.${SOEXT}" ] && [ -d "${DIR}/include/${INCNAME}" ]; then
        echo "${DIR}"
        exit 0
    fi
fi

exit 1
