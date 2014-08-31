message(STATUS "Finding distributions")

execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_in_venv_like.sh distributions_shared distributions
    OUTPUT_VARIABLE DISTRIBUTIONS_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(DISTRIBUTIONS_ROOT)
    set(DISTRIBUTIONS_INCLUDE_DIRS ${DISTRIBUTIONS_ROOT}/include)
    set(DISTRIBUTIONS_LIBRARY_DIRS ${DISTRIBUTIONS_ROOT}/lib)
    set(DISTRIBUTIONS_LIBRARIES distributions_shared)
    set(DISTRIBUTIONS_FOUND true)
else()
    message(STATUS "could not locate distributions") 
    set(DISTRIBUTIONS_FOUND false)
endif()
