message(STATUS "Finding microscopes-common")

execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_in_venv_like.sh microscopes_common microscopes
    OUTPUT_VARIABLE MICROSCOPES_COMMON_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(MICROSCOPES_COMMON_ROOT)
    set(MICROSCOPES_COMMON_INCLUDE_DIRS ${MICROSCOPES_COMMON_ROOT}/include)
    set(MICROSCOPES_COMMON_LIBRARY_DIRS ${MICROSCOPES_COMMON_ROOT}/lib)
    set(MICROSCOPES_COMMON_LIBRARIES microscopes_common)
    set(MICROSCOPES_COMMON_FOUND true)
else()
    message(STATUS "could not locate microscopes_common") 
    set(MICROSCOPES_COMMON_FOUND false)
endif()
