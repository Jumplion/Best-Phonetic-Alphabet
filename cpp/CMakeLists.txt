cmake_minimum_required(VERSION 3.16)
project(BestPhoneticWordPairScorer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Provide an option to build tests/examples
option(BUILD_EXAMPLES "Build example executable" ON)

# Allow skipping the MSYS/MinGW vs MSVC toolchain mismatch check when the user
# knows what they're doing (use with caution). Default: OFF
option(SKIP_SQLITE_TOOLCHAIN_CHECK "Skip the MinGW/MSYS vs MSVC SQLite toolchain mismatch fatal error" OFF)

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# If vcpkg/unofficial sqlite provides a CMake target, use it and create a consistent alias
find_package(unofficial-sqlite3 CONFIG QUIET)
if (TARGET unofficial::sqlite3::sqlite3)
	# create a consistent target name that the rest of the CMakeLists uses
	add_library(SQLite::SQLite3 ALIAS unofficial::sqlite3::sqlite3)
endif()

# Find SQLite3 (may populate SQLite3_LIBRARY(S) and SQLite3_INCLUDE_DIR)
find_package(SQLite3)

# If a local vcpkg install exists, prefer its sqlite3.lib/include for MSVC builds
if (NOT DEFINED VCPKG_TRIPLET)
	set(VCPKG_TRIPLET "x64-windows")
endif()
set(_vcpkg_root "${PROJECT_SOURCE_DIR}/vcpkg")
set(_vcpkg_lib_path "${_vcpkg_root}/installed/${VCPKG_TRIPLET}/lib")
set(_vcpkg_include_path "${_vcpkg_root}/installed/${VCPKG_TRIPLET}/include")
if (EXISTS "${_vcpkg_lib_path}/sqlite3.lib")
	message(STATUS "Found vcpkg sqlite at ${_vcpkg_lib_path}/sqlite3.lib, will prefer it for MSVC builds.")
	# override the detected variables from find_package(SQLite3)
	set(SQLite3_LIBRARIES "${_vcpkg_lib_path}/sqlite3.lib" CACHE STRING "SQLite3 libraries (overridden by vcpkg)" FORCE)
	set(SQLite3_INCLUDE_DIR "${_vcpkg_include_path}" CACHE STRING "SQLite3 include dir (overridden by vcpkg)" FORCE)
	if (NOT TARGET SQLite::SQLite3)
		add_library(SQLite::SQLite3 UNKNOWN IMPORTED)
		set_target_properties(SQLite::SQLite3 PROPERTIES IMPORTED_LOCATION "${SQLite3_LIBRARIES}")
		set_target_properties(SQLite::SQLite3 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIR}")
	endif()
endif()

# Allow user override with -DSQLITE3_LIBRARY and -DSQLITE3_INCLUDE_DIR (common names)
if (DEFINED SQLITE3_LIBRARY AND NOT DEFINED SQLite3_LIBRARIES)
	set(SQLite3_LIBRARIES ${SQLITE3_LIBRARY})
endif()
if (DEFINED SQLITE3_INCLUDE_DIR AND NOT DEFINED SQLite3_INCLUDE_DIR)
	set(SQLite3_INCLUDE_DIR ${SQLITE3_INCLUDE_DIR})
endif()

# Some CMake/Find modules use different capitalization; normalize common variants
if (NOT DEFINED SQLite3_LIBRARIES AND DEFINED SQLite3_LIBRARY)
	set(SQLite3_LIBRARIES ${SQLite3_LIBRARY})
endif()
if (NOT DEFINED SQLite3_INCLUDE_DIR AND DEFINED SQLite3_INCLUDE_DIRS)
	set(SQLite3_INCLUDE_DIR ${SQLite3_INCLUDE_DIRS})
endif()

# If CMake found SQLite but the library looks like a MinGW/MSYS .a while user selected MSVC,
# print a helpful error unless the user explicitly provided their own SQLITE3_OVERRIDE variables
# or they set SKIP_SQLITE_TOOLCHAIN_CHECK=ON.
if (DEFINED SQLite3_LIBRARIES)
	if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		string(TOLOWER "${SQLite3_LIBRARIES}" _sqlite_lib_lc)
		if (_sqlite_lib_lc MATCHES "msys" OR _sqlite_lib_lc MATCHES "mingw" OR _sqlite_lib_lc MATCHES "\\.a$")
			if (NOT (DEFINED SQLITE3_LIBRARY AND DEFINED SQLITE3_INCLUDE_DIR))
				if (SKIP_SQLITE_TOOLCHAIN_CHECK)
					message(WARNING "CMake found SQLite in a MinGW/MSYS path (${SQLite3_LIBRARIES}) while the chosen generator uses MSVC, but SKIP_SQLITE_TOOLCHAIN_CHECK=ON so configure will continue. This will likely fail to link later. Prefer installing an MSVC-compatible sqlite (eg via vcpkg) or pass SQLITE3_LIBRARY/SQLITE3_INCLUDE_DIR.")
				else()
					message(FATAL_ERROR "CMake found SQLite in a MinGW/MSYS path (${SQLite3_LIBRARIES}) while the chosen generator uses MSVC.\nPlease install SQLite for MSVC, or pass the correct SQLITE3_LIBRARY and SQLITE3_INCLUDE_DIR variables to CMake. Example:\ncmake -DSQLITE3_LIBRARY=\"C:/path/to/sqlite3.lib\" -DSQLITE3_INCLUDE_DIR=\"C:/path/to\" ..\nAlternatively, generate with a MinGW/MSYS generator to match the found SQLite library.")
				endif()
			endif()
		endif()
	endif()
endif()

# If we have a library path, create an imported target for consistent linking
if (NOT TARGET SQLite::SQLite3 AND DEFINED SQLite3_LIBRARIES)
	add_library(SQLite::SQLite3 UNKNOWN IMPORTED)
	set_target_properties(SQLite::SQLite3 PROPERTIES
		IMPORTED_LOCATION "${SQLite3_LIBRARIES}"
	)
	if (DEFINED SQLite3_INCLUDE_DIR)
		set_target_properties(SQLite::SQLite3 PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIR}"
		)
	endif()
endif()

add_subdirectory(src)
