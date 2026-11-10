cmake_minimum_required(VERSION 3.20)
project(FaceGateCP LANGUAGES CXX)

if(NOT MSVC)
  message(FATAL_ERROR "The credential provider builds with MSVC only.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT: LogonUI must not depend on a redistributable being installed.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

function(facegate_cp target)
  add_library(${target} SHARED FaceGateCP.cpp FaceGateCP.def)
  target_compile_definitions(${target} PRIVATE UNICODE _UNICODE NOMINMAX)
  target_compile_options(${target} PRIVATE /W4 /WX /permissive- /guard:cf /sdl)
  target_link_options(${target} PRIVATE /guard:cf /DYNAMICBASE /NXCOMPAT)
  target_link_libraries(${target} PRIVATE secur32 shlwapi gdiplus ole32 advapi32)
endfunction()

# The DLL that gets installed.
facegate_cp(FaceGateCP)

# Same source, pointed at a private pipe and a fixed SID, plus a harness that
# plays LogonUI. Never installed.
option(FACEGATE_CP_TESTS "Build the credential provider test harness" ON)
if(FACEGATE_CP_TESTS)
  facegate_cp(FaceGateCP_test)
  target_compile_definitions(FaceGateCP_test PRIVATE FACEGATE_TEST_BUILD)
  add_executable(cp_test test/cp_test.cpp)
  target_compile_definitions(cp_test PRIVATE UNICODE _UNICODE)
  target_compile_options(cp_test PRIVATE /W4 /permissive-)
  target_link_libraries(cp_test PRIVATE ole32)
  add_dependencies(cp_test FaceGateCP_test)
  enable_testing()
  add_test(NAME credential_provider COMMAND cp_test $<TARGET_FILE:FaceGateCP_test>)
endif()
