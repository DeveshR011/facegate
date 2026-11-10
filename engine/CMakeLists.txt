cmake_minimum_required(VERSION 3.20)
project(FaceGateEngine LANGUAGES CXX)

if(NOT MSVC)
  message(FATAL_ERROR "The engine builds with MSVC only (DirectML is Windows-only).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT: the DLL then depends on onnxruntime.dll and the OS, nothing else.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ort)
set(DML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/directml)
foreach(p ${ORT_DIR}/runtimes/win-x64/native/onnxruntime.lib ${DML_DIR}/bin/x64-win/DirectML.dll)
  if(NOT EXISTS ${p})
    message(FATAL_ERROR "Missing ${p} - run engine\\fetch_deps.ps1 first.")
  endif()
endforeach()

add_library(facegate_engine SHARED src/engine.cpp)
target_compile_definitions(facegate_engine PRIVATE FACEGATE_BUILD UNICODE _UNICODE NOMINMAX)
target_include_directories(facegate_engine PRIVATE include src)
# Treat the vendor headers as system headers so /W4 /WX applies to our code only.
target_include_directories(facegate_engine SYSTEM PRIVATE ${ORT_DIR}/build/native/include)
target_compile_options(facegate_engine PRIVATE /W4 /WX /permissive- /O2 /GL /guard:cf /sdl /EHsc)
target_link_options(facegate_engine PRIVATE /LTCG /guard:cf /DYNAMICBASE /NXCOMPAT /HIGHENTROPYVA)
target_link_libraries(facegate_engine PRIVATE
  ${ORT_DIR}/runtimes/win-x64/native/onnxruntime.lib dxgi)

# Stage the runtime next to the engine so it can be loaded (and bundled) as one folder.
add_custom_command(TARGET facegate_engine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${ORT_DIR}/runtimes/win-x64/native/onnxruntime.dll
          ${ORT_DIR}/runtimes/win-x64/native/onnxruntime_providers_shared.dll
          ${DML_DIR}/bin/x64-win/DirectML.dll
          $<TARGET_FILE_DIR:facegate_engine>
  COMMENT "Staging onnxruntime.dll and DirectML.dll")
