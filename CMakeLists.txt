cmake_minimum_required(VERSION 3.16)
project(np_mpc_public CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(casadi REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(npmpc
    src_cpp/npmpc/dynamics/integrators/furuta_np.cpp
    src_cpp/npmpc/nps/casadi_utils.cpp
    src_cpp/npmpc/nps/decoder.cpp
    src_cpp/npmpc/nps/NeuralProcess.cpp
    src_cpp/npmpc/nps/weights_io.cpp
    src_cpp/npmpc/mpc/problem/MPCBase.cpp
    src_cpp/npmpc/mpc/problem/furuta_cost.cpp
    src_cpp/npmpc/mpc/problem/FurutaNPMPC.cpp
    src_cpp/npmpc/mpc/solver.cpp
    src_cpp/npmpc/mpc/controller.cpp
    src_cpp/npmpc/mpc/mpc_config_io.cpp
)
target_include_directories(npmpc PUBLIC src_cpp)
target_link_libraries(npmpc PUBLIC casadi::casadi yaml-cpp::yaml-cpp)

add_executable(eval_furuta_np src_cpp/tests/eval_furuta_np.cpp)
target_link_libraries(eval_furuta_np PRIVATE npmpc)
target_compile_definitions(eval_furuta_np PRIVATE NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(eval_furuta_mpc src_cpp/tests/eval_furuta_mpc.cpp)
target_link_libraries(eval_furuta_mpc PRIVATE npmpc)
target_compile_definitions(eval_furuta_mpc PRIVATE NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
