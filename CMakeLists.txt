cmake_minimum_required(VERSION 3.16)
project(np_mpc_public CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(laopt REQUIRED)
find_package(Ipopt REQUIRED)
find_package(piqp REQUIRED)
find_package(casadi REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(npmpc
    src_cpp/npmpc/dynamics/integrators/furuta_np.cpp
    src_cpp/npmpc/nps/casadi_utils.cpp
    src_cpp/npmpc/nps/decoder.cpp
    src_cpp/npmpc/nps/NeuralProcess.cpp
    src_cpp/npmpc/nps/weights_io.cpp
    src_cpp/npmpc/nps/eigen_utils.cpp
    src_cpp/npmpc/nps/FurutaNPEigen.cpp
    src_cpp/npmpc/mpc/problem/MPCBase.cpp
    src_cpp/npmpc/mpc/problem/furuta_cost.cpp
    src_cpp/npmpc/mpc/problem/FurutaNPMPC.cpp
    src_cpp/npmpc/mpc/solver.cpp
    src_cpp/npmpc/mpc/controller.cpp
    src_cpp/npmpc/mpc/mpc_config_io.cpp
    src_cpp/npmpc/mpc/qube_client.cpp
)
target_include_directories(npmpc PUBLIC src_cpp)
target_link_libraries(npmpc PUBLIC casadi::casadi yaml-cpp::yaml-cpp Eigen3::Eigen)

add_executable(eval_furuta_np src_cpp/tests/eval_furuta_np.cpp)
target_link_libraries(eval_furuta_np PRIVATE npmpc)
target_compile_definitions(eval_furuta_np PRIVATE NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(eval_furuta_np_eigen src_cpp/tests/eval_furuta_np_eigen.cpp)
target_link_libraries(eval_furuta_np_eigen PRIVATE npmpc)
target_compile_definitions(eval_furuta_np_eigen PRIVATE NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# laopt NP MPC with IPOPT or SQP/PIQP (the installed laopt package has no
# laopt_target_enable_* helpers, so the solvers are wired in here; IPOPT via
# cmake/FindIpopt.cmake).
add_executable(eval_furuta_np_laopt src_cpp/tests/eval_furuta_np_laopt.cpp)
target_link_libraries(eval_furuta_np_laopt PRIVATE npmpc laopt::laopt ${Ipopt_LIBRARIES} piqp::piqp_header_only)
target_include_directories(eval_furuta_np_laopt PRIVATE ${Ipopt_INCLUDE_DIRS})
target_compile_definitions(eval_furuta_np_laopt PRIVATE LAOPT_WITH_IPOPT LAOPT_WITH_PIQP NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(eval_furuta_eq_laopt src_cpp/tests/eval_furuta_eq_laopt.cpp)
target_link_libraries(eval_furuta_eq_laopt PRIVATE npmpc laopt::laopt ${Ipopt_LIBRARIES} piqp::piqp_header_only)
target_include_directories(eval_furuta_eq_laopt PRIVATE ${Ipopt_INCLUDE_DIRS})
target_compile_definitions(eval_furuta_eq_laopt PRIVATE LAOPT_WITH_IPOPT LAOPT_WITH_PIQP NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(run_furuta_eq_laopt_client src_cpp/tests/run_furuta_eq_laopt_client.cpp)
target_link_libraries(run_furuta_eq_laopt_client PRIVATE npmpc laopt::laopt ${Ipopt_LIBRARIES} piqp::piqp_header_only)
target_include_directories(run_furuta_eq_laopt_client PRIVATE ${Ipopt_INCLUDE_DIRS})
target_compile_definitions(run_furuta_eq_laopt_client PRIVATE LAOPT_WITH_IPOPT LAOPT_WITH_PIQP NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(run_furuta_np_laopt_client src_cpp/tests/run_furuta_np_laopt_client.cpp)
target_link_libraries(run_furuta_np_laopt_client PRIVATE npmpc laopt::laopt ${Ipopt_LIBRARIES} piqp::piqp_header_only)
target_include_directories(run_furuta_np_laopt_client PRIVATE ${Ipopt_INCLUDE_DIRS})
target_compile_definitions(run_furuta_np_laopt_client PRIVATE LAOPT_WITH_IPOPT LAOPT_WITH_PIQP NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(eval_furuta_mpc src_cpp/tests/eval_furuta_mpc.cpp)
target_link_libraries(eval_furuta_mpc PRIVATE npmpc)
target_compile_definitions(eval_furuta_mpc PRIVATE NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(run_furuta_np_mpc_client src_cpp/tests/run_furuta_np_mpc_client.cpp)
target_link_libraries(run_furuta_np_mpc_client PRIVATE npmpc)
target_compile_definitions(run_furuta_np_mpc_client PRIVATE NPMPC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
