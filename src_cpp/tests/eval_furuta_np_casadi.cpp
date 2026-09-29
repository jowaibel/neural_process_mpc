// Validates the C++ NP CasADi MPC (FurutaNPMPC + MPCController) against the
// Python FurutaNPMPC solution (the same file as eval_furuta_np_laopt.cpp,
// written by
//   scripts/eval_furuta_mpc.py --method neural --z <z> --dump logs/np_python_solution.yaml
// with the same z and config as model/mpc_config.yaml), the NP counterpart of
// eval_furuta_eq_casadi.cpp:
//  1. Cost: furutaCost at the Python solution must equal Python's objective.
//  2. Solve: MPCController::warmStart for the Python solution's x0 (same cold
//     start as Python's warm_start) must reproduce the Python solution.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <casadi/casadi.hpp>
#include <yaml-cpp/yaml.h>

#include "npmpc/mpc/controller.hpp"
#include "npmpc/mpc/mpc_config_io.hpp"
#include "npmpc/mpc/problem/FurutaNPMPC.hpp"
#include "npmpc/mpc/problem/furuta_cost.hpp"
#include "npmpc/nps/weights_io.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

namespace {

struct PythonSolution {
    casadi::DM x0; // (1, 4)
    casadi::DM x;  // (N+1, 4), rows = time steps
    casadi::DM u;  // (N, 1)
    double objective;
};

PythonSolution loadPythonSolution(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    PythonSolution py;
    py.x0 = casadi::DM(root["x0"].as<std::vector<double>>()).T();
    py.x = casadi::DM(root["x"].as<std::vector<std::vector<double>>>());
    py.u = casadi::DM(root["u"].as<std::vector<std::vector<double>>>());
    py.objective = root["objective"].as<double>();
    return py;
}

double maxAbsDiff(const casadi::DM& a, const casadi::DM& b) {
    return static_cast<double>(casadi::DM::mmax(casadi::DM::abs(a - b)));
}

} // namespace

int main(int argc, char** argv) {
    const std::string weightsPath =
        argc > 1 ? argv[1] : std::string(NPMPC_PROJECT_ROOT) + "/model/np_weights.yaml";
    const std::string mpcConfigPath =
        argc > 2 ? argv[2] : std::string(NPMPC_PROJECT_ROOT) + "/model/mpc_config.yaml";
    const std::string pythonSolutionPath =
        argc > 3 ? argv[3] : std::string(NPMPC_PROJECT_ROOT) + "/logs/np_python_solution.yaml";

    npmpc::nps::NeuralProcess np = npmpc::nps::loadNeuralProcessFromYaml(weightsPath);
    npmpc::mpc::problem::MPCParams params = npmpc::mpc::loadMPCParamsFromYaml(mpcConfigPath);
    if (params.z.is_empty()) {
        throw std::runtime_error("No latent `z` in " + mpcConfigPath +
                                 " (export it with scripts/export_mpc_config.py --z <z>)");
    }
    const PythonSolution py = loadPythonSolution(pythonSolutionPath);
    if (py.x.size1() != params.horizonSteps + 1 || py.u.size1() != params.horizonSteps) {
        throw std::runtime_error("Python solution horizon differs from horizon_steps in " + mpcConfigPath);
    }
    bool ok = true;

    // -- 1. Cost at the Python solution (without the slack cost) --
    {
        casadi::MX xSym = casadi::MX::sym("x", params.horizonSteps + 1, params.xSize);
        casadi::MX uSym = casadi::MX::sym("u", params.horizonSteps, params.uSize);
        casadi::Function costFn("furuta_cost", {xSym, uSym},
                                {npmpc::mpc::problem::furutaCost(xSym, uSym, params.cost)});
        const double cost = static_cast<double>(costFn(std::vector<casadi::DM>{py.x, py.u}).at(0));
        const double relDiff = std::abs(cost - py.objective) / std::max(1.0, std::abs(py.objective));
        std::cout.precision(12);
        std::cout << "[cost] Python: " << py.objective << "  C++ CasADi: " << cost << "  rel diff: " << relDiff << "\n";
        std::cout.precision(6);
        if (relDiff > 1e-6) {
            std::cerr << "[cost] MISMATCH\n";
            ok = false;
        }
    }

    // -- 2. Solve from the cold start and compare with the Python solution --
    npmpc::mpc::problem::FurutaNPMPC mpc(&np, params.z, params.cost);
    npmpc::mpc::MPCController controller(mpc, params);

    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    controller.warmStart(py.x0);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const double xDiff = maxAbsDiff(controller.lastX(), py.x);
    const double uDiff = maxAbsDiff(controller.lastU(), py.u);
    std::cout << "\n[C++ CasADi] warmStart: " << ms << " ms, "
              << (controller.lastConverged() ? "converged" : "NOT converged") << " ("
              << controller.lastIterCount() << " IPOPT iter)\n";
    std::cout << "U C++ CasADi: " << controller.lastU().T() << "\n";
    std::cout << "U Python:     " << py.u.T() << "\n";
    std::cout << "max |X C++ - X Python| = " << xDiff << ",  max |U C++ - U Python| = " << uDiff << "\n";
    if (xDiff > 1e-3 || uDiff > 1e-3) {
        std::cerr << "[C++ CasADi] MISMATCH with Python solution\n";
        ok = false;
    }

    std::cout << "\n" << (ok ? "OK: C++ NP CasADi MPC matches the Python FurutaNPMPC." : "FAILED") << std::endl;
    return ok ? 0 : 1;
}
