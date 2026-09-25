// Validates the equation-based laopt OCP (FurutaEqOcpEigen.hpp) against the
// Python FurutaMPC solution (YAML written by
//   scripts/eval_furuta_mpc.py --method equation --dump model/eq_python_solution.yaml
// from the same config as model/mpc_config_equation.yaml, written by
//   scripts/export_mpc_config.py --method equation):
//  1. Dynamics: the implicit-midpoint residual of the Furuta ODE port on the
//     Python solution must be at solver-tolerance level.
//  2. Cost: the laopt objective at the Python solution must equal Python's objective.
//  3. Solve, with IPOPT and with SQP (PIQP): from the same cold start as
//     MPCController.warm_start (x linear from x0 to [2pi,0,0,0], u = 0),
//     re-solving until converged like Python's warm-start retries, the solution
//     must match the Python one.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "laopt/laopt.hpp"
#include "npmpc/mpc/laopt/multiple_shooting_xdiff.hpp"

#include "npmpc/mpc/laopt/FurutaEqOcpEigen.hpp"
#include "npmpc/mpc/laopt/laopt_solver.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

namespace {

using namespace npmpc::mpc::furuta_laopt;

constexpr int N = 12; // must match horizon_steps in the config
constexpr int kMaxSolves = 100; // re-solves until converged, as MPCController.warm_start's retries

using Ocp = FurutaEqOCP<N>;
using Transcription = laopt_tools::MultipleShootingXDiff<Ocp, N, laopt::IRK2>; // implicit midpoint, as in Python
using OptProblem = laopt::Problem<Transcription>;

using StateTrajectory = Transcription::StateTrajectory; // (NX, N+1)
using InputTrajectory = Transcription::InputTrajectory; // (NU, N)

struct PythonSolution {
    Ocp::State x0;
    StateTrajectory x;
    InputTrajectory u;
    double objective;
};

PythonSolution loadPythonSolution(const std::string& path)
{
    YAML::Node root = YAML::LoadFile(path);
    PythonSolution py;
    auto x0 = root["x0"].as<std::vector<double>>();
    for (int j = 0; j < kNX; ++j) { py.x0(j) = x0.at(j); }
    auto x = root["x"].as<std::vector<std::vector<double>>>();
    auto u = root["u"].as<std::vector<std::vector<double>>>();
    if (x.size() != static_cast<size_t>(N + 1) || u.size() != static_cast<size_t>(N)) {
        throw std::runtime_error("loadPythonSolution: " + path + " has a different horizon than N = " + std::to_string(N));
    }
    for (int k = 0; k <= N; ++k) {
        for (int j = 0; j < kNX; ++j) { py.x(j, k) = x[k].at(j); }
    }
    for (int k = 0; k < N; ++k) { py.u(0, k) = u[k].at(0); }
    py.objective = root["objective"].as<double>();
    return py;
}

// laopt objective as MultipleShootingXDiff assembles it for continuous dynamics:
// sum_k h * lagrange(x_k, x_{k+1}, u_k) + mayer, h = 1/N.
double ocpObjective(Ocp& ocp, const StateTrajectory& x, const InputTrajectory& u, const Ocp::Param& p)
{
    const Eigen::Vector<double, 1> t0 = Eigen::Vector<double, 1>::Constant(ocp.t0);
    const Eigen::Vector<double, 1> tf = Eigen::Vector<double, 1>::Constant(ocp.tf_lb);
    const double h = 1.0 / N;
    double obj = 0.0;
    for (int k = 0; k < N; ++k) {
        obj += h * ocp.lagrange_term_impl(Ocp::State(x.col(k)), Ocp::State(x.col(k + 1)), Ocp::Input(u.col(k)), p, t0, tf, double(k) / N);
    }
    return obj + ocp.mayer_term_impl(Ocp::State(x.col(N)), p, t0, tf);
}

// Cold start as in MPCController.warm_start: x linear from x0 to [2pi,0,0,0], u = 0.
StateTrajectory coldStartX(const Ocp::State& x0)
{
    const Ocp::State target(2.0 * M_PI, 0.0, 0.0, 0.0);
    StateTrajectory x;
    for (int i = 0; i <= N; ++i) {
        const double t = static_cast<double>(i) / N;
        x.col(i) = x0 + t * (target - x0);
    }
    return x;
}

// Solves the OCP with solver S from the cold start (re-solving until
// converged) and compares with the Python solution. Returns true on a match.
template<SolverType S>
bool solveAndCompare(const std::string& mpcConfigPath, const PythonSolution& py)
{
    using Solver = SolverFor<S, OptProblem>;
    using namespace std::chrono;

    std::shared_ptr<Ocp> ocp = std::make_shared<Ocp>(mpcConfigPath);
    ocp->set_initial_state(py.x0);
    std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
    std::shared_ptr<OptProblem> optProblem = std::make_shared<OptProblem>(transcription); // generates the tape
    Solver solver(optProblem);
    configureSolver(solver);

    // Guesses must be set after the solver is constructed (see MultipleShooting::set_X_guess).
    transcription->set_X_guess(coldStartX(py.x0));
    transcription->set_U_guess(InputTrajectory(InputTrajectory::Zero())); // typed: overloads are ambiguous for NU = 1
    transcription->set_p_guess(Ocp::Param::Zero());

    const steady_clock::time_point t0 = steady_clock::now();
    SolveResult result = solveOnce(solver);
    int nSolves = 1;
    while (!result.converged && nSolves < kMaxSolves) {
        result = solveOnce(solver);
        ++nSolves;
    }
    const double ms = duration<double, std::milli>(steady_clock::now() - t0).count();

    const StateTrajectory x = transcription->get_X_opt();
    const InputTrajectory u = transcription->get_U_opt();
    const double obj = ocpObjective(*ocp, x, u, transcription->get_p_opt());
    const double xDiff = (x - py.x).cwiseAbs().maxCoeff();
    const double uDiff = (u - py.u).cwiseAbs().maxCoeff();

    std::cout << "\n[" << solverName<Solver>() << "] " << result.status << " after " << nSolves
              << " solve() calls, " << ms << " ms\n";
    std::cout << "U laopt:  " << u << "\n";
    std::cout << "U Python: " << py.u << "\n";
    std::cout << "objective laopt: " << obj << "  Python: " << py.objective << "\n";
    std::cout << "max |X laopt - X Python| = " << xDiff << ",  max |U laopt - U Python| = " << uDiff << "\n";

    bool ok = true;
    if (!result.converged) {
        std::cerr << "[" << solverName<Solver>() << "] did not converge\n";
        ok = false;
    }
    if (xDiff > 1e-3 || uDiff > 1e-3) {
        std::cerr << "[" << solverName<Solver>() << "] MISMATCH with Python solution\n";
        ok = false;
    }
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    std::cout << std::unitbuf; // unbuffered, so progress is visible even if a later step crashes
    const std::string mpcConfigPath =
        argc > 1 ? argv[1] : std::string(NPMPC_PROJECT_ROOT) + "/model/mpc_config_equation.yaml";
    const std::string pythonSolutionPath =
        argc > 2 ? argv[2] : std::string(NPMPC_PROJECT_ROOT) + "/model/eq_python_solution.yaml";

    const PythonSolution py = loadPythonSolution(pythonSolutionPath);
    Ocp ocp(mpcConfigPath);
    bool ok = true;

    // -- 1. Dynamics: implicit-midpoint residual of the ODE port on the Python solution --
    {
        const double dt = ocp.settings.dt;
        double maxResidual = 0.0;
        for (int k = 0; k < N; ++k) {
            const Ocp::State xMid = 0.5 * (py.x.col(k) + py.x.col(k + 1));
            const Ocp::State f = ocp.furuta_ode<double>(xMid, Ocp::Input(py.u.col(k)));
            const Ocp::State residual = py.x.col(k) + dt * f - py.x.col(k + 1);
            maxResidual = std::max(maxResidual, residual.cwiseAbs().maxCoeff());
        }
        std::cout << "[dynamics] max implicit-midpoint residual on the Python solution: " << maxResidual << "\n";
        if (maxResidual > 1e-5) {
            std::cerr << "[dynamics] MISMATCH: the ODE port does not reproduce the Python dynamics\n";
            ok = false;
        }
    }

    // -- 2. Cost at the Python solution (slack 0) --
    {
        const double obj = ocpObjective(ocp, py.x, py.u, Ocp::Param::Zero());
        const double relDiff = std::abs(obj - py.objective) / std::max(1.0, std::abs(py.objective));
        std::cout.precision(12);
        std::cout << "[cost] Python: " << py.objective << "  laopt OCP: " << obj << "  rel diff: " << relDiff << "\n";
        std::cout.precision(6);
        if (relDiff > 1e-6) {
            std::cerr << "[cost] MISMATCH\n";
            ok = false;
        }
    }

    // -- 3. Solve with both solvers --
    ok = solveAndCompare<SolverType::IPOPT>(mpcConfigPath, py) && ok;
    ok = solveAndCompare<SolverType::SQP_PIQP>(mpcConfigPath, py) && ok;

    std::cout << "\n" << (ok ? "OK: equation-based laopt OCP matches the Python FurutaMPC." : "FAILED") << std::endl;
    return ok ? 0 : 1;
}
