// Connects to the Python Qube simulator server (scripts/run_qube_server.py)
// and drives it in closed loop with the laopt NP MPC (FurutaNPOcpEigen.hpp),
// the laopt counterpart of run_furuta_np_mpc_client.cpp.
//
// The loop runs as fast as possible: as soon as a solve is done and its first
// input is sent, the next solve starts from the newest state measurement
// (older states still queued in the socket are discarded; if none is queued
// yet, it waits for the next one). There is no sleep and no retry: whatever
// the solver returns (converged or not, e.g. SQP at its iteration limit),
// U.col(0) is sent. The problem is taped once; each solve is warm-started
// from the previous solution.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/multiple_shooting.hpp"

#include "npmpc/mpc/laopt/FurutaNPOcpEigen.hpp"
#include "npmpc/mpc/laopt/laopt_solver.hpp"
#include "npmpc/mpc/qube_client.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

namespace {

constexpr int N = 12; // must match horizon_steps in mpc_config.yaml

/* Solver switch: IPOPT or SQP with PIQP as QP solver (settings in laopt_solver.hpp). */
using npmpc::mpc::furuta_laopt::SolverType;

// constexpr SolverType kSolver = SolverType::IPOPT;
constexpr SolverType kSolver = SolverType::SQP_PIQP;

using Ocp = npmpc::mpc::furuta_laopt::FurutaNPOCP<N>;
using Transcription = laopt_tools::MultipleShooting<Ocp, N, laopt::ERK4>; // integrator unused (DiscreteDynamics)
using OptProblem = laopt::Problem<Transcription>;
using Solver = npmpc::mpc::furuta_laopt::SolverFor<kSolver, OptProblem>;

constexpr int NXP = npmpc::mpc::furuta_laopt::kNX;      // physical state size (Ocp::NX = 2 * NXP: [x; d])
using StateTrajectory = Ocp::PhysStateTrajectory;       // physical states (NXP, N+1)
using InputTrajectory = Transcription::InputTrajectory; // (NU, N)

// Cold start as in MPCController::warmStart: x linear from x0 to [2pi,0,0,0], u = 0.
StateTrajectory coldStartX(const Ocp::PhysState& x0)
{
    const Ocp::PhysState target(2.0 * M_PI, 0.0, 0.0, 0.0);
    StateTrajectory x;
    for (int i = 0; i <= N; ++i) {
        const double t = static_cast<double>(i) / N;
        x.col(i) = x0 + t * (target - x0);
    }
    return x;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace std::chrono;

    const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    const int port = argc > 2 ? std::stoi(argv[2]) : 56123;
    const std::string weightsPath =
        argc > 3 ? argv[3] : std::string(NPMPC_PROJECT_ROOT) + "/model/np_weights.yaml";
    const std::string mpcConfigPath =
        argc > 4 ? argv[4] : std::string(NPMPC_PROJECT_ROOT) + "/model/mpc_config.yaml";

    /* Build OCP, transcription, problem (tape) and solver once */
    std::shared_ptr<Ocp> ocp = std::make_shared<Ocp>(weightsPath, mpcConfigPath);
    std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
    std::shared_ptr<OptProblem> optProblem = std::make_shared<OptProblem>(transcription); // generates the tape
    Solver solver(optProblem);
    npmpc::mpc::furuta_laopt::configureSolver(solver);

    // Fixed run length in simulator time, sim_steps * dt, as in the Python agent
    // (MPCController.run_controller): the same simulated span for every solver,
    // however many solves each manages in it.
    const double runTime = ocp->settings.simSteps * ocp->settings.dt;

    std::cout << "Connecting to Qube server at " << host << ":" << port << "...\n";
    npmpc::mpc::QubeClient client(host, port);
    std::cout << "Connected. Running closed loop (" << npmpc::mpc::furuta_laopt::solverName<Solver>()
              << ") for " << runTime << " s.\n";

    npmpc::mpc::QubeState state;
    double tStart = 0.0;
    int nSolves = 0;
    int nConverged = 0;
    int nSkippedTotal = 0;
    double solveMsSum = 0.0;
    double solveMsMax = 0.0;

    while (true) {
        int skipped = 0;
        if (!client.receiveLatestState(state, &skipped)) {
            std::cerr << "Server closed the connection after " << nSolves << " solves.\n";
            break;
        }
        if (state.x.size() != static_cast<size_t>(NXP)) {
            std::cerr << "Received state of size " << state.x.size() << ", expected " << NXP << ".\n";
            return 1;
        }
        const Ocp::PhysState x = Eigen::Map<const Ocp::PhysState>(state.x.data());

        if (nSolves == 0) {
            tStart = state.t;
            // Cold start; guesses must be set after the solver is constructed.
            transcription->set_X_guess(Ocp::augment(coldStartX(x)));
            transcription->set_U_guess(InputTrajectory(InputTrajectory::Zero())); // typed: overloads are ambiguous for NU = 1
            transcription->set_p_guess(Ocp::Param::Zero());
        }
        if (state.t - tStart >= runTime) {
            break;
        }

        ocp->set_initial_state(x);

        const steady_clock::time_point tSolve0 = steady_clock::now();
        const npmpc::mpc::furuta_laopt::SolveResult result = npmpc::mpc::furuta_laopt::solveOnce(solver);
        const double solveMs = duration<double, std::milli>(steady_clock::now() - tSolve0).count();

        // First input, sent whether or not the solver converged.
        const Ocp::Input u0 = transcription->get_U_opt().col(0);
        client.sendInput(std::vector<double>(u0.data(), u0.data() + u0.size()), solveMs);

        ++nSolves;
        nConverged += result.converged ? 1 : 0;
        nSkippedTotal += skipped;
        solveMsSum += solveMs;
        solveMsMax = std::max(solveMsMax, solveMs);

        std::cout << "solve " << nSolves << " | t=" << state.t - tStart << " | skipped " << skipped
                  << " | " << solveMs << " ms | " << result.status
                  << " | x0=" << x.transpose() << " | u0=" << u0.transpose() << "\n";
    }

    if (nSolves > 0) {
        std::cout << "\nSolves: " << nSolves << " (" << nConverged << " converged)"
                  << " | solve time mean " << solveMsSum / nSolves << " ms, max " << solveMsMax << " ms"
                  << " | skipped states: " << nSkippedTotal << std::endl;
    }
    return 0;
}
