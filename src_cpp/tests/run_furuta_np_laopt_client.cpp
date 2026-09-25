// Connects to the Python Qube simulator server (scripts/run_qube_server.py)
// and drives it in closed loop with the laopt NP MPC (FurutaNPOcpEigen.hpp),
// the laopt counterpart of run_furuta_np_casadi_client.cpp.
//
// The loop runs as fast as possible: as soon as a solve is done and its first
// input is sent, the next solve starts from the newest state measurement
// (older states still queued in the socket are discarded; if none is queued
// yet, it waits for the next one). There is no sleep and no retry: whatever
// the solver returns (converged or not, e.g. SQP at its iteration limit),
// U.col(0) is sent. The problem is taped once; each solve is warm-started
// from the previous solution. Only the very first solve is repeated until it
// converges (up to kMaxInitialSolves calls) before its input is sent.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "npmpc/mpc/laopt/multiple_shooting_xdiff.hpp"

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
using Transcription = laopt_tools::MultipleShootingXDiff<Ocp, N, laopt::ERK4>; // integrator unused (DiscreteDynamics)
using OptProblem = laopt::Problem<Transcription>;
using Solver = npmpc::mpc::furuta_laopt::SolverFor<kSolver, OptProblem>;

// Cap on solve() calls for the initial solve (repeated until converged before
// the first input is sent).
constexpr int kMaxInitialSolves = 500;

// Artificial computation delay, to compare with slower MPCs (e.g. CasADi/IPOPT):
// the input is sent no earlier than kMinSolveMs after the solve started
// (waiting if the solve was faster). 0 = send as soon as the solve is done.
// The logged solve_ms stays the actual solver time.
constexpr double kMinSolveMs = 5.0;

constexpr int NXP = Ocp::NX;                            // state size
using StateTrajectory = Transcription::StateTrajectory; // (NX, N+1)
using InputTrajectory = Transcription::InputTrajectory; // (NU, N)

// Cold-start input guess: constant at the lower bound (negative u drives
// positive theta_dot), to leave the symmetric hanging position where the
// Gauss-Newton SQP sees no gradient.
InputTrajectory coldStartU(const Ocp& ocp)
{
    return InputTrajectory::Constant(ocp.settings.uLb(0));
}

// Cold-start state guess: the NP rollout of the input guess from x0, so the
// guess satisfies the dynamics constraints exactly (instead of MPCController's
// linear interpolation to [2pi,0,0,0], which violates them).
StateTrajectory coldStartX(const Ocp& ocp, const Ocp::State& x0, const InputTrajectory& u)
{
    return ocp.model.rollout<N>(x0, u, ocp.settings.z, ocp.settings.dt);
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
        const Ocp::State x = Eigen::Map<const Ocp::State>(state.x.data());

        if (nSolves == 0) {
            tStart = state.t;
            // Cold start (dynamically consistent: X = NP rollout of U from x);
            // guesses must be set after the solver is constructed.
            const InputTrajectory uGuess = coldStartU(*ocp);
            transcription->set_X_guess(coldStartX(*ocp, x, uGuess));
            transcription->set_U_guess(uGuess); // typed as a trajectory: the overloads are ambiguous for NU = 1
            transcription->set_p_guess(Ocp::Param::Zero());
        }
        if (state.t - tStart >= runTime) {
            break;
        }

        ocp->set_initial_state(x);

        const steady_clock::time_point tSolve0 = steady_clock::now();
        npmpc::mpc::furuta_laopt::SolveResult result = npmpc::mpc::furuta_laopt::solveOnce(solver);
        if (nSolves == 0) {
            // Before the first input is sent, re-solve (each call continues from
            // the previous iterate) until converged, so the closed loop starts
            // from a converged solution. The pendulum keeps hanging meanwhile.
            int nInitialSolves = 1;
            while (!result.converged && nInitialSolves < kMaxInitialSolves) {
                result = npmpc::mpc::furuta_laopt::solveOnce(solver);
                ++nInitialSolves;
            }
            std::cout << "Initial solve: " << nInitialSolves << " solve() calls, "
                      << (result.converged ? "converged" : "NOT converged (cap reached)") << ", "
                      << duration<double, std::milli>(steady_clock::now() - tSolve0).count() << " ms\n";
        }
        const double solveMs = duration<double, std::milli>(steady_clock::now() - tSolve0).count();
        if (solveMs < kMinSolveMs) {
            std::this_thread::sleep_until(tSolve0 + duration<double, std::milli>(kMinSolveMs));
        }
        const double sendMs = duration<double, std::milli>(steady_clock::now() - tSolve0).count();

        // First input, sent whether or not the solver converged.
        const Ocp::Input u0 = transcription->get_U_opt().col(0);
        const std::vector<double> uOut(u0.data(), u0.data() + u0.size());
        if (state.predictionRequested) {
            // Open-loop prediction of this solve, only when the server asks for it (logging).
            const npmpc::mpc::MpcPrediction prediction =
                npmpc::mpc::furuta_laopt::predictionOf(*transcription, state.t);
            client.sendInput(uOut, solveMs, &prediction);
        } else {
            client.sendInput(uOut, solveMs);
        }

        ++nSolves;
        nConverged += result.converged ? 1 : 0;
        nSkippedTotal += skipped;
        solveMsSum += solveMs;
        solveMsMax = std::max(solveMsMax, solveMs);

        std::cout << "solve " << nSolves << " | t=" << state.t - tStart << " | skipped " << skipped
                  << " | " << solveMs << " ms (sent after " << sendMs << " ms) | " << result.status
                  << " | x0=" << x.transpose() << " | u0=" << u0.transpose() << "\n";
    }

    if (nSolves > 0) {
        std::cout << "\nSolves: " << nSolves << " (" << nConverged << " converged)"
                  << " | solve time mean " << solveMsSum / nSolves << " ms, max " << solveMsMax << " ms"
                  << " | skipped states: " << nSkippedTotal << std::endl;
    }
    return 0;
}
