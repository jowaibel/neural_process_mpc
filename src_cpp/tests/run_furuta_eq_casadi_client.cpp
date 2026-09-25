// Connects to the Python Qube simulator server (scripts/run_qube_server.py)
// and drives it in closed loop with the C++ equation-based CasADi MPC
// (FurutaMPC: analytical Furuta ODE); otherwise the same as
// run_furuta_np_casadi_client.cpp (NP dynamics): receive the current state,
// solve, send back the first optimal input, repeat.
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include <casadi/casadi.hpp>

#include "npmpc/mpc/mpc_config_io.hpp"
#include "npmpc/mpc/mpc_service.hpp"
#include "npmpc/mpc/problem/FurutaMPC.hpp"
#include "npmpc/mpc/qube_client.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

int main(int argc, char** argv) {
    const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    const int port = argc > 2 ? std::stoi(argv[2]) : 56123;
    const std::string mpcConfigPath =
        argc > 3 ? argv[3] : std::string(NPMPC_PROJECT_ROOT) + "/model/mpc_config_equation.yaml";

    npmpc::mpc::problem::MPCParams params = npmpc::mpc::loadMPCParamsFromYaml(mpcConfigPath);
    if (params.p.is_empty()) {
        throw std::runtime_error("No plant parameters `p` in " + mpcConfigPath +
                                 " (export it with scripts/export_mpc_config.py --method equation)");
    }

    npmpc::mpc::problem::FurutaMPC mpc(params.p, params.cost);
    npmpc::mpc::MPCService service(mpc, params);

    // Warm-up solve before connecting: the one-time setup of the first solve
    // (NLP solver construction, derivative generation, IPOPT init) happens here,
    // not in the closed loop. The first closed-loop solve still cold-starts.
    {
        const std::chrono::steady_clock::time_point tPrepare0 = std::chrono::steady_clock::now();
        service.prepare(params.x0.T()); // (x_size,1) -> (1, x_size)
        const double prepareMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tPrepare0).count();
        std::cout << "Warm-up solve from the config x0: " << prepareMs << " ms, "
                  << (service.converged() ? "converged" : "NOT converged") << " ("
                  << service.iterCount() << " IPOPT iter)\n";
    }

    std::cout << "Connecting to Qube server at " << host << ":" << port << "...\n";
    npmpc::mpc::QubeClient client(host, port);
    // Fixed run length in simulator time, sim_steps * dt, as in the Python agent
    // (MPCController.run_controller). The server streams states at its own
    // rate, so always solve from the newest one (older queued states are
    // discarded) and start the next solve right after sending.
    const double runTime = params.simSteps * params.dt;
    std::cout << "Connected. Running closed loop (equation MPC, CasADi) for " << runTime << " s.\n";

    npmpc::mpc::QubeState state;
    double tStart = 0.0;
    for (int step = 0;; ++step) {
        int skipped = 0;
        if (!client.receiveLatestState(state, &skipped)) {
            std::cerr << "Server closed the connection at step " << step << ".\n";
            break;
        }
        if (step == 0) {
            tStart = state.t;
        }
        if (state.t - tStart >= runTime) {
            break;
        }

        casadi::DM x0 = casadi::DM(state.x).T(); // (x_size,1) -> (1, x_size)
        service.setInitialState(x0);
        const std::chrono::steady_clock::time_point tSolve0 = std::chrono::steady_clock::now();
        service.solve();
        const double solveMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tSolve0).count();

        casadi::DM u0 = service.firstInput();
        std::vector<double> uOut = u0.nonzeros();
        if (state.predictionRequested) {
            // Open-loop prediction of this solve, only when the server asks for it (logging).
            const npmpc::mpc::MpcPrediction prediction = service.prediction(state.t);
            client.sendInput(uOut, solveMs, &prediction);
        } else {
            client.sendInput(uOut, solveMs);
        }

        std::cout << "step " << step << " | t=" << state.t - tStart << " | skipped " << skipped
                  << " | " << solveMs << " ms | " << (service.converged() ? "converged" : "NOT converged")
                  << " (" << service.iterCount() << " IPOPT iter) | x0=" << x0 << " | u0=" << u0 << "\n";
    }

    return 0;
}
