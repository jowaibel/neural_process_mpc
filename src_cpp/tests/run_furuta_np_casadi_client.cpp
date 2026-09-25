// Connects to the Python Qube simulator server (scripts/run_qube_server.py)
// and drives it in closed loop with the C++ NP-dynamics MPC: receive the
// current state, solve, send back the first optimal input, repeat.
#include <chrono>
#include <iostream>
#include <string>

#include <casadi/casadi.hpp>

#include "npmpc/mpc/mpc_config_io.hpp"
#include "npmpc/mpc/mpc_service.hpp"
#include "npmpc/mpc/problem/FurutaNPMPC.hpp"
#include "npmpc/mpc/qube_client.hpp"
#include "npmpc/nps/weights_io.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

int main(int argc, char** argv) {
    const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    const int port = argc > 2 ? std::stoi(argv[2]) : 56123;
    const std::string weightsPath =
        argc > 3 ? argv[3] : std::string(NPMPC_PROJECT_ROOT) + "/model/np_weights.yaml";
    const std::string mpcConfigPath =
        argc > 4 ? argv[4] : std::string(NPMPC_PROJECT_ROOT) + "/model/mpc_config.yaml";

    npmpc::nps::NeuralProcess np = npmpc::nps::loadNeuralProcessFromYaml(weightsPath);
    npmpc::mpc::problem::MPCParams params = npmpc::mpc::loadMPCParamsFromYaml(mpcConfigPath);

    npmpc::mpc::problem::FurutaNPMPC mpc(&np, params.z, params.cost);
    npmpc::mpc::MPCService service(mpc, params);

    std::cout << "Connecting to Qube server at " << host << ":" << port << "...\n";
    npmpc::mpc::QubeClient client(host, port);
    // Fixed run length in simulator time, sim_steps * dt, as in the Python agent
    // (MPCController.run_controller). The server streams states at its own
    // rate, so always solve from the newest one (older queued states are
    // discarded) and start the next solve right after sending.
    const double runTime = params.simSteps * params.dt;
    std::cout << "Connected. Running closed loop for " << runTime << " s.\n";

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
        client.sendInput(uOut, solveMs);

        std::cout << "step " << step << " | t=" << state.t - tStart << " | skipped " << skipped
                  << " | " << solveMs << " ms | x0=" << x0 << " | u0=" << u0 << "\n";
    }

    return 0;
}
