// Connects to the Python Qube simulator server (scripts/run_qube_server.py)
// and drives it in closed loop with the C++ NP-dynamics MPC: receive the
// current state, solve, send back the first optimal input, repeat.
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
    std::cout << "Connected. Running closed loop for " << params.simSteps << " steps.\n";

    npmpc::mpc::QubeState state;
    for (int step = 0; step < params.simSteps; ++step) {
        if (!client.receiveState(state)) {
            std::cerr << "Server closed the connection at step " << step << ".\n";
            break;
        }

        casadi::DM x0 = casadi::DM(state.x).T(); // (x_size,1) -> (1, x_size)
        service.setInitialState(x0);
        service.solve();

        casadi::DM u0 = service.firstInput();
        std::vector<double> uOut = u0.nonzeros();
        client.sendInput(uOut);

        std::cout << "step " << step << " | t=" << state.t << " | x0=" << x0 << " | u0=" << u0 << "\n";
    }

    return 0;
}
