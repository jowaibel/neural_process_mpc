// Minimal evaluation test: build the NP-dynamics MPC problem (FurutaNPMPC),
// wrap it in MPCController, and solve one warm-start NLP for the x0 in
// model/mpc_config.yaml.
#include <iostream>
#include <string>

#include <casadi/casadi.hpp>

#include "npmpc/mpc/controller.hpp"
#include "npmpc/mpc/mpc_config_io.hpp"
#include "npmpc/mpc/problem/FurutaNPMPC.hpp"
#include "npmpc/nps/weights_io.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

int main(int argc, char** argv) {
    const std::string weightsPath =
        argc > 1 ? argv[1] : std::string(NPMPC_PROJECT_ROOT) + "/model/np_weights.yaml";
    const std::string mpcConfigPath =
        argc > 2 ? argv[2] : std::string(NPMPC_PROJECT_ROOT) + "/model/mpc_config.yaml";

    npmpc::nps::NeuralProcess np = npmpc::nps::loadNeuralProcessFromYaml(weightsPath);
    npmpc::mpc::problem::MPCParams params = npmpc::mpc::loadMPCParamsFromYaml(mpcConfigPath);

    npmpc::mpc::problem::FurutaNPMPC mpc(&np, params.z, params.cost);
    npmpc::mpc::MPCController controller(mpc, params);

    casadi::DM x0 = params.x0.T(); // (x_size,1) -> (1, x_size)
    controller.warmStart(x0);

    std::cout << "x0: " << x0 << "\n";
    std::cout << "x trajectory (rows = time steps):\n" << controller.lastX() << "\n";
    std::cout << "u trajectory:\n" << controller.lastU() << std::endl;

    return 0;
}
