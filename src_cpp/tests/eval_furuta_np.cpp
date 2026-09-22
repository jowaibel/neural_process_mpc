// Minimal evaluation test: load a trained NeuralProcess from YAML, wrap it in
// FurutaNPIntegration, and roll out a short trajectory for some fixed test
// numbers using the CasADi explicit integrator.
#include <iostream>
#include <string>

#include <casadi/casadi.hpp>

#include "npmpc/dynamics/integrators/furuta_np.hpp"
#include "npmpc/nps/weights_io.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

int main(int argc, char** argv) {
    const std::string weightsPath = argc > 1 ? argv[1] : std::string(NPMPC_PROJECT_ROOT) + "/model/np_weights.yaml";

    npmpc::nps::NeuralProcess np = npmpc::nps::loadNeuralProcessFromYaml(weightsPath);
    npmpc::dynamics::integrators::FurutaNPIntegration integrator(&np);

    const int nSteps = 5;
    const int xDim = 4;   // [theta, phi, theta_dot, phi_dot]
    const int uDim = 1;   // motor torque
    const int zDim = 4;   // NP latent

    casadi::MX x0 = casadi::MX::sym("x0", 1, xDim);
    casadi::MX u = casadi::MX::sym("u", nSteps, uDim);
    casadi::MX z = casadi::MX::sym("z", 1, zDim);

    casadi::MX xTraj = integrator.casadiExplicit(x0, u, /*dt=*/0.01, z);
    casadi::Function rollout("furuta_np_rollout", {x0, u, z}, {xTraj});

    casadi::DM x0Val = casadi::DM::zeros(1, xDim);
    x0Val(0, 0) = 0.1; // small initial pendulum angle

    casadi::DM uVal = casadi::DM::ones(nSteps, uDim) * 0.5;
    casadi::DM zVal = casadi::DM::zeros(1, zDim);

    std::vector<casadi::DM> result = rollout(std::vector<casadi::DM>{x0Val, uVal, zVal});

    std::cout << "x0:  " << x0Val << "\n";
    std::cout << "u:   " << uVal << "\n";
    std::cout << "z:   " << zVal << "\n";
    std::cout << "x trajectory (rows = time steps):\n" << result.at(0) << std::endl;

    return 0;
}
