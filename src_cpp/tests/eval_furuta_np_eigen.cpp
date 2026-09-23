// Validates FurutaNPEigen (the numeric Eigen port) against the CasADi
// NeuralProcess/FurutaNPIntegration path (eval_furuta_np.cpp) for the same
// weights, initial state, inputs and latent code: the two rollouts must
// agree to numerical precision.
#include <cmath>
#include <iostream>
#include <string>

#include <casadi/casadi.hpp>
#include <Eigen/Dense>

#include "npmpc/dynamics/integrators/furuta_np.hpp"
#include "npmpc/nps/FurutaNPEigen.hpp"
#include "npmpc/nps/weights_io.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

using npmpc::nps::FurutaNPEigen;

int main(int argc, char** argv) {
    const std::string weightsPath = argc > 1 ? argv[1] : std::string(NPMPC_PROJECT_ROOT) + "/model/np_weights.yaml";

    constexpr int nSteps = 5;
    constexpr int xDim = FurutaNPEigen::kStateDim;
    constexpr int uDim = FurutaNPEigen::kInputDim;
    constexpr int zDim = FurutaNPEigen::kLatentDim;
    const double dt = 0.01;

    // -- CasADi rollout (ground truth) --
    npmpc::nps::NeuralProcess npCasadi = npmpc::nps::loadNeuralProcessFromYaml(weightsPath);
    npmpc::dynamics::integrators::FurutaNPIntegration integrator(&npCasadi);

    casadi::MX x0Sym = casadi::MX::sym("x0", 1, xDim);
    casadi::MX uSym = casadi::MX::sym("u", nSteps, uDim);
    casadi::MX zSym = casadi::MX::sym("z", 1, zDim);
    casadi::MX xTraj = integrator.casadiExplicit(x0Sym, uSym, dt, zSym);
    casadi::Function rollout("furuta_np_rollout", {x0Sym, uSym, zSym}, {xTraj});

    casadi::DM x0Val = casadi::DM::zeros(1, xDim);
    x0Val(0, 0) = 0.1;
    casadi::DM uVal = casadi::DM::ones(nSteps, uDim) * 0.5;
    casadi::DM zVal = casadi::DM({-1.0, 0.5, 2.0, -0.5}).T();

    casadi::DM xTrajCasadi = rollout(std::vector<casadi::DM>{x0Val, uVal, zVal}).at(0);

    // -- Eigen rollout (same weights, inputs, latent code) --
    FurutaNPEigen npEigen = FurutaNPEigen::fromYaml(weightsPath);

    FurutaNPEigen::State<double> x0(0.1, 0.0, 0.0, 0.0);
    FurutaNPEigen::InputTrajectory<double, nSteps> u =
        FurutaNPEigen::InputTrajectory<double, nSteps>::Constant(0.5);
    FurutaNPEigen::Latent<double> z(-1.0, 0.5, 2.0, -0.5);

    // Columns are time steps (CasADi trajectory rows are time steps).
    FurutaNPEigen::StateTrajectory<double, nSteps> xTrajEigen = npEigen.rollout<nSteps>(x0, u, z, dt);

    // -- Compare --
    std::cout << "CasADi trajectory:\n" << xTrajCasadi << "\n\n";
    std::cout << "Eigen trajectory (transposed):\n" << xTrajEigen.transpose() << "\n\n";

    double maxAbsDiff = 0.0;
    for (int i = 0; i <= nSteps; ++i) {
        for (int j = 0; j < xDim; ++j) {
            double diff = std::abs(static_cast<double>(xTrajCasadi(i, j)) - xTrajEigen(j, i));
            maxAbsDiff = std::max(maxAbsDiff, diff);
        }
    }
    std::cout << "Max abs diff: " << maxAbsDiff << std::endl;

    if (maxAbsDiff > 1e-9) {
        std::cerr << "MISMATCH: Eigen port disagrees with CasADi reference." << std::endl;
        return 1;
    }
    std::cout << "OK: Eigen port matches CasADi reference." << std::endl;
    return 0;
}
