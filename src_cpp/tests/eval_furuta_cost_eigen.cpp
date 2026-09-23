// Validates furutaCostEigen (the Eigen port) against the CasADi furutaCost
// for the cost weights in model/mpc_config.yaml, evaluated on the same
// NP-rolled-out trajectory: the two costs must agree to numerical precision.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include <casadi/casadi.hpp>
#include <Eigen/Dense>

#include "npmpc/mpc/mpc_config_io.hpp"
#include "npmpc/mpc/problem/furuta_cost.hpp"
#include "npmpc/mpc/problem/furuta_cost_eigen.hpp"
#include "npmpc/nps/FurutaNPEigen.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

using npmpc::nps::FurutaNPEigen;

int main(int argc, char** argv) {
    const std::string weightsPath =
        argc > 1 ? argv[1] : std::string(NPMPC_PROJECT_ROOT) + "/model/np_weights.yaml";
    const std::string mpcConfigPath =
        argc > 2 ? argv[2] : std::string(NPMPC_PROJECT_ROOT) + "/model/mpc_config.yaml";

    constexpr int nSteps = 5;
    constexpr int xDim = FurutaNPEigen::kStateDim;
    constexpr int uDim = FurutaNPEigen::kInputDim;
    const double dt = 0.01;

    npmpc::mpc::problem::MPCParams params = npmpc::mpc::loadMPCParamsFromYaml(mpcConfigPath);
    npmpc::mpc::problem::CostParamsEigen costEigen = npmpc::mpc::problem::toCostParamsEigen(params.cost);

    // -- Test trajectory: NP rollout with a non-constant input sequence --
    FurutaNPEigen np = FurutaNPEigen::fromYaml(weightsPath);
    FurutaNPEigen::State<double> x0(0.1, 0.2, -0.3, 0.4);
    FurutaNPEigen::InputTrajectory<double, nSteps> u;
    u << 0.5, -0.2, 0.1, 0.3, -0.4;
    FurutaNPEigen::Latent<double> z(-1.0, 0.5, 2.0, -0.5);
    FurutaNPEigen::StateTrajectory<double, nSteps> x = np.rollout<nSteps>(x0, u, z, dt);

    // -- Eigen cost --
    double costEigenVal = npmpc::mpc::problem::furutaCostEigen<nSteps>(x, u, costEigen);

    // -- CasADi cost (ground truth), rows = time steps --
    casadi::MX xSym = casadi::MX::sym("x", nSteps + 1, xDim);
    casadi::MX uSym = casadi::MX::sym("u", nSteps, uDim);
    casadi::Function costFn("furuta_cost", {xSym, uSym},
                            {npmpc::mpc::problem::furutaCost(xSym, uSym, params.cost)});

    casadi::DM xVal = casadi::DM::zeros(nSteps + 1, xDim);
    casadi::DM uVal = casadi::DM::zeros(nSteps, uDim);
    for (int k = 0; k <= nSteps; ++k) {
        for (int j = 0; j < xDim; ++j) {
            xVal(k, j) = x(j, k);
        }
    }
    for (int k = 0; k < nSteps; ++k) {
        for (int j = 0; j < uDim; ++j) {
            uVal(k, j) = u(j, k);
        }
    }
    double costCasadiVal = static_cast<double>(costFn(std::vector<casadi::DM>{xVal, uVal}).at(0));

    // -- Compare --
    std::cout.precision(17);
    std::cout << "CasADi cost: " << costCasadiVal << "\n";
    std::cout << "Eigen cost:  " << costEigenVal << "\n";

    double relDiff = std::abs(costCasadiVal - costEigenVal) / std::max(1.0, std::abs(costCasadiVal));
    std::cout << "Rel diff: " << relDiff << std::endl;

    if (relDiff > 1e-12) {
        std::cerr << "MISMATCH: Eigen cost disagrees with CasADi reference." << std::endl;
        return 1;
    }
    std::cout << "OK: Eigen cost matches CasADi reference." << std::endl;
    return 0;
}
