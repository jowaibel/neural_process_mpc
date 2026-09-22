#pragma once

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

namespace npmpc::mpc::problem {

// One [lower, upper] hard bound; +-infinity means "unbounded" (skip the
// constraint), matching how `hard_bound`/`slack_bound` are used in the
// Python params dict.
struct Bound {
    double lower;
    double upper;
};

struct HardBound {
    std::vector<Bound> x;
    std::vector<Bound> u;
};

// Port of the `cost` sub-dict consumed by furuta_cost(). terminalP is the
// LQR terminal-cost matrix (FurutaNPMPC.compute_terminal_P() in Python);
// this port treats it as a precomputed constant, see
// scripts/export_mpc_config.py.
struct CostParams {
    std::vector<double> x;
    std::vector<double> xDiff;
    std::vector<double> xEnd;
    std::vector<double> u;
    casadi::DM terminalP;
};

// Port of the params dict consumed by MPCController / FurutaNPMPC, trimmed
// to what the CasADi-only path needs.
struct MPCParams {
    double dt;
    int horizonSteps;
    int xSize;
    int uSize;
    std::string solver;
    casadi::DM z;                  // NP latent code (casadi_decoder input)
    HardBound hardBound;
    std::vector<double> slackBoundX;
    CostParams cost;
    casadi::DM x0;
    int simSteps;
};

} // namespace npmpc::mpc::problem
