#pragma once

#include <casadi/casadi.hpp>

#include "npmpc/mpc/problem/MPCBase.hpp"
#include "npmpc/mpc/problem/params.hpp"

namespace npmpc::mpc {

// Port of MPCController (controller.py), minimal version: builds the
// ca.Opti problem (_build_optimization) and can run a single warm-start
// solve (warm_start). `mpc` is injected directly rather than looked up via
// PROBLEM_REGISTRY (a plain factory, not casadi-specific -- out of scope).
//
// TODO: run_controller()'s per-step re-solve loop over FurutaRuntime
// (hardware I/O, Monte Carlo rollouts, data logging) is not ported --
// FurutaRuntime is torch/hardware-specific.
class MPCController {
public:
    MPCController(problem::MPCBase& mpc, problem::MPCParams params);

    // Solves once from a cold start (linear interpolation toward the
    // upright equilibrium), retrying with the last iterate on failure, as
    // in Python's warm_start().
    void warmStart(const casadi::DM& x0);

    [[noreturn]] void runController();

    const casadi::DM& lastX() const { return lastX_; }
    const casadi::DM& lastU() const { return lastU_; }
    const casadi::DM& lastSlack() const { return lastSlack_; }

private:
    void buildOptimization();

    problem::MPCBase& mpc_;
    problem::MPCParams params_;
    casadi::Opti problem_;
    casadi::MX x_, u_, slack_, x0Param_;
    casadi::DM lastX_, lastU_, lastSlack_, lastLamG_;
};

} // namespace npmpc::mpc
