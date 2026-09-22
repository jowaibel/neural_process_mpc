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

    // Solves for a new x0. On the first call, cold-starts (linear
    // interpolation toward the upright equilibrium, zero u) as in Python's
    // warm_start(); on later calls, reuses the previous solution as the
    // initial guess (no shifting/delay compensation -- TODO: port
    // SolutionBuffer's shift + delay-compensation logic). Either way,
    // retries with the last iterate on solve failure, as in Python.
    void warmStart(const casadi::DM& x0);

    [[noreturn]] void runController();

    const casadi::DM& lastX() const { return lastX_; }
    const casadi::DM& lastU() const { return lastU_; }
    const casadi::DM& lastSlack() const { return lastSlack_; }

private:
    void buildOptimization();
    void solveWithRetry();

    problem::MPCBase& mpc_;
    problem::MPCParams params_;
    casadi::Opti problem_;
    casadi::MX x_, u_, slack_, x0Param_;
    casadi::DM lastX_, lastU_, lastSlack_, lastLamG_;
};

} // namespace npmpc::mpc
