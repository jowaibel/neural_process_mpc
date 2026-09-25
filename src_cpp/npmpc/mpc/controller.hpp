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

    // Warm-up before the closed loop starts: one cold-start solve from x0, so
    // that the one-time setup of the first solve (Opti builds the NLP solver,
    // CasADi generates the derivative functions, IPOPT initializes) happens
    // here. The solution is then discarded, so the next warmStart cold-starts
    // as without prepare(). lastConverged()/lastIterCount() report this solve.
    void prepare(const casadi::DM& x0);

    [[noreturn]] void runController();

    const casadi::DM& lastX() const { return lastX_; }
    const casadi::DM& lastU() const { return lastU_; }
    const casadi::DM& lastSlack() const { return lastSlack_; }

    // Of the last warmStart: whether a solve attempt converged, and the IPOPT
    // iterations summed over all its attempts.
    bool lastConverged() const { return lastConverged_; }
    int lastIterCount() const { return lastIterCount_; }

private:
    void buildOptimization();
    void setColdStartGuess(const casadi::DM& x0);
    void solveWithRetry();

    problem::MPCBase& mpc_;
    problem::MPCParams params_;
    casadi::Opti problem_;
    casadi::MX x_, u_, slack_, x0Param_;
    casadi::DM lastX_, lastU_, lastSlack_, lastLamG_;
    bool lastConverged_ = false;
    int lastIterCount_ = 0;
};

} // namespace npmpc::mpc
