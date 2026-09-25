#include "npmpc/mpc/controller.hpp"

#include <cmath>
#include <stdexcept>

#include "npmpc/mpc/solver.hpp"

namespace npmpc::mpc {

namespace {
// Max solve() attempts per warmStart: on failure, retry from the last iterate
// (Python's warm_start uses 100).
constexpr int kMaxSolveAttempts = 1;
} // namespace

MPCController::MPCController(problem::MPCBase& mpc, problem::MPCParams params)
    : mpc_(mpc), params_(std::move(params)) {
    buildOptimization();
    configureSolver(problem_, params_.solver);
}

void MPCController::buildOptimization() {
    const int N = params_.horizonSteps;
    const int xSize = params_.xSize;
    const int uSize = params_.uSize;

    x_ = problem_.variable(N + 1, xSize);
    u_ = problem_.variable(N, uSize);
    slack_ = problem_.variable(1, xSize);
    x0Param_ = problem_.parameter(1, xSize);

    // Initial state.
    problem_.subject_to(x_(0, casadi::Slice()) <= x0Param_ + 1e-3);
    problem_.subject_to(x_(0, casadi::Slice()) >= x0Param_ - 1e-3);

    // Constraints.
    for (const auto& c : mpc_.integrationConstraints(x_, u_, params_.dt)) {
        problem_.subject_to(c);
    }
    for (const auto& c : mpc_.stateConstraints(x_, slack_, params_.hardBound, params_.slackBoundX)) {
        problem_.subject_to(c);
    }
    for (const auto& c : mpc_.inputConstraints(u_, params_.hardBound)) {
        problem_.subject_to(c);
    }
    for (const auto& c : mpc_.slackConstraints(slack_, params_.slackBoundX)) {
        problem_.subject_to(c);
    }

    // Cost.
    casadi::MX cost = mpc_.costFunction(x_, u_) + mpc_.slackCost(slack_, params_.slackBoundX);
    problem_.minimize(cost);
}

void MPCController::setColdStartGuess(const casadi::DM& x0) {
    // Linearly interpolate x toward the upright equilibrium, zero u -- as in
    // Python's warm_start().
    const int N = params_.horizonSteps;
    casadi::DM uInit = casadi::DM::zeros(N, params_.uSize);
    casadi::DM target = casadi::DM({2.0 * M_PI, 0.0, 0.0, 0.0}).T();
    casadi::DM xInit = casadi::DM::zeros(N + 1, params_.xSize);
    for (casadi_int i = 0; i <= N; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(N);
        xInit(i, casadi::Slice()) = x0 + t * (target - x0);
    }
    problem_.set_initial(x_, xInit);
    problem_.set_initial(u_, uInit);
}

void MPCController::prepare(const casadi::DM& x0) {
    // One cold-start solve so that Opti builds the NLP solver, CasADi
    // generates the derivative functions and IPOPT initializes -- all done
    // lazily on the first solve() and cached afterwards.
    problem_.set_value(x0Param_, x0);
    setColdStartGuess(x0);
    solveWithRetry();

    // Forget the solution, so that the next warmStart cold-starts as before.
    lastX_ = casadi::DM();
    lastU_ = casadi::DM();
    lastSlack_ = casadi::DM();
    lastLamG_ = casadi::DM();
}

void MPCController::warmStart(const casadi::DM& x0) {
    problem_.set_value(x0Param_, x0);

    if (lastX_.is_empty()) {
        setColdStartGuess(x0);
    } else {
        // Warm start from the previous solution, reused as-is (no
        // shifting/delay compensation).
        problem_.set_initial(x_, lastX_);
        problem_.set_initial(u_, lastU_);
        problem_.set_initial(problem_.lam_g(), lastLamG_);
    }

    solveWithRetry();
}

void MPCController::solveWithRetry() {
    bool solved = false;
    lastIterCount_ = 0;
    // Adds the IPOPT iterations of the last solve attempt (converged or not).
    auto addIterations = [this]() {
        try {
            const casadi::Dict stats = problem_.stats();
            auto it = stats.find("iter_count");
            if (it != stats.end()) {
                lastIterCount_ += static_cast<int>(it->second.to_int());
            }
        } catch (const std::exception&) {
            // no stats available: leave the count as is
        }
    };
    for (int attempt = 0; attempt < kMaxSolveAttempts && !solved; ++attempt) {
        try {
            casadi::OptiSol sol = problem_.solve();
            addIterations();
            lastX_ = sol.value(x_);
            lastU_ = sol.value(u_);
            lastSlack_ = sol.value(slack_);
            lastLamG_ = sol.value(problem_.lam_g());
            solved = true;
        } catch (const std::exception&) {
            addIterations();
            casadi::OptiAdvanced debug = problem_.debug();
            problem_.set_initial(x_, debug.value(x_));
            problem_.set_initial(u_, debug.value(u_));
            problem_.set_initial(slack_, debug.value(slack_));
            problem_.set_initial(problem_.lam_g(), debug.value(problem_.lam_g()));
            // Keep the last iterate: used as the solution if no attempt converges.
            lastX_ = debug.value(x_);
            lastU_ = debug.value(u_);
            lastSlack_ = debug.value(slack_);
            lastLamG_ = debug.value(problem_.lam_g());
        }
    }
    // If no attempt converged, the last (unconverged) iterate is used, as in
    // the laopt clients (no exception).
    lastConverged_ = solved;
}

void MPCController::runController() {
    throw std::logic_error(
        "MPCController::runController is not ported: the per-step re-solve "
        "loop over FurutaRuntime (hardware I/O, Monte Carlo rollouts, data "
        "logging) is torch/hardware-specific and out of scope for this port.");
}

} // namespace npmpc::mpc
