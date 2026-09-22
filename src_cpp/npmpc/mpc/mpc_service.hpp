#pragma once

#include <casadi/casadi.hpp>

#include "npmpc/mpc/controller.hpp"
#include "npmpc/mpc/problem/MPCBase.hpp"
#include "npmpc/mpc/problem/params.hpp"

namespace npmpc::mpc {

// Thin socket-facing wrapper around MPCController: set a new initial state,
// compute the solution, and read back the optimal trajectory or just the
// first input. Each solve() reuses the previous solution as the warm start
// (see MPCController::warmStart).
class MPCService {
public:
    MPCService(problem::MPCBase& mpc, problem::MPCParams params) : controller_(mpc, std::move(params)) {}

    void setInitialState(const casadi::DM& x0) { x0_ = x0; }
    void solve() { controller_.warmStart(x0_); }

    // (1, u_size) first optimal input, to be applied this control step.
    casadi::DM firstInput() const { return controller_.lastU()(0, casadi::Slice()); }

    // (horizon+1, x_size) optimal state trajectory.
    casadi::DM optimalTrajectory() const { return controller_.lastX(); }

    // (horizon, u_size) optimal input trajectory.
    casadi::DM optimalInputs() const { return controller_.lastU(); }

private:
    MPCController controller_;
    casadi::DM x0_;
};

} // namespace npmpc::mpc
