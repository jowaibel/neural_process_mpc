#pragma once

#include <casadi/casadi.hpp>
#include <vector>

#include "npmpc/mpc/controller.hpp"
#include "npmpc/mpc/qube_client.hpp"
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

    // Warm-up solve from x0 before the closed loop (see MPCController::prepare).
    void prepare(const casadi::DM& x0) { controller_.prepare(x0); }

    // (1, u_size) first optimal input, to be applied this control step.
    casadi::DM firstInput() const { return controller_.lastU()(0, casadi::Slice()); }

    // (horizon+1, x_size) optimal state trajectory.
    casadi::DM optimalTrajectory() const { return controller_.lastX(); }

    // (horizon, u_size) optimal input trajectory.
    casadi::DM optimalInputs() const { return controller_.lastU(); }

    // Of the last solve(): converged, and IPOPT iterations (see MPCController).
    bool converged() const { return controller_.lastConverged(); }
    int iterCount() const { return controller_.lastIterCount(); }

    // Open-loop prediction of the last solve (for logging via QubeClient::sendInput);
    // tState is the server time of the state the solve started from.
    MpcPrediction prediction(double tState) const {
        return {tState, rows(controller_.lastX()), rows(controller_.lastU())};
    }

private:
    static std::vector<std::vector<double>> rows(const casadi::DM& m) {
        std::vector<std::vector<double>> out(static_cast<size_t>(m.size1()));
        for (casadi_int r = 0; r < m.size1(); ++r) {
            for (casadi_int c = 0; c < m.size2(); ++c) {
                out[static_cast<size_t>(r)].push_back(static_cast<double>(m(r, c)));
            }
        }
        return out;
    }

    MPCController controller_;
    casadi::DM x0_;
};

} // namespace npmpc::mpc
