#pragma once

#include <casadi/casadi.hpp>
#include <vector>

#include "npmpc/mpc/problem/params.hpp"

namespace npmpc::mpc::problem {

// Port of MPCBase (MPCBase.py). Dynamics-specific pieces (cost_function,
// integration_constraints) are left pure virtual, as in Python subclasses
// like FurutaNPMPC.
class MPCBase {
public:
    virtual ~MPCBase() = default;

    virtual casadi::MX costFunction(const casadi::MX& x, const casadi::MX& u) const = 0;
    virtual std::vector<casadi::MX> integrationConstraints(const casadi::MX& x, const casadi::MX& u, double dt) const = 0;

    // Bounds on the slack variables.
    std::vector<casadi::MX> slackConstraints(const casadi::MX& slack, const std::vector<double>& slackBoundX) const;

    // Penalty cost on the slack variables.
    casadi::MX slackCost(const casadi::MX& slack, const std::vector<double>& slackBoundX) const;

    // State bound constraints, softened by slack where allowed.
    std::vector<casadi::MX> stateConstraints(const casadi::MX& x, const casadi::MX& slack,
                                              const HardBound& hardBound,
                                              const std::vector<double>& slackBoundX) const;

    // Input bound constraints.
    std::vector<casadi::MX> inputConstraints(const casadi::MX& u, const HardBound& hardBound) const;

protected:
    double slackWeight_ = 1000.0;
};

} // namespace npmpc::mpc::problem
