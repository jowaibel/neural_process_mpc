#include "npmpc/mpc/problem/MPCBase.hpp"

#include <limits>

namespace npmpc::mpc::problem {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

std::vector<casadi::MX> MPCBase::slackConstraints(const casadi::MX& slack,
                                                    const std::vector<double>& slackBoundX) const {
    std::vector<casadi::MX> constraints;
    for (size_t i = 0; i < slackBoundX.size(); ++i) {
        if (slackBoundX[i] != kInf) {
            casadi::MX s = slack(0, static_cast<casadi_int>(i));
            constraints.push_back(s >= 0);
            constraints.push_back(s <= slackBoundX[i]);
        }
    }
    return constraints;
}

casadi::MX MPCBase::slackCost(const casadi::MX& slack, const std::vector<double>& slackBoundX) const {
    casadi::MX cost = 0;
    for (size_t i = 0; i < slackBoundX.size(); ++i) {
        if (slackBoundX[i] != kInf) {
            casadi::MX s = slack(0, static_cast<casadi_int>(i));
            cost = cost + slackWeight_ * 0.5 * (casadi::MX::pow(s, 2) + s);
        }
    }
    return cost;
}

std::vector<casadi::MX> MPCBase::stateConstraints(const casadi::MX& x, const casadi::MX& slack,
                                                    const HardBound& hardBound,
                                                    const std::vector<double>& slackBoundX) const {
    std::vector<casadi::MX> constraints;
    for (size_t i = 0; i < hardBound.x.size(); ++i) {
        const Bound& bound = hardBound.x[i];
        casadi::MX xi = x(casadi::Slice(), static_cast<casadi_int>(i));
        if (bound.lower != -kInf) {
            if (slackBoundX[i] != kInf) {
                constraints.push_back(xi >= bound.lower - slack(0, static_cast<casadi_int>(i)));
            } else {
                constraints.push_back(xi >= bound.lower);
            }
        }
        if (bound.upper != kInf) {
            if (slackBoundX[i] != kInf) {
                constraints.push_back(xi <= bound.upper + slack(0, static_cast<casadi_int>(i)));
            } else {
                constraints.push_back(xi <= bound.upper);
            }
        }
    }
    return constraints;
}

std::vector<casadi::MX> MPCBase::inputConstraints(const casadi::MX& u, const HardBound& hardBound) const {
    std::vector<casadi::MX> constraints;
    for (size_t i = 0; i < hardBound.u.size(); ++i) {
        const Bound& bound = hardBound.u[i];
        casadi::MX ui = u(casadi::Slice(), static_cast<casadi_int>(i));
        if (bound.lower != -kInf) {
            constraints.push_back(ui >= bound.lower);
        }
        if (bound.upper != kInf) {
            constraints.push_back(ui <= bound.upper);
        }
    }
    return constraints;
}

} // namespace npmpc::mpc::problem
