#include "npmpc/mpc/problem/FurutaMPC.hpp"

#include "npmpc/mpc/problem/furuta_cost.hpp"

namespace npmpc::mpc::problem {

casadi::MX FurutaMPC::costFunction(const casadi::MX& x, const casadi::MX& u) const {
    return furutaCost(x, u, cost_);
}

std::vector<casadi::MX> FurutaMPC::integrationConstraints(const casadi::MX& x, const casadi::MX& u, double dt) const {
    return integrator_.casadiImplicit(x, u, dt, p_);
}

} // namespace npmpc::mpc::problem
