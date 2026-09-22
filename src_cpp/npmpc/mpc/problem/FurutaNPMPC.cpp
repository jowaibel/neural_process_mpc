#include "npmpc/mpc/problem/FurutaNPMPC.hpp"

#include "npmpc/mpc/problem/furuta_cost.hpp"

namespace npmpc::mpc::problem {

casadi::MX FurutaNPMPC::costFunction(const casadi::MX& x, const casadi::MX& u) const {
    return furutaCost(x, u, cost_);
}

std::vector<casadi::MX> FurutaNPMPC::integrationConstraints(const casadi::MX& x, const casadi::MX& u, double dt) const {
    return integrator_.casadiImplicit(x, u, dt, zCasadi_);
}

} // namespace npmpc::mpc::problem
