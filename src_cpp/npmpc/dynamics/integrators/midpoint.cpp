#include "npmpc/dynamics/integrators/midpoint.hpp"

#include "npmpc/dynamics/furuta.hpp"

namespace npmpc::dynamics::integrators {

std::vector<casadi::MX> MidpointIntegration::casadiImplicit(
        const casadi::MX& x, const casadi::MX& u, double dt, const casadi::DM& p) const {
    std::vector<casadi::MX> constraints;
    constraints.reserve(static_cast<size_t>(u.size1()));

    for (casadi_int i = 0; i < u.size1(); ++i) {
        casadi::MX xi = x(i, casadi::Slice());
        casadi::MX xNext = x(i + 1, casadi::Slice());
        casadi::MX xMid = (xi + xNext) / 2;
        constraints.push_back(xNext == xi + dt * FurutaDynamics::casadiDynamics(xMid, u(i, casadi::Slice()), p));
    }
    return constraints;
}

} // namespace npmpc::dynamics::integrators
