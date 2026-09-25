#pragma once

#include <casadi/casadi.hpp>
#include <vector>

namespace npmpc::dynamics::integrators {

// Port of MidpointIntegration (midpoint.py) for the Furuta dynamics
// (FurutaDynamics, furuta.hpp). Only the CasADi implicit integration used by
// the MPC (casadi_implicit) is ported; the torch step()/integrate() and
// casadi_explicit are left out.
class MidpointIntegration {
public:
    // Implicit midpoint constraints x[i+1] == x[i] + dt * f((x[i] + x[i+1]) / 2, u[i], p)
    // for x (N+1, 4), u (N, 1), p = [lp, mp, lr, mr].
    std::vector<casadi::MX> casadiImplicit(const casadi::MX& x, const casadi::MX& u,
                                            double dt, const casadi::DM& p) const;
};

} // namespace npmpc::dynamics::integrators
