#pragma once

#include <casadi/casadi.hpp>

namespace npmpc::dynamics {

// Port of FurutaDynamics (furuta.py): only the CasADi dynamics
// (casadi_dynamics) are ported; the torch dynamics() is left out.
//
// x: [theta, phi, theta_dot, phi_dot], u: [torque], p: [lp, mp, lr, mr]
// (pendulum length/mass, arm length/mass).
class FurutaDynamics {
public:
    static constexpr double g = 9.81;
    static constexpr int xSize = 4;
    static constexpr int uSize = 1;

    // Port of FurutaDynamics.casadi_dynamics: x (1, 4), u (1, 1), p (4 entries)
    // -> x_dot (1, 4) = [theta_dot, phi_dot, theta_ddot, phi_ddot].
    static casadi::MX casadiDynamics(const casadi::MX& x, const casadi::MX& u, const casadi::DM& p);
};

} // namespace npmpc::dynamics
