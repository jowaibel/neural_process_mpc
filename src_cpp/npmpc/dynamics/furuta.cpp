#include "npmpc/dynamics/furuta.hpp"

namespace npmpc::dynamics {

casadi::MX FurutaDynamics::casadiDynamics(const casadi::MX& x, const casadi::MX& u, const casadi::DM& p) {
    casadi::MX theta = x(0, 0);
    casadi::MX thetaDot = x(0, 2);
    casadi::MX phiDot = x(0, 3);
    casadi::MX torque = u(0, 0);
    const double lp = static_cast<double>(p(0));
    const double mp = static_cast<double>(p(1));
    const double lr = static_cast<double>(p(2));
    const double mr = static_cast<double>(p(3));

    // Inertia calculation
    const double jR = mr * lr * lr / 3.0 + mp * lr * lr;
    const double jP = mp * lp * lp / 3.0;

    // B definition
    casadi::MX b00 = jP;
    casadi::MX b01 = -mp * lr * lp / 2.0 * casadi::MX::cos(theta);
    casadi::MX b10 = b01;
    casadi::MX b11 = jR + jP * casadi::MX::pow(casadi::MX::sin(theta), 2);

    // A definition
    casadi::MX a0 = jP * casadi::MX::sin(2.0 * theta) / 2.0 * casadi::MX::pow(phiDot, 2)
                    + mp * lp * g / 2.0 * casadi::MX::sin(theta);
    casadi::MX a1 = -jP * casadi::MX::sin(2.0 * theta) * phiDot * thetaDot
                    - mp * lr * lp / 2.0 * casadi::MX::sin(theta) * casadi::MX::pow(thetaDot, 2) + torque;

    // [theta_ddot, phi_ddot] = adj(B) / det(B) @ A
    casadi::MX bDet = b00 * b11 - b01 * b10;
    casadi::MX thetaDdot = (b11 * a0 - b01 * a1) / bDet;
    casadi::MX phiDdot = (-b10 * a0 + b00 * a1) / bDet;

    return casadi::MX::horzcat({thetaDot, phiDot, thetaDdot, phiDdot});
}

} // namespace npmpc::dynamics
