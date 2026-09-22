#pragma once

#include <casadi/casadi.hpp>

#include "npmpc/mpc/problem/params.hpp"

namespace npmpc::mpc::problem {

// Port of furuta_cost.py: stage cost + LQR terminal cost, using the
// 2*pi-periodic half-angle lift theta -> 2*sin(theta/2) on the pendulum
// angle so all upright configurations cost the same.
casadi::MX furutaCost(const casadi::MX& x, const casadi::MX& u, const CostParams& cost);

} // namespace npmpc::mpc::problem
