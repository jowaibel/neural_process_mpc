#pragma once

#include <casadi/casadi.hpp>
#include <string>

namespace npmpc::mpc {

// Port of solver.py: configures the CasADi Opti solver.
void configureSolver(casadi::Opti& problem, const std::string& solverName);

} // namespace npmpc::mpc
