#include "npmpc/mpc/solver.hpp"

#include <stdexcept>

namespace npmpc::mpc {

void configureSolver(casadi::Opti& problem, const std::string& solverName) {
    if (solverName == "ipopt") {
        problem.solver("ipopt", {
            {"ipopt.max_iter", 50},
            {"ipopt.tol", 1e-6},
            {"ipopt.print_level", 0},
            {"ipopt.sb", "yes"},
            {"print_time", 0},
        });
    } else if (solverName == "sqp") {
        problem.solver("sqpmethod", {
            {"qpsol", "qrqp"},
            {"jit", true},
            {"hessian_approximation", "limited-memory"},
            {"print_header", false},
            {"print_iteration", false},
            {"print_status", false},
            {"print_time", false},
            {"max_iter", 8},
            {"qpsol_options.max_iter", 8},
            {"qpsol_options.dual_inf_tol", 1e-2},
            {"qpsol_options.print_iter", false},
            {"qpsol_options.print_header", false},
            {"qpsol_options.print_info", false},
            {"qpsol_options.print_lincomb", false},
            {"qpsol_options.error_on_fail", false},
            {"expand", false},
        });
    } else {
        throw std::invalid_argument("Solver '" + solverName + "' not implemented");
    }
}

} // namespace npmpc::mpc
