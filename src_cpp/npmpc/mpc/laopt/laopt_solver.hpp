#pragma once

// Solver selection for the laopt Furuta NP MPC: IPOPT or SQP with PIQP as QP
// solver, behind one interface (configureSolver / solveOnce), so the test and
// the closed-loop client treat both solvers the same way.

#include <string>
#include <type_traits>
#include <utility>

#include "laopt/laopt.hpp"
#include "laopt/solvers/ipopt_interface.hpp"
#include "laopt/solvers/piqp_interface.hpp"
#include "laopt/solvers/sqp_solver.hpp"

namespace npmpc::mpc::furuta_laopt {

enum class SolverType { IPOPT, SQP_PIQP };

template<SolverType S, typename OptProblem>
using SolverFor = std::conditional_t<S == SolverType::IPOPT,
                                     laopt::IpoptSolver<OptProblem>,
                                     laopt::SQPSolver<OptProblem, laopt::PIQPSolver<>>>;

template<typename SolverT>
inline constexpr bool kIsIpopt = std::is_same_v<decltype(std::declval<SolverT&>().solve()), Ipopt::ApplicationReturnStatus>;

template<typename SolverT>
constexpr const char* solverName()
{
    return kIsIpopt<SolverT> ? "IPOPT" : "SQP (PIQP)";
}

// Solver settings used by all laopt MPC executables.
template<typename SolverT>
void configureSolver(SolverT& solver)
{
    if constexpr (kIsIpopt<SolverT>) {
        solver.set_banner_message(false);
        solver.set_print_level(0);
        solver.set_tol(1e-6);    // as configureSolver("ipopt") in the CasADi path
        solver.set_max_iter(50); // as configureSolver("ipopt") in the CasADi path
    }
    else
    {
        solver.settings().max_iter = 1; // laopt defaults otherwise (eps_prim 1e-6, eps_dual 1e-4, Gauss-Newton Hessian)
        solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;
        // solver.settings().globalization_strategy = laopt::globalization_t::LINE_SEARCH_L1;
    }
}

struct SolveResult {
    bool converged;
    std::string status; // human-readable solver status
};

// One solve() call. No retries: in the MPC loop the (possibly unconverged)
// iterate is used as is, for both solvers.
template<typename SolverT>
SolveResult solveOnce(SolverT& solver)
{
    const auto status = solver.solve();
    if constexpr (kIsIpopt<SolverT>) {
        return {status == Ipopt::Solve_Succeeded || status == Ipopt::Solved_To_Acceptable_Level,
                SolverT::ipopt_status_text(status)};
    } else {
        std::string text;
        switch (status.status) {
            case laopt::sqp_status_t::SOLVED: text = "SOLVED"; break;
            case laopt::sqp_status_t::MAX_ITER_REACHED: text = "MAX_ITER_REACHED"; break;
            case laopt::sqp_status_t::INFEASIBLE: text = "INFEASIBLE"; break;
            case laopt::sqp_status_t::NON_CONVEX_QP: text = "NON_CONVEX_QP"; break;
            case laopt::sqp_status_t::QP_SOLVER_ERROR: text = "QP_SOLVER_ERROR"; break;
            case laopt::sqp_status_t::UNSOLVED: text = "UNSOLVED"; break;
            case laopt::sqp_status_t::INVALID_SETTINGS: text = "INVALID_SETTINGS"; break;
        }
        text += " (" + std::to_string(status.iter) + " SQP iter, " + std::to_string(status.qp_iter) + " QP iter)";
        return {status.status == laopt::sqp_status_t::SOLVED, text};
    }
}

} // namespace npmpc::mpc::furuta_laopt
