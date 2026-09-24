// Validates the laopt OCP (FurutaNPOcpEigen.hpp) against the CasADi/Opti path:
//  1. Cost check: the OCP's objective terms on an NP rollout must equal the
//     CasADi furutaCost.
//  2. Solve: MultipleShooting + IPOPT or SQP/PIQP (switch: kSolver) from the MPCController cold start must
//     reproduce the CasADi MPCController solution for the x0 in mpc_config.yaml.
//  3. Re-solve from a different x0 without re-taping (the x0 bounds are
//     changed on the OCP between solves).
// Both sides use the full cost from mpc_config.yaml, including cost.x_diff
// (in the laopt OCP via the state change d = x_k - x_{k-1} as extra states).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <casadi/casadi.hpp>
#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/multiple_shooting.hpp"

#include "npmpc/mpc/controller.hpp"
#include "npmpc/mpc/laopt/FurutaNPOcpEigen.hpp"
#include "npmpc/mpc/laopt/laopt_solver.hpp"
#include "npmpc/mpc/mpc_config_io.hpp"
#include "npmpc/mpc/problem/FurutaNPMPC.hpp"
#include "npmpc/mpc/problem/furuta_cost.hpp"
#include "npmpc/nps/weights_io.hpp"

#ifndef NPMPC_PROJECT_ROOT
#define NPMPC_PROJECT_ROOT "."
#endif

namespace {

constexpr int N = 12; // must match horizon_steps in mpc_config.yaml

/* Solver switch: IPOPT or SQP with PIQP as QP solver (settings in laopt_solver.hpp). */
using npmpc::mpc::furuta_laopt::SolverType;
constexpr SolverType kSolver = SolverType::SQP_PIQP;

using Ocp = npmpc::mpc::furuta_laopt::FurutaNPOCP<N>;
using Transcription = laopt_tools::MultipleShooting<Ocp, N, laopt::ERK4>; // integrator unused (DiscreteDynamics)
using OptProblem = laopt::Problem<Transcription>;
using Solver = npmpc::mpc::furuta_laopt::SolverFor<kSolver, OptProblem>;
constexpr const char* kSolverName = npmpc::mpc::furuta_laopt::solverName<Solver>();

constexpr int NXP = npmpc::mpc::furuta_laopt::kNX;      // physical state size (Ocp::NX = 2 * NXP: [x; d])
using StateTrajectory = Ocp::PhysStateTrajectory;       // physical states (NXP, N+1)
using InputTrajectory = Transcription::InputTrajectory; // (NU, N)

// laopt objective as MultipleShooting assembles it: sum_k h * lagrange + mayer, h = 1/N,
// on the OCP states [x; d] of the physical trajectory x.
double ocpObjective(Ocp& ocp, const StateTrajectory& x, const InputTrajectory& u, const Ocp::Param& p)
{
    const Eigen::Vector<double, 1> t0 = Eigen::Vector<double, 1>::Constant(ocp.t0);
    const Eigen::Vector<double, 1> tf = Eigen::Vector<double, 1>::Constant(ocp.tf_lb);
    const Ocp::StateTrajectory xa = Ocp::augment(x);
    const double h = 1.0 / N;
    double obj = 0.0;
    for (int k = 0; k < N; ++k) {
        obj += h * ocp.lagrange_term_impl(Ocp::State(xa.col(k)), Ocp::Input(u.col(k)), p, t0, tf, double(k) / N);
    }
    return obj + ocp.mayer_term_impl(Ocp::State(xa.col(N)), p, t0, tf);
}

// CasADi DM (rows = time steps) -> Eigen (columns = time steps).
template<int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> fromCasadiTransposed(const casadi::DM& m)
{
    Eigen::Matrix<double, Rows, Cols> out;
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) { out(r, c) = static_cast<double>(m(c, r)); }
    }
    return out;
}

// Cold start as in MPCController::warmStart: x linear from x0 to [2pi,0,0,0], u = 0.
StateTrajectory coldStartX(const Ocp::PhysState& x0)
{
    const Ocp::PhysState target(2.0 * M_PI, 0.0, 0.0, 0.0);
    StateTrajectory x;
    for (int i = 0; i <= N; ++i) {
        const double t = static_cast<double>(i) / N;
        x.col(i) = x0 + t * (target - x0);
    }
    return x;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace std::chrono;

    const std::string weightsPath =
        argc > 1 ? argv[1] : std::string(NPMPC_PROJECT_ROOT) + "/model/np_weights.yaml";
    const std::string mpcConfigPath =
        argc > 2 ? argv[2] : std::string(NPMPC_PROJECT_ROOT) + "/model/mpc_config.yaml";

    bool ok = true;

    /* laopt OCP */
    std::shared_ptr<Ocp> ocp = std::make_shared<Ocp>(weightsPath, mpcConfigPath);
    const Ocp::PhysState x0 = ocp->settings.x0;

    /* CasADi reference, same config (incl. x_diff) */
    npmpc::nps::NeuralProcess npCasadi = npmpc::nps::loadNeuralProcessFromYaml(weightsPath);
    npmpc::mpc::problem::MPCParams params = npmpc::mpc::loadMPCParamsFromYaml(mpcConfigPath);

    // -- 1. Cost check on an NP rollout --
    {
        InputTrajectory u;
        for (int k = 0; k < N; ++k) { u(0, k) = 0.04 * std::sin(0.7 * k); }
        const StateTrajectory x = ocp->model.rollout<N>(x0, u, ocp->settings.z, ocp->settings.dt);

        const double objLaopt = ocpObjective(*ocp, x, u, Ocp::Param::Zero());

        casadi::MX xSym = casadi::MX::sym("x", N + 1, NXP);
        casadi::MX uSym = casadi::MX::sym("u", N, Ocp::NU);
        casadi::Function costFn("furuta_cost", {xSym, uSym},
                                {npmpc::mpc::problem::furutaCost(xSym, uSym, params.cost)});
        casadi::DM xVal = casadi::DM::zeros(N + 1, NXP);
        casadi::DM uVal = casadi::DM::zeros(N, Ocp::NU);
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NXP; ++j) { xVal(k, j) = x(j, k); }
        }
        for (int k = 0; k < N; ++k) { uVal(k, 0) = u(0, k); }
        const double objCasadi = static_cast<double>(costFn(std::vector<casadi::DM>{xVal, uVal}).at(0));

        const double relDiff = std::abs(objLaopt - objCasadi) / std::max(1.0, std::abs(objCasadi));
        std::cout.precision(17);
        std::cout << "[cost] CasADi: " << objCasadi << "  laopt OCP: " << objLaopt << "  rel diff: " << relDiff << "\n";
        if (relDiff > 1e-12) {
            std::cerr << "[cost] MISMATCH\n";
            ok = false;
        }
        std::cout.precision(6);
    }

    // -- 2. Solve from the cold start and compare with CasADi MPCController --
    std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);

    std::shared_ptr<OptProblem> optProblem = std::make_shared<OptProblem>(transcription); // generates the tape

    Solver solver(optProblem);
    npmpc::mpc::furuta_laopt::configureSolver(solver);

    // Guesses must be set after the solver is constructed (see MultipleShooting::set_X_guess).
    transcription->set_X_guess(Ocp::augment(coldStartX(x0)));
    transcription->set_U_guess(InputTrajectory(InputTrajectory::Zero())); // typed: Input/InputTrajectory overloads are ambiguous for NU = 1
    transcription->set_p_guess(Ocp::Param::Zero());

    const steady_clock::time_point tSolve0 = steady_clock::now();
    const auto result1 = npmpc::mpc::furuta_laopt::solveOnce(solver);
    const double solve1Ms = duration<double, std::milli>(steady_clock::now() - tSolve0).count();

    const StateTrajectory xLaopt = transcription->get_X_opt().topRows<NXP>(); // physical part of [x; d]
    const InputTrajectory uLaopt = transcription->get_U_opt();

    npmpc::mpc::problem::FurutaNPMPC mpc(&npCasadi, params.z, params.cost);
    npmpc::mpc::MPCController controller(mpc, params);
    controller.warmStart(params.x0.T());
    const StateTrajectory xCasadi = fromCasadiTransposed<NXP, N + 1>(controller.lastX());
    const InputTrajectory uCasadi = fromCasadiTransposed<Ocp::NU, N>(controller.lastU());

    const double xDiff = (xLaopt - xCasadi).cwiseAbs().maxCoeff();
    const double uDiff = (uLaopt - uCasadi).cwiseAbs().maxCoeff();

    std::cout << "\n[solve 1] " << kSolverName << " status: " << result1.status
              << "  solve: " << solve1Ms << " ms\n";
    std::cout << "X laopt (rows = time steps):\n" << xLaopt.transpose() << "\n";
    std::cout << "U laopt: " << uLaopt << "\n";
    std::cout << "U CasADi: " << uCasadi << "\n";
    std::cout << "slack laopt: " << transcription->get_p_opt().transpose() << "\n";
    std::cout << "max |X laopt - X CasADi| = " << xDiff << ",  max |U laopt - U CasADi| = " << uDiff << "\n";
    if (!result1.converged) {
        std::cerr << "[solve 1] " << kSolverName << " did not converge\n";
        ok = false;
    }
    if (xDiff > 1e-4 || uDiff > 1e-4) {
        std::cerr << "[solve 1] MISMATCH with CasADi solution\n";
        ok = false;
    }

    // -- 3. Re-solve from a new x0, reusing tape/problem/solver (warm start from solve 1) --
    const Ocp::PhysState x0b(M_PI - 0.2, 0.1, 0.5, -0.3);
    ocp->set_initial_state(x0b);

    const steady_clock::time_point tSolve1 = steady_clock::now();
    const auto result2 = npmpc::mpc::furuta_laopt::solveOnce(solver);
    const double solve2Ms = duration<double, std::milli>(steady_clock::now() - tSolve1).count();

    const Ocp::PhysState x0Sol = transcription->get_X_opt().col(0).head<NXP>();
    const double x0Err = (x0Sol - x0b).cwiseAbs().maxCoeff();
    std::cout << "\n[solve 2] " << kSolverName << " status: " << result2.status << "  solve: " << solve2Ms << " ms\n";
    std::cout << "x0 requested: " << x0b.transpose() << "\n";
    std::cout << "x0 solution:  " << x0Sol.transpose() << "  (max err " << x0Err << ")\n";
    if (!result2.converged) {
        std::cerr << "[solve 2] " << kSolverName << " did not converge\n";
        ok = false;
    }
    if (x0Err > 1e-3 + 1e-6) {
        std::cerr << "[solve 2] x0 bounds changed on the OCP were NOT picked up by the solver\n";
        ok = false;
    }

    std::cout << "\n" << (ok ? "OK: laopt OCP matches CasADi reference." : "FAILED") << std::endl;
    return ok ? 0 : 1;
}
