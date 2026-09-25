#pragma once

// laopt OCP for the Furuta pendulum with Neural Process dynamics: the laopt
// counterpart of the CasADi/Opti path (MPCBase + FurutaNPMPC + furuta_cost +
// MPCController::buildOptimization). All OCP modules (cost, constraints,
// dynamics call) live in this one file, following
// laopt/examples/fixed_wing/LonOcpEigen.hpp; only the config loader is shared
// with the equation-based OCP (furuta_ocp_settings.hpp).
//
// Transcribe with MultipleShootingXDiff<FurutaNPOCP<N>, N, ...>
// (multiple_shooting_xdiff.hpp: its Lagrange term also gets x_{k+1}, for the
// x_diff cost). The OCP uses DiscreteDynamics, so the integrator template
// argument is unused, the transcription constrains
// x_{k+1} == discrete_dynamics_impl(x_k, u_k), and the Lagrange terms are
// summed unweighted.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/control_problem_base.hpp"
#include "npmpc/nps/laopt_ad_erf.hpp" // before FurutaNPEigen: erf for laopt's AD scalar
#include "npmpc/nps/FurutaNPEigen.hpp"
#include "npmpc/mpc/laopt/furuta_ocp_settings.hpp"

namespace npmpc::mpc::furuta_laopt {

using Model = npmpc::nps::FurutaNPEigen;

static_assert(Model::kStateDim == kNX && Model::kInputDim == kNU && Model::kLatentDim == kNZ,
              "FurutaOcpSettings dimensions must match the NP model");

/* FurutaNPOCP: decision variables per MultipleShooting node are x_k (NX) and
 * u_k (NU); the optimized parameters p (NP = NX) are the horizon-wide state
 * slacks of the CasADi version (MPCController's `slack_`, 1 x x_size).
 *
 * CasADi -> laopt mapping:
 *   integrationConstraints    -> discrete_dynamics_impl (FurutaNPEigen::step)
 *   furutaCost, stage terms   -> lagrange_term_impl(x_k, x_{k+1}, u_k): stage cost + x_diff on x_{k+1} - x_k
 *   furutaCost, terminal term -> mayer_term_impl
 *   slackCost                 -> mayer_term_impl (p is horizon-wide)
 *   slackConstraints          -> p_lb / p_ub
 *   stateConstraints          -> inequality_constraints{,0,f}_impl: g = [x - s; x + s],
 *                                x - s <= x_ub, x + s >= x_lb, at all N+1 nodes
 *   inputConstraints          -> u_lb / u_ub
 *   x0 +- 1e-3                -> x0_lb / x0_ub (set_initial_state)
 */
template<int N_>
class FurutaNPOCP :
        public laopt_tools::ControlProblemBase</*Scalar*/ double, /*NX*/ kNX, /*NU*/ kNU, /*NP*/ kNP,
                                               /*NG*/ kNG, /*NG0*/ kNG, /*NGF*/ kNG,
                                               laopt_tools::DiscreteDynamics>
{
public:
    static constexpr int N = N_; // horizon steps (MultipleShooting segments)

    Model model;
    FurutaOcpSettings settings;

    // Loads the NP weights (np_weights.yaml) and the MPC config (mpc_config.yaml).
    FurutaNPOCP(const std::string& weightsPath, const std::string& mpcConfigPath) :
            model(Model::fromYaml(weightsPath)),
            settings(FurutaOcpSettings::fromYaml(mpcConfigPath, N))
    {
        if (!settings.hasZ) {
            throw std::runtime_error("FurutaNPOCP: no latent `z` in " + mpcConfigPath +
                                     " (export it with scripts/export_mpc_config.py --method neural)");
        }
        apply_settings();
        set_initial_state(settings.x0);
    }

    // Transfers `settings` into the laopt bounds. Call again after changing
    // bounds in `settings`; cost weights, z and dt are read at every evaluation.
    void apply_settings()
    {
        constexpr double inf = std::numeric_limits<double>::infinity();

        u_lb = settings.uLb;
        u_ub = settings.uUb;

        // States are bounded through g (so they can be softened); no box bounds.
        x_lb = State::Constant(-inf);
        x_ub = State::Constant(inf);

        // Slacks: [0, slack_bound], fixed to 0 where slack_bound is inf (hard bound).
        for (int i = 0; i < kNX; ++i) {
            p_lb(i) = 0.0;
            p_ub(i) = std::isfinite(settings.slackBound(i)) ? settings.slackBound(i) : 0.0;
        }

        // g = [x - s; x + s]:  x - s <= x_ub,  x + s >= x_lb  (rows with inf bounds are free).
        g_ub << settings.xUb, State::Constant(inf);
        g_lb << State::Constant(-inf), settings.xLb;
        g0_ub = g_ub;
        g0_lb = g_lb;
        gf_ub = g_ub;
        gf_lb = g_lb;

        set_tf(N * settings.dt); // only defines the time grid; the dynamics use dt
    }

    // Initial state constraint, as in MPCController: x_0 within x0 +- 1e-3.
    void set_initial_state(const State& x0)
    {
        x0_lb = x0 - State::Constant(1e-3);
        x0_ub = x0 + State::Constant(1e-3);
    }

    /*
     * Cost terms (port of furuta_cost.cpp / MPCBase::slackCost)
     */
    // Stage cost with the 2*pi-periodic half-angle lift on theta:
    // 2 * (1 - cos(theta)) == (2 * sin(theta / 2))^2.
    template<typename T>
    T stage_cost(const state_t<T>& x, const input_t<T>& u) const
    {
        using std::cos;
        state_t<T> lifted;
        lifted << T(2.0) * (T(1.0) - cos(x(0))), x(1) * x(1), x(2) * x(2), x(3) * x(3);
        return settings.costX.template cast<T>().dot(lifted)
               + settings.costU.template cast<T>().dot(u.cwiseProduct(u));
    }

    // x_diff cost costXDiff . dx^2 on the state change dx = x_{k+1} - x_k.
    template<typename T>
    T diff_cost(const state_t<T>& dx) const
    {
        return settings.costXDiff.template cast<T>().dot(dx.cwiseProduct(dx));
    }

    // LQR terminal cost eN^T P eN on the half-angle-lifted terminal state.
    template<typename T>
    T terminal_cost(const state_t<T>& xf) const
    {
        using std::sin;
        state_t<T> eN;
        eN << T(2.0) * sin(xf(0) / T(2.0)), xf(1), xf(2), xf(3);
        return eN.dot(settings.terminalP.template cast<T>() * eN);
    }

    // slackWeight * 0.5 * (s^2 + s), only for states with a finite slack bound.
    template<typename T>
    T slack_cost(const param_t<T>& s) const
    {
        T cost(0.0);
        for (int i = 0; i < kNX; ++i) {
            if (std::isfinite(settings.slackBound(i))) {
                cost += settings.slackWeight * 0.5 * (s(i) * s(i) + s(i));
            }
        }
        return cost;
    }

    // Softened state bounds g = [x - s; x + s], see apply_settings().
    template<typename T>
    ineq_constr_t<T> state_bound_constraints(const state_t<T>& x, const param_t<T>& s) const
    {
        ineq_constr_t<T> g;
        g << x - s, x + s;
        return g;
    }

    /*
     * laopt interface (MultipleShootingXDiff: the Lagrange term gets x_{k+1})
     */
    template<typename x_t, typename xn_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    T lagrange_term_impl(const Eigen::MatrixBase<x_t>& x,
                         const Eigen::MatrixBase<xn_t>& x_next,
                         const Eigen::MatrixBase<u_t>& u,
                         const Eigen::MatrixBase<p_t>& p,
                         const Eigen::MatrixBase<t0_t>& t0,
                         const Eigen::MatrixBase<tf_t>& tf,
                         const tau_t& tau)
    {
        unused(p, t0, tf, tau);
        // Discrete dynamics: the Lagrange terms are summed unweighted, as in furutaCost.
        const state_t<T> xk(x);
        return stage_cost<T>(xk, input_t<T>(u)) + diff_cost<T>(state_t<T>(x_next) - xk);
    }

    template<typename xf_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    T mayer_term_impl(const Eigen::MatrixBase<xf_t>& xf,
                      const Eigen::MatrixBase<p_t>& p,
                      const Eigen::MatrixBase<t0_t>& t0,
                      const Eigen::MatrixBase<tf_t>& tf)
    {
        unused(t0, tf);
        return terminal_cost<T>(state_t<T>(xf)) + slack_cost<T>(param_t<T>(p));
    }

    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    state_t<T> discrete_dynamics_impl(const Eigen::MatrixBase<x_t>& x,
                                      const Eigen::MatrixBase<u_t>& u,
                                      const Eigen::MatrixBase<p_t>& p,
                                      const Eigen::MatrixBase<t0_t>& t0,
                                      const Eigen::MatrixBase<tf_t>& tf,
                                      const tau_t& tau)
    {
        unused(p, t0, tf, tau);
        return model.step<T>(state_t<T>(x), input_t<T>(u), settings.z.template cast<T>(), settings.dt);
    }

    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    ineq_constr_t<T> inequality_constraints_impl(const Eigen::MatrixBase<x_t>& x,
                                                 const Eigen::MatrixBase<u_t>& u,
                                                 const Eigen::MatrixBase<p_t>& p,
                                                 const Eigen::MatrixBase<t0_t>& t0,
                                                 const Eigen::MatrixBase<tf_t>& tf,
                                                 const tau_t& tau)
    {
        unused(u, t0, tf, tau);
        return state_bound_constraints<T>(state_t<T>(x), param_t<T>(p));
    }

    template<typename x_t, typename u_t, typename p_t, typename t0_t,
            typename T = typename x_t::Scalar> // T is scalar type
    ineq_constr0_t<T> inequality_constraints0_impl(const Eigen::MatrixBase<x_t>& x0,
                                                   const Eigen::MatrixBase<u_t>& u0,
                                                   const Eigen::MatrixBase<p_t>& p,
                                                   const Eigen::MatrixBase<t0_t>& t0)
    {
        unused(u0, t0);
        return state_bound_constraints<T>(state_t<T>(x0), param_t<T>(p));
    }

    template<typename xf_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    ineq_constrf_t<T> inequality_constraintsf_impl(const Eigen::MatrixBase<xf_t>& xf,
                                                   const Eigen::MatrixBase<p_t>& p,
                                                   const Eigen::MatrixBase<t0_t>& t0,
                                                   const Eigen::MatrixBase<tf_t>& tf)
    {
        unused(t0, tf);
        return state_bound_constraints<T>(state_t<T>(xf), param_t<T>(p));
    }
};

} // namespace npmpc::mpc::furuta_laopt
