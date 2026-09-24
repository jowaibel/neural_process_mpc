#pragma once

// laopt OCP for the Furuta pendulum with the analytical (equation-based)
// dynamics: the laopt counterpart of the Python FurutaMPC (method
// 'equation'). Cost, slacks and constraints are the same as in
// FurutaNPOcpEigen.hpp; only the dynamics differ. All OCP modules live in this
// one file (following laopt/examples/fixed_wing/LonOcpEigen.hpp); only the
// config loader is shared (furuta_ocp_settings.hpp).
//
// Transcribe with laopt_tools::MultipleShooting<FurutaEqOCP<N>, N, laopt::IRK2>:
// continuous dynamics (dynamics_impl = the Furuta ODE) discretized by laopt's
// implicit midpoint rule, x_{k+1} = x_k + dt * f((x_k + x_{k+1}) / 2, u_k), which
// is exactly MidpointIntegration.casadi_implicit in Python. This needs
// tf = N * dt (set in apply_settings), since MultipleShooting steps by
// (tf - t0) / N.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/control_problem_base.hpp"
#include "npmpc/mpc/laopt/furuta_ocp_settings.hpp"

namespace npmpc::mpc::furuta_laopt {

/* FurutaEqOCP: decision variables per MultipleShooting node are the OCP state
 * [x_k; d_k] (NX = 2 * kNX) and u_k (NU), where x is the physical state and
 * d_k = x_k - x_{k-1} its change over the previous step (d_0 fixed to 0), for
 * the x_diff cost as in FurutaNPOCP. The optimized parameters p (NP = kNX) are
 * the horizon-wide state slacks.
 *
 * With continuous dynamics, d gets the ODE  d_dot = 2 f(x) - 2 d / dt.  Under
 * the implicit midpoint rule (IRK2) with step dt this gives exactly
 * d_{k+1} = dt * f(x_mid) = x_{k+1} - x_k  (d_k cancels), i.e. the same d as
 * in the discrete NP OCP. This holds only for IRK2 with step dt.
 *
 * Python -> laopt mapping:
 *   integration_constraints (implicit midpoint) -> dynamics_impl + laopt::IRK2
 *   furuta_cost, stage + interstage terms       -> lagrange_term_impl (scaled by N)
 *   furuta_cost, terminal term (+ x_diff on d_N) -> mayer_term_impl
 *   slack_cost                                  -> mayer_term_impl (p is horizon-wide)
 *   slack/state/input constraints, x0 +- 1e-3   -> as in FurutaNPOCP
 */
template<int N_>
class FurutaEqOCP :
        public laopt_tools::ControlProblemBase</*Scalar*/ double, /*NX*/ kNXOcp, /*NU*/ kNU, /*NP*/ kNP,
                                               /*NG*/ kNG, /*NG0*/ kNG, /*NGF*/ kNG>
{
public:
    static constexpr int N = N_; // horizon steps (MultipleShooting segments)
    static constexpr double kGravity = 9.81; // FurutaDynamics.g

    /* Physical state (kNX) vs. OCP state (NX = [x; d]) */
    template<typename T> using phys_state_t = Eigen::Vector<T, kNX>;
    using PhysState = phys_state_t<double>;
    using PhysStateTrajectory = Eigen::Matrix<double, kNX, N + 1>; // column k = x_k
    using StateTrajectory = Eigen::Matrix<double, kNXOcp, N + 1>;   // column k = [x_k; d_k]

    FurutaOcpSettings settings;

    // Loads the MPC config (mpc_config_equation.yaml, with the plant parameters `p`).
    explicit FurutaEqOCP(const std::string& mpcConfigPath) :
            settings(FurutaOcpSettings::fromYaml(mpcConfigPath, N))
    {
        if (!settings.hasP) {
            throw std::runtime_error("FurutaEqOCP: no plant parameters `p` in " + mpcConfigPath +
                                     " (export it with scripts/export_mpc_config.py --method equation)");
        }
        apply_settings();
        set_initial_state(settings.x0);
    }

    // Transfers `settings` into the laopt bounds. Call again after changing
    // bounds in `settings`; cost weights, p and dt are read at every evaluation.
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
        g_ub << settings.xUb, PhysState::Constant(inf);
        g_lb << PhysState::Constant(-inf), settings.xLb;
        g0_ub = g_ub;
        g0_lb = g_lb;
        gf_ub = g_ub;
        gf_lb = g_lb;

        set_tf(N * settings.dt); // MultipleShooting steps by tf / N = dt
    }

    // Initial state constraint, as in MPCController: x_0 within x0 +- 1e-3,
    // for the physical state x0. d_0 is fixed to 0.
    void set_initial_state(const PhysState& x0)
    {
        x0_lb << x0 - PhysState::Constant(1e-3), PhysState::Zero();
        x0_ub << x0 + PhysState::Constant(1e-3), PhysState::Zero();
    }

    // OCP state trajectory [x; d] for a physical state trajectory x (for guesses).
    static StateTrajectory augment(const PhysStateTrajectory& x)
    {
        StateTrajectory xa;
        xa.template topRows<kNX>() = x;
        xa.template bottomRows<kNX>().col(0).setZero();
        for (int k = 1; k <= N; ++k) {
            xa.template bottomRows<kNX>().col(k) = x.col(k) - x.col(k - 1);
        }
        return xa;
    }

    /*
     * Dynamics (port of FurutaDynamics.casadi_dynamics)
     */
    // Furuta pendulum ODE x_dot = f(x, u; p) for x = [theta, phi, theta_dot, phi_dot],
    // u = [torque], p = [lp, mp, lr, mr]: B(theta) [theta_ddot; phi_ddot] = A(x, u).
    template<typename T>
    phys_state_t<T> furuta_ode(const phys_state_t<T>& x, const input_t<T>& u) const
    {
        using std::cos;
        using std::sin;
        const double lp = settings.p(0), mp = settings.p(1), lr = settings.p(2), mr = settings.p(3);
        const double jr = mr * lr * lr / 3.0 + mp * lr * lr;
        const double jp = mp * lp * lp / 3.0;

        const T& theta = x(0);
        const T& thetaDot = x(2);
        const T& phiDot = x(3);
        const T& torque = u(0);

        const T sinTheta = sin(theta);
        const T sin2Theta = sin(T(2.0) * theta);

        const T b00 = T(jp);
        const T b01 = -mp * lr * lp / 2.0 * cos(theta); // = b10
        const T b11 = jr + jp * sinTheta * sinTheta;

        const T a0 = jp * sin2Theta / 2.0 * phiDot * phiDot + mp * lp * kGravity / 2.0 * sinTheta;
        const T a1 = -jp * sin2Theta * phiDot * thetaDot - mp * lr * lp / 2.0 * sinTheta * thetaDot * thetaDot + torque;

        // [theta_ddot; phi_ddot] = adj(B) / det(B) * A
        const T det = b00 * b11 - b01 * b01;
        phys_state_t<T> xDot;
        xDot << thetaDot, phiDot, (b11 * a0 - b01 * a1) / det, (-b01 * a0 + b00 * a1) / det;
        return xDot;
    }

    /*
     * Cost terms (as in FurutaNPOCP / furuta_cost.py), on physical states
     */
    // Stage cost with the 2*pi-periodic half-angle lift on theta:
    // 2 * (1 - cos(theta)) == (2 * sin(theta / 2))^2.
    template<typename T>
    T stage_cost(const phys_state_t<T>& x, const input_t<T>& u) const
    {
        using std::cos;
        phys_state_t<T> lifted;
        lifted << T(2.0) * (T(1.0) - cos(x(0))), x(1) * x(1), x(2) * x(2), x(3) * x(3);
        return settings.costX.template cast<T>().dot(lifted)
               + settings.costU.template cast<T>().dot(u.cwiseProduct(u));
    }

    // x_diff cost costXDiff . d^2 on the state change d = x_k - x_{k-1}.
    template<typename T>
    T diff_cost(const phys_state_t<T>& d) const
    {
        return settings.costXDiff.template cast<T>().dot(d.cwiseProduct(d));
    }

    // LQR terminal cost eN^T P eN on the half-angle-lifted terminal state.
    template<typename T>
    T terminal_cost(const phys_state_t<T>& xf) const
    {
        using std::sin;
        phys_state_t<T> eN;
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

    // Softened state bounds g = [x - s; x + s] on the physical state, see apply_settings().
    template<typename T>
    ineq_constr_t<T> state_bound_constraints(const phys_state_t<T>& x, const param_t<T>& s) const
    {
        ineq_constr_t<T> g;
        g << x - s, x + s;
        return g;
    }

    /*
     * laopt interface
     */
    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    T lagrange_term_impl(const Eigen::MatrixBase<x_t>& x,
                         const Eigen::MatrixBase<u_t>& u,
                         const Eigen::MatrixBase<p_t>& p,
                         const Eigen::MatrixBase<t0_t>& t0,
                         const Eigen::MatrixBase<tf_t>& tf,
                         const tau_t& tau)
    {
        unused(p, t0, tf, tau);
        const state_t<T> xa(x);

        // MultipleShooting adds h * lagrange with h = 1/N; scale by N so the
        // objective is the plain sum over stages, as in furuta_cost.
        return static_cast<double>(N) * (stage_cost<T>(xa.template head<kNX>(), input_t<T>(u))
                                         + diff_cost<T>(xa.template tail<kNX>()));
    }

    template<typename xf_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    T mayer_term_impl(const Eigen::MatrixBase<xf_t>& xf,
                      const Eigen::MatrixBase<p_t>& p,
                      const Eigen::MatrixBase<t0_t>& t0,
                      const Eigen::MatrixBase<tf_t>& tf)
    {
        unused(t0, tf);
        const state_t<T> xa(xf);
        return terminal_cost<T>(xa.template head<kNX>()) + diff_cost<T>(xa.template tail<kNX>())
               + slack_cost<T>(param_t<T>(p));
    }

    // Continuous dynamics of the OCP state [x; d]: x_dot = f(x, u), d_dot = 2 f(x, u) - 2 d / dt
    // (see the class comment: exact d_{k+1} = x_{k+1} - x_k under IRK2 with step dt).
    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::MatrixBase<x_t>& x,
                             const Eigen::MatrixBase<u_t>& u,
                             const Eigen::MatrixBase<p_t>& p,
                             const Eigen::MatrixBase<t0_t>& t0,
                             const Eigen::MatrixBase<tf_t>& tf,
                             const tau_t& tau)
    {
        unused(p, t0, tf, tau);
        const state_t<T> xa(x);
        const phys_state_t<T> f = furuta_ode<T>(xa.template head<kNX>(), input_t<T>(u));
        state_t<T> xaDot;
        xaDot << f, T(2.0) * f - T(2.0 / settings.dt) * xa.template tail<kNX>();
        return xaDot;
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
        return state_bound_constraints<T>(state_t<T>(x).template head<kNX>(), param_t<T>(p));
    }

    template<typename x_t, typename u_t, typename p_t, typename t0_t,
            typename T = typename x_t::Scalar> // T is scalar type
    ineq_constr0_t<T> inequality_constraints0_impl(const Eigen::MatrixBase<x_t>& x0,
                                                   const Eigen::MatrixBase<u_t>& u0,
                                                   const Eigen::MatrixBase<p_t>& p,
                                                   const Eigen::MatrixBase<t0_t>& t0)
    {
        unused(u0, t0);
        return state_bound_constraints<T>(state_t<T>(x0).template head<kNX>(), param_t<T>(p));
    }

    template<typename xf_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    ineq_constrf_t<T> inequality_constraintsf_impl(const Eigen::MatrixBase<xf_t>& xf,
                                                   const Eigen::MatrixBase<p_t>& p,
                                                   const Eigen::MatrixBase<t0_t>& t0,
                                                   const Eigen::MatrixBase<tf_t>& tf)
    {
        unused(t0, tf);
        return state_bound_constraints<T>(state_t<T>(xf).template head<kNX>(), param_t<T>(p));
    }
};

} // namespace npmpc::mpc::furuta_laopt
