#pragma once

// laopt OCP for the Furuta pendulum with Neural Process dynamics: the laopt
// counterpart of the CasADi/Opti path (MPCBase + FurutaNPMPC + furuta_cost +
// MPCController::buildOptimization). All OCP modules (settings, cost,
// constraints, dynamics call) live in this one file, following
// laopt/examples/fixed_wing/LonOcpEigen.hpp.
//
// Transcribe with laopt_tools::MultipleShooting<FurutaNPOCP<N>, N, ...>: the
// OCP uses DiscreteDynamics, so the integrator template argument is unused and
// MultipleShooting constrains x_{k+1} == discrete_dynamics_impl(x_k, u_k).

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "laopt/laopt.hpp"
#include "laopt/tools/control_problem_base.hpp"
#include "npmpc/nps/laopt_ad_erf.hpp" // before FurutaNPEigen: erf for laopt's AD scalar
#include "npmpc/nps/FurutaNPEigen.hpp"

namespace npmpc::mpc::furuta_laopt {

using Model = npmpc::nps::FurutaNPEigen;

inline constexpr int kNX = Model::kStateDim;   // [theta, phi, theta_dot, phi_dot]
inline constexpr int kNU = Model::kInputDim;   // motor torque
inline constexpr int kNP = kNX;                // one slack per state, shared by the whole horizon
inline constexpr int kNG = 2 * kNX;            // softened state bounds: [x - s; x + s]

// Fixed-size counterpart of MPCParams/CostParams (params.hpp), read from
// model/mpc_config.yaml without going through CasADi types.
struct FurutaNPOcpSettings {
    using StateVec = Eigen::Vector<double, kNX>;
    using InputVec = Eigen::Vector<double, kNU>;

    double dt{};
    Model::Latent<double> z;             // NP latent code (fixed during a solve)

    StateVec xLb, xUb;                   // hard_bound.x (+-inf = unbounded)
    InputVec uLb, uUb;                   // hard_bound.u
    StateVec slackBound;                 // slack_bound.x (inf = hard constraint)
    double slackWeight{1000.0};          // MPCBase::slackWeight_

    StateVec costX;                      // cost.x (on [2(1-cos theta), phi^2, theta_dot^2, phi_dot^2])
    StateVec costXDiff;                  // cost.x_diff -- loaded but NOT used yet, see TODO(x_diff)
    InputVec costU;                      // cost.u
    Eigen::Matrix<double, kNX, kNX> terminalP; // cost.terminal_p (LQR terminal cost)

    StateVec x0;                         // experiment_options.x0

    // Loads model/mpc_config.yaml (as written by scripts/export_mpc_config.py).
    // Throws if horizon_steps != horizonSteps or any size differs from the
    // compile-time dimensions.
    static FurutaNPOcpSettings fromYaml(const std::string& path, int horizonSteps)
    {
        YAML::Node root = YAML::LoadFile(path);

        auto fail = [&](const std::string& msg) {
            throw std::runtime_error("FurutaNPOcpSettings::fromYaml(" + path + "): " + msg);
        };
        auto readVector = [&](const YAML::Node& node, auto& out, const std::string& what) {
            auto vals = node.as<std::vector<double>>();
            if (vals.size() != static_cast<size_t>(out.size())) {
                fail(what + " has size " + std::to_string(vals.size()) + ", expected " + std::to_string(out.size()));
            }
            for (Eigen::Index i = 0; i < out.size(); ++i) { out(i) = vals[i]; }
        };
        auto readBounds = [&](const YAML::Node& node, auto& lb, auto& ub, const std::string& what) {
            if (node.size() != static_cast<size_t>(lb.size())) {
                fail(what + " has " + std::to_string(node.size()) + " entries, expected " + std::to_string(lb.size()));
            }
            for (Eigen::Index i = 0; i < lb.size(); ++i) {
                auto pair = node[i].as<std::vector<double>>();
                if (pair.size() != 2) { fail(what + " entries must be [lower, upper]"); }
                lb(i) = pair[0];
                ub(i) = pair[1];
            }
        };

        if (root["horizon_steps"].as<int>() != horizonSteps) {
            fail("horizon_steps = " + std::to_string(root["horizon_steps"].as<int>()) +
                 ", but the OCP is compiled for N = " + std::to_string(horizonSteps));
        }
        if (root["x_size"].as<int>() != kNX || root["u_size"].as<int>() != kNU) {
            fail("x_size/u_size differ from the compiled dimensions");
        }

        FurutaNPOcpSettings s;
        s.dt = root["dt"].as<double>();
        readVector(root["z"], s.z, "z");

        readBounds(root["hard_bound"]["x"], s.xLb, s.xUb, "hard_bound.x");
        readBounds(root["hard_bound"]["u"], s.uLb, s.uUb, "hard_bound.u");
        readVector(root["slack_bound"]["x"], s.slackBound, "slack_bound.x");

        readVector(root["cost"]["x"], s.costX, "cost.x");
        readVector(root["cost"]["x_diff"], s.costXDiff, "cost.x_diff");
        readVector(root["cost"]["u"], s.costU, "cost.u");

        const YAML::Node pNode = root["cost"]["terminal_p"];
        if (pNode.size() != static_cast<size_t>(kNX)) { fail("cost.terminal_p must be " + std::to_string(kNX) + "x" + std::to_string(kNX)); }
        for (int r = 0; r < kNX; ++r) {
            Eigen::Vector<double, kNX> row;
            readVector(pNode[r], row, "cost.terminal_p row");
            s.terminalP.row(r) = row.transpose();
        }

        readVector(root["experiment_options"]["x0"], s.x0, "experiment_options.x0");

        // TODO(x_diff): see FurutaNPOCP::lagrange_term_impl.
        if (!s.costXDiff.isZero()) {
            std::cerr << "FurutaNPOcpSettings: WARNING cost.x_diff = [" << s.costXDiff.transpose()
                      << "] is ignored -- the x_diff cost is not implemented in the laopt OCP yet.\n";
        }
        return s;
    }
};

/* FurutaNPOCP: decision variables per MultipleShooting node are x_k (NX) and
 * u_k (NU); the optimized parameters p (NP = NX) are the horizon-wide state
 * slacks of the CasADi version (MPCController's `slack_`, 1 x x_size).
 *
 * CasADi -> laopt mapping:
 *   integrationConstraints    -> discrete_dynamics_impl (FurutaNPEigen::step)
 *   furutaCost, stage terms   -> lagrange_term_impl (scaled by N, see there)
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
    FurutaNPOcpSettings settings;

    // Loads the NP weights (np_weights.yaml) and the MPC config (mpc_config.yaml).
    FurutaNPOCP(const std::string& weightsPath, const std::string& mpcConfigPath) :
            model(Model::fromYaml(weightsPath)),
            settings(FurutaNPOcpSettings::fromYaml(mpcConfigPath, N))
    {
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

        // TODO(x_diff): the CasADi cost also has sum_k costXDiff . (x_{k+1} - x_k)^2
        // (furuta_cost.cpp, cost.x_diff in mpc_config.yaml). It is NOT included
        // here, because a laopt stage term only sees (x_k, u_k). To add it, either
        // augment the state with dx_k = x_k - x_{k-1} (NX 4 -> 8, discrete dynamics
        // also return x+ - x) or compute dx = model.step(x_k, u_k) - x_k here
        // (keeps NX = 4, but evaluates the NN twice per stage).

        // MultipleShooting adds h * lagrange with h = 1/N; scale by N so the
        // objective is the plain sum over stages, as in furutaCost.
        return static_cast<double>(N) * stage_cost<T>(state_t<T>(x), input_t<T>(u));
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
