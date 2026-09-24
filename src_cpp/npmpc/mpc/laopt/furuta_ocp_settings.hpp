#pragma once

// Dimensions and settings shared by the laopt Furuta OCPs (FurutaNPOcpEigen.hpp:
// NP dynamics, FurutaEqOcpEigen.hpp: analytical ODE), read from the YAML
// written by scripts/export_mpc_config.py.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

namespace npmpc::mpc::furuta_laopt {

inline constexpr int kNX = 4;                  // physical state [theta, phi, theta_dot, phi_dot]
inline constexpr int kNXOcp = 2 * kNX;         // OCP state [x; d], d_k = x_k - x_{k-1} (for the x_diff cost)
inline constexpr int kNU = 1;                  // motor torque
inline constexpr int kNP = kNX;                // one slack per physical state, shared by the whole horizon
inline constexpr int kNG = 2 * kNX;            // softened state bounds: [x - s; x + s]
inline constexpr int kNZ = 4;                  // NP latent size
inline constexpr int kNPlant = 4;              // analytical model parameters [lp, mp, lr, mr]

// Fixed-size counterpart of MPCParams/CostParams (params.hpp), read from the
// MPC config YAML without going through CasADi types. All vectors are in
// physical-state units (kNX). `z` (NP OCP) and `p` (equation OCP) are
// optional; each OCP checks for the one it needs.
struct FurutaOcpSettings {
    using StateVec = Eigen::Vector<double, kNX>;
    using InputVec = Eigen::Vector<double, kNU>;

    double dt{};
    bool hasZ{false};
    Eigen::Vector<double, kNZ> z;        // NP latent code (fixed during a solve)
    bool hasP{false};
    Eigen::Vector<double, kNPlant> p;    // analytical model parameters [lp, mp, lr, mr]

    StateVec xLb, xUb;                   // hard_bound.x (+-inf = unbounded)
    InputVec uLb, uUb;                   // hard_bound.u
    StateVec slackBound;                 // slack_bound.x (inf = hard constraint)
    double slackWeight{1000.0};          // MPCBase::slackWeight_

    StateVec costX;                      // cost.x (on [2(1-cos theta), phi^2, theta_dot^2, phi_dot^2])
    StateVec costXDiff;                  // cost.x_diff (on (x_{k+1} - x_k)^2)
    InputVec costU;                      // cost.u
    Eigen::Matrix<double, kNX, kNX> terminalP; // cost.terminal_p (LQR terminal cost)

    StateVec x0;                         // experiment_options.x0
    int simSteps{};                      // experiment_options.sim_steps (closed-loop run length = simSteps * dt)

    // Loads the MPC config YAML (as written by scripts/export_mpc_config.py).
    // Throws if horizon_steps != horizonSteps or any size differs from the
    // compile-time dimensions.
    static FurutaOcpSettings fromYaml(const std::string& path, int horizonSteps)
    {
        YAML::Node root = YAML::LoadFile(path);

        auto fail = [&](const std::string& msg) {
            throw std::runtime_error("FurutaOcpSettings::fromYaml(" + path + "): " + msg);
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

        FurutaOcpSettings s;
        s.dt = root["dt"].as<double>();
        if (root["z"]) {
            readVector(root["z"], s.z, "z");
            s.hasZ = true;
        }
        if (root["p"]) {
            readVector(root["p"], s.p, "p");
            s.hasP = true;
        }

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
        s.simSteps = root["experiment_options"]["sim_steps"].as<int>();
        return s;
    }
};

} // namespace npmpc::mpc::furuta_laopt
