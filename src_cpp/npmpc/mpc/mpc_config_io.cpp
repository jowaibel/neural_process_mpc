#include "npmpc/mpc/mpc_config_io.hpp"

#include <vector>

#include <yaml-cpp/yaml.h>

namespace npmpc::mpc {

using npmpc::mpc::problem::Bound;
using npmpc::mpc::problem::HardBound;
using npmpc::mpc::problem::CostParams;
using npmpc::mpc::problem::MPCParams;

namespace {

std::vector<Bound> readBounds(const YAML::Node& node) {
    std::vector<Bound> bounds;
    for (const auto& pairNode : node) {
        auto pair = pairNode.as<std::vector<double>>();
        bounds.push_back({pair.at(0), pair.at(1)});
    }
    return bounds;
}

} // namespace

problem::MPCParams loadMPCParamsFromYaml(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);

    MPCParams params;
    params.dt = root["dt"].as<double>();
    params.horizonSteps = root["horizon_steps"].as<int>();
    params.xSize = root["x_size"].as<int>();
    params.uSize = root["u_size"].as<int>();
    params.solver = root["solver"].as<std::string>();
    params.z = casadi::DM(root["z"].as<std::vector<double>>());

    params.hardBound.x = readBounds(root["hard_bound"]["x"]);
    params.hardBound.u = readBounds(root["hard_bound"]["u"]);
    params.slackBoundX = root["slack_bound"]["x"].as<std::vector<double>>();

    CostParams cost;
    cost.x = root["cost"]["x"].as<std::vector<double>>();
    cost.xDiff = root["cost"]["x_diff"].as<std::vector<double>>();
    cost.xEnd = root["cost"]["x_end"].as<std::vector<double>>();
    cost.u = root["cost"]["u"].as<std::vector<double>>();
    cost.terminalP = casadi::DM(root["cost"]["terminal_p"].as<std::vector<std::vector<double>>>());
    params.cost = std::move(cost);

    params.x0 = casadi::DM(root["experiment_options"]["x0"].as<std::vector<double>>());
    params.simSteps = root["experiment_options"]["sim_steps"].as<int>();

    return params;
}

} // namespace npmpc::mpc
