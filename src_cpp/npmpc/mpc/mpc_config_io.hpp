#pragma once

#include <string>

#include "npmpc/mpc/problem/params.hpp"

namespace npmpc::mpc {

// Loads MPCParams from the YAML file written by
// scripts/export_mpc_config.py.
problem::MPCParams loadMPCParamsFromYaml(const std::string& path);

} // namespace npmpc::mpc
