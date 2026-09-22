#pragma once

#include <string>

#include "npmpc/nps/NeuralProcess.hpp"

namespace npmpc::nps {

// Loads a NeuralProcess from the YAML file written by
// scripts/export_np_weights.py (mu_decoder layers + x_direct_scaler /
// y_inverse_scaler), for use with the CasADi decoder path.
NeuralProcess loadNeuralProcessFromYaml(const std::string& path);

} // namespace npmpc::nps
