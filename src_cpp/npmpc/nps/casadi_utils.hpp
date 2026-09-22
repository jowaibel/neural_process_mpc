#pragma once

#include <casadi/casadi.hpp>
#include <functional>
#include <string>
#include <vector>

namespace npmpc::nps {

using CasadiActivation = std::function<casadi::MX(const casadi::MX&)>;

// One Linear layer's weights (out_dim x in_dim) and bias (out_dim x 1),
// equivalent to one `{name}.{layer_id}.weight`/`.bias` entry in a torch
// state_dict, already resolved into layer order.
struct MlpLayer {
    casadi::DM weight;
    casadi::DM bias;
};

// Port of casadi_utils.create_casadi_scaler: elementwise x * weight + bias.
casadi::MX createCasadiScaler(const casadi::MX& x, const casadi::DM& weight, const casadi::DM& bias);

// Port of casadi_utils.create_casadi_mlp. `layers` gives the ordered Linear
// layers of the MLP; innerActivation is applied after every layer except the
// last, outerActivation is applied once at the very end.
casadi::MX createCasadiMlp(const casadi::MX& x, const std::vector<MlpLayer>& layers,
                            const CasadiActivation& innerActivation,
                            const CasadiActivation& outerActivation);

// Port of casadi_utils.get_casadi_activation.
CasadiActivation getCasadiActivation(const std::string& name);

casadi::MX casadiLinearLayer(const casadi::MX& input, const casadi::DM& weight, const casadi::DM& bias);
casadi::MX casadiGeluLayer(const casadi::MX& input);
casadi::MX casadiReluLayer(const casadi::MX& input);
casadi::MX casadiSigmoidLayer(const casadi::MX& input);
casadi::MX casadiNoLayer(const casadi::MX& input);
casadi::MX casadiSoftplusLayer(const casadi::MX& input);

} // namespace npmpc::nps
