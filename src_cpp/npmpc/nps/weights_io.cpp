#include "npmpc/nps/weights_io.hpp"

#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace npmpc::nps {

NeuralProcess loadNeuralProcessFromYaml(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);

    std::vector<MlpLayer> muDecoderLayers;
    for (const auto& layerNode : root["mu_decoder_layers"]) {
        auto weight = layerNode["weight"].as<std::vector<std::vector<double>>>();
        auto bias = layerNode["bias"].as<std::vector<double>>();
        muDecoderLayers.push_back({casadi::DM(weight), casadi::DM(bias)});
    }
    if (muDecoderLayers.empty()) {
        throw std::runtime_error("loadNeuralProcessFromYaml: no mu_decoder_layers found in " + path);
    }

    std::string innerActivation = root["cnp_decoder"]["inner_activation"].as<std::string>();
    std::string muActivation = root["cnp_decoder"]["mu_activation"].as<std::string>();
    CNPDecoder decoder(std::move(muDecoderLayers), innerActivation, muActivation);

    casadi::DM xScalerWeight(root["x_direct_scaler"]["weight"].as<std::vector<double>>());
    casadi::DM xScalerBias(root["x_direct_scaler"]["bias"].as<std::vector<double>>());
    casadi::DM yScalerWeight(root["y_inverse_scaler"]["weight"].as<std::vector<double>>());
    casadi::DM yScalerBias(root["y_inverse_scaler"]["bias"].as<std::vector<double>>());

    return NeuralProcess(std::move(decoder), xScalerWeight, xScalerBias, yScalerWeight, yScalerBias);
}

} // namespace npmpc::nps
