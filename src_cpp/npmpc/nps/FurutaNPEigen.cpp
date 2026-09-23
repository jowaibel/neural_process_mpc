#include "npmpc/nps/FurutaNPEigen.hpp"

#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace npmpc::nps {

namespace {

template <int N>
Eigen::Vector<double, N> readVector(const YAML::Node& node, const std::string& what) {
    auto vals = node.as<std::vector<double>>();
    if (vals.size() != static_cast<size_t>(N)) {
        throw std::runtime_error("FurutaNPEigen::fromYaml: " + what + " has size " + std::to_string(vals.size()) +
                                 ", expected " + std::to_string(N));
    }
    return Eigen::Map<const Eigen::Vector<double, N>>(vals.data());
}

template <int Out, int In>
LinearLayerEigen<Out, In> readLayer(const YAML::Node& node, const std::string& what) {
    auto rows = node["weight"].as<std::vector<std::vector<double>>>();
    if (rows.size() != static_cast<size_t>(Out)) {
        throw std::runtime_error("FurutaNPEigen::fromYaml: " + what + ".weight has " + std::to_string(rows.size()) +
                                 " rows, expected " + std::to_string(Out));
    }
    LinearLayerEigen<Out, In> layer;
    for (int r = 0; r < Out; ++r) {
        if (rows[r].size() != static_cast<size_t>(In)) {
            throw std::runtime_error("FurutaNPEigen::fromYaml: " + what + ".weight has " +
                                     std::to_string(rows[r].size()) + " columns, expected " + std::to_string(In));
        }
        for (int c = 0; c < In; ++c) {
            layer.weight(r, c) = rows[r][c];
        }
    }
    layer.bias = readVector<Out>(node["bias"], what + ".bias");
    return layer;
}

} // namespace

FurutaNPEigen FurutaNPEigen::fromYaml(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);

    const YAML::Node layers = root["mu_decoder_layers"];
    if (!layers || layers.size() != 3) {
        throw std::runtime_error("FurutaNPEigen::fromYaml: expected 3 mu_decoder_layers in " + path);
    }
    MuDecoder muDecoder{
        readLayer<kHiddenDim, kDecoderInputDim>(layers[0], "mu_decoder_layers[0]"),
        readLayer<kHiddenDim, kHiddenDim>(layers[1], "mu_decoder_layers[1]"),
        readLayer<kNnOutputDim, kHiddenDim>(layers[2], "mu_decoder_layers[2]"),
    };

    Activation innerActivation = activationFromName(root["cnp_decoder"]["inner_activation"].as<std::string>());
    Activation muActivation = activationFromName(root["cnp_decoder"]["mu_activation"].as<std::string>());

    return FurutaNPEigen(std::move(muDecoder),
                         readVector<kNnInputDim>(root["x_direct_scaler"]["weight"], "x_direct_scaler.weight"),
                         readVector<kNnInputDim>(root["x_direct_scaler"]["bias"], "x_direct_scaler.bias"),
                         readVector<kNnOutputDim>(root["y_inverse_scaler"]["weight"], "y_inverse_scaler.weight"),
                         readVector<kNnOutputDim>(root["y_inverse_scaler"]["bias"], "y_inverse_scaler.bias"),
                         innerActivation, muActivation);
}

} // namespace npmpc::nps
