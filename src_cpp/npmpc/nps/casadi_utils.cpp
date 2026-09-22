#include "npmpc/nps/casadi_utils.hpp"

namespace npmpc::nps {

casadi::MX createCasadiScaler(const casadi::MX& x, const casadi::DM& weight, const casadi::DM& bias) {
    return x * weight.T() + bias.T();
}

casadi::MX createCasadiMlp(const casadi::MX& x, const std::vector<MlpLayer>& layers,
                            const CasadiActivation& innerActivation,
                            const CasadiActivation& outerActivation) {
    casadi::MX current = x;
    for (size_t i = 0; i < layers.size(); ++i) {
        current = casadiLinearLayer(current, layers[i].weight, layers[i].bias);
        if (i + 1 < layers.size()) {
            current = innerActivation(current);
        }
    }
    return outerActivation(current);
}

CasadiActivation getCasadiActivation(const std::string& name) {
    if (name == "GELU") {
        return casadiGeluLayer;
    } else if (name == "ReLU") {
        return casadiReluLayer;
    } else if (name == "Sigmoid") {
        return casadiSigmoidLayer;
    } else {
        return casadiNoLayer;
    }
}

casadi::MX casadiLinearLayer(const casadi::MX& input, const casadi::DM& weight, const casadi::DM& bias) {
    return casadi::MX::mtimes(input, weight.T()) + bias.T();
}

casadi::MX casadiGeluLayer(const casadi::MX& input) {
    return input * 0.5 * (1 + casadi::MX::erf(input / std::sqrt(2.0)));
}

casadi::MX casadiReluLayer(const casadi::MX& input) {
    return casadi::MX::fmax(0, input);
}

casadi::MX casadiSigmoidLayer(const casadi::MX& input) {
    return 1 / (1 + casadi::MX::exp(-input));
}

casadi::MX casadiNoLayer(const casadi::MX& input) {
    return input;
}

casadi::MX casadiSoftplusLayer(const casadi::MX& input) {
    return casadi::MX::log(1 + casadi::MX::exp(input));
}

} // namespace npmpc::nps
