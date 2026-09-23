#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <string>

namespace npmpc::nps {

// Eigen counterparts of casadi_utils.hpp. All functions are templated on the
// scalar type T (double, or an AD scalar) and use fixed-size Eigen types.
// Weights are always stored as double and cast to T on evaluation, so the
// same loaded model can be evaluated with any scalar type.
//
// Elementwise math uses unqualified calls after `using std::...` so that
// AD scalar types can provide their overloads via ADL.

// One Linear layer's weights (Out x In) and bias (Out), the fixed-size Eigen
// counterpart of MlpLayer (casadi_utils.hpp).
template <int Out, int In>
struct LinearLayerEigen {
    Eigen::Matrix<double, Out, In> weight;
    Eigen::Vector<double, Out> bias;
};

// Runtime-selectable activation (read from the weights file), replacing the
// std::function returned by getCasadiActivation.
enum class Activation { None, GELU, ReLU, Sigmoid, Softplus };

// Port of casadi_utils.get_casadi_activation: unknown names map to None.
Activation activationFromName(const std::string& name);

// Port of casadi_utils.create_casadi_scaler: elementwise x * weight + bias.
template <typename T, int N>
Eigen::Vector<T, N> eigenScaler(const Eigen::Vector<T, N>& x,
                                const Eigen::Vector<double, N>& weight,
                                const Eigen::Vector<double, N>& bias) {
    return x.cwiseProduct(weight.template cast<T>()) + bias.template cast<T>();
}

template <typename T, int Out, int In>
Eigen::Vector<T, Out> eigenLinearLayer(const Eigen::Vector<T, In>& input, const LinearLayerEigen<Out, In>& layer) {
    return layer.weight.template cast<T>() * input + layer.bias.template cast<T>();
}

template <typename T, int N>
Eigen::Vector<T, N> eigenGeluLayer(const Eigen::Vector<T, N>& input) {
    return input.unaryExpr([](const T& v) -> T {
        using std::erf;
        return v * T(0.5) * (T(1.0) + erf(v / T(std::sqrt(2.0))));
    });
}

template <typename T, int N>
Eigen::Vector<T, N> eigenReluLayer(const Eigen::Vector<T, N>& input) {
    return input.unaryExpr([](const T& v) -> T { return v > T(0.0) ? v : T(0.0); });
}

template <typename T, int N>
Eigen::Vector<T, N> eigenSigmoidLayer(const Eigen::Vector<T, N>& input) {
    return input.unaryExpr([](const T& v) -> T {
        using std::exp;
        return T(1.0) / (T(1.0) + exp(-v));
    });
}

template <typename T, int N>
Eigen::Vector<T, N> eigenSoftplusLayer(const Eigen::Vector<T, N>& input) {
    return input.unaryExpr([](const T& v) -> T {
        using std::exp;
        using std::log;
        return log(T(1.0) + exp(v));
    });
}

template <typename T, int N>
Eigen::Vector<T, N> eigenActivation(Activation activation, const Eigen::Vector<T, N>& input) {
    switch (activation) {
        case Activation::GELU: return eigenGeluLayer(input);
        case Activation::ReLU: return eigenReluLayer(input);
        case Activation::Sigmoid: return eigenSigmoidLayer(input);
        case Activation::Softplus: return eigenSoftplusLayer(input);
        case Activation::None: break;
    }
    return input;
}

} // namespace npmpc::nps
