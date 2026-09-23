#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <string>

#include "npmpc/nps/eigen_utils.hpp"

namespace npmpc::nps {

// Numeric (Eigen) port of the CasADi decoder path (CNPDecoder + NeuralProcess
// in decoder.hpp/NeuralProcess.hpp) plus the Furuta NP integrator step
// (FurutaNPIntegration in dynamics/integrators/furuta_np.hpp), bundled into
// one class since a non-symbolic optimizer (unlike CasADi/Opti) needs a plain
// forward/step function rather than separate symbolic-graph builders.
//
// All model dimensions are fixed at compile time; only the weights and
// activations are read at runtime (fromYaml checks that the file matches).
// decode/step/rollout are templated on the scalar type T so they can be
// evaluated with AD scalars; weights stay double and are cast to T.
class FurutaNPEigen {
public:
    static constexpr int kStateDim = 4;        // [theta, phi, theta_dot, phi_dot]
    static constexpr int kInputDim = 1;        // motor torque
    static constexpr int kNnInputDim = 5;      // [sin(theta), cos(theta), theta_dot, phi_dot, u]
    static constexpr int kNnOutputDim = 2;     // [theta_ddot, phi_ddot] (per-step velocity increments)
    static constexpr int kLatentDim = 4;       // NP latent z
    static constexpr int kHiddenDim = 32;      // mu_decoder hidden width
    static constexpr int kDecoderInputDim = kNnInputDim + kLatentDim;

    template <typename T> using State = Eigen::Vector<T, kStateDim>;
    template <typename T> using Input = Eigen::Vector<T, kInputDim>;
    template <typename T> using NnInput = Eigen::Vector<T, kNnInputDim>;
    template <typename T> using NnOutput = Eigen::Vector<T, kNnOutputDim>;
    template <typename T> using Latent = Eigen::Vector<T, kLatentDim>;
    // Column k is the state/input at time step k.
    template <typename T, int N> using StateTrajectory = Eigen::Matrix<T, kStateDim, N + 1>;
    template <typename T, int N> using InputTrajectory = Eigen::Matrix<T, kInputDim, N>;

    // mu_decoder: kDecoderInputDim -> kHiddenDim -> kHiddenDim -> kNnOutputDim.
    struct MuDecoder {
        LinearLayerEigen<kHiddenDim, kDecoderInputDim> layer0;
        LinearLayerEigen<kHiddenDim, kHiddenDim> layer1;
        LinearLayerEigen<kNnOutputDim, kHiddenDim> layer2;
    };

    FurutaNPEigen(MuDecoder muDecoder,
                  Eigen::Vector<double, kNnInputDim> xScalerWeight, Eigen::Vector<double, kNnInputDim> xScalerBias,
                  Eigen::Vector<double, kNnOutputDim> yScalerWeight, Eigen::Vector<double, kNnOutputDim> yScalerBias,
                  Activation innerActivation = Activation::GELU,
                  Activation muActivation = Activation::None)
        : muDecoder_(std::move(muDecoder)),
          xScalerWeight_(xScalerWeight), xScalerBias_(xScalerBias),
          yScalerWeight_(yScalerWeight), yScalerBias_(yScalerBias),
          innerActivation_(innerActivation), muActivation_(muActivation) {}

    // Loads the weights YAML written by scripts/export_np_weights.py (the
    // same file loadNeuralProcessFromYaml reads for the CasADi path). Throws
    // if the file's dimensions differ from the compile-time ones above.
    static FurutaNPEigen fromYaml(const std::string& path);

    // Port of CNPDecoder.casadi_forward / NeuralProcess.casadi_decoder.
    template <typename T>
    NnOutput<T> decode(const NnInput<T>& x, const Latent<T>& z) const {
        Eigen::Vector<T, kDecoderInputDim> decoderInput;
        decoderInput << eigenScaler(x, xScalerWeight_, xScalerBias_), z;

        Eigen::Vector<T, kHiddenDim> h0 =
            eigenActivation(innerActivation_, eigenLinearLayer(decoderInput, muDecoder_.layer0));
        Eigen::Vector<T, kHiddenDim> h1 =
            eigenActivation(innerActivation_, eigenLinearLayer(h0, muDecoder_.layer1));
        NnOutput<T> yScaled = eigenActivation(muActivation_, eigenLinearLayer(h1, muDecoder_.layer2));
        return eigenScaler(yScaled, yScalerWeight_, yScalerBias_);
    }

    // Port of FurutaNPIntegration::casadiExplicit's per-step body: one
    // midpoint-style integration step from state x and input u to x_next.
    template <typename T>
    State<T> step(const State<T>& x, const Input<T>& u, const Latent<T>& z, double dt) const {
        using std::cos;
        using std::sin;
        NnInput<T> xNn;
        xNn << sin(x(0)), cos(x(0)), x(2), x(3), u(0);

        NnOutput<T> yNn = decode(xNn, z);

        State<T> xNext;
        xNext.template head<2>() = x.template head<2>() + T(dt) * (x.template tail<2>() + yNn / T(2.0));
        xNext.template tail<2>() = x.template tail<2>() + yNn;
        return xNext;
    }

    // Port of FurutaNPIntegration::casadiExplicit: rolls step() forward over
    // an N-step input sequence, returning the N+1 state trajectory with
    // column 0 == x0.
    template <int N, typename T>
    StateTrajectory<T, N> rollout(const State<T>& x0, const InputTrajectory<T, N>& u,
                                  const Latent<T>& z, double dt) const {
        StateTrajectory<T, N> x;
        x.col(0) = x0;
        for (int i = 0; i < N; ++i) {
            x.col(i + 1) = step<T>(x.col(i), u.col(i), z, dt);
        }
        return x;
    }

private:
    MuDecoder muDecoder_;
    Eigen::Vector<double, kNnInputDim> xScalerWeight_, xScalerBias_;
    Eigen::Vector<double, kNnOutputDim> yScalerWeight_, yScalerBias_;
    Activation innerActivation_;
    Activation muActivation_;
};

} // namespace npmpc::nps
