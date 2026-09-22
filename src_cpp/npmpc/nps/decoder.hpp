#pragma once

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

#include "npmpc/nps/casadi_utils.hpp"

namespace npmpc::nps {

// Port of CNPDecoder (decoder.py): only the CasADi decoder path
// (casadi_forward) is ported. The torch forward()/loss() methods, and the
// sigma_decoder (unused by casadi_forward), are intentionally left out.
class CNPDecoder {
public:
    CNPDecoder(std::vector<MlpLayer> muDecoderLayers,
               std::string innerActivationName = "GELU",
               std::string muActivationName = "None")
        : muDecoderLayers_(std::move(muDecoderLayers)),
          innerActivationName_(std::move(innerActivationName)),
          muActivationName_(std::move(muActivationName)) {}

    // Port of CNPDecoder.casadi_forward. x_scaler_*/y_inv_scaler_* give the
    // elementwise (weight, bias) pairs that were the x_direct_scaler and
    // y_inverse_scaler state_dicts in the Python version.
    casadi::MX casadiForward(const casadi::MX& x, const casadi::MX& z,
                              const casadi::DM& xScalerWeight, const casadi::DM& xScalerBias,
                              const casadi::DM& yInvScalerWeight, const casadi::DM& yInvScalerBias) const;

private:
    std::vector<MlpLayer> muDecoderLayers_;
    std::string innerActivationName_;
    std::string muActivationName_;
};

} // namespace npmpc::nps
