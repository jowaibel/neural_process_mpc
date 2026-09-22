#pragma once

#include <casadi/casadi.hpp>

#include "npmpc/dynamics/integrators/furuta_np.hpp"
#include "npmpc/nps/decoder.hpp"

namespace npmpc::nps {

// Port of NeuralProcess (NeuralProcess.py): only the CasADi decoder path
// (casadi_decoder) is ported. The encoder, torch encode()/decode(), and
// checkpoint loading are not part of this port; weights/scalers must be
// supplied directly to the constructor for now (TODO: load from a trained
// checkpoint once an export format is decided).
class NeuralProcess : public npmpc::dynamics::integrators::INeuralProcessCasadi {
public:
    NeuralProcess(CNPDecoder decoder,
                  casadi::DM xDirectScalerWeight, casadi::DM xDirectScalerBias,
                  casadi::DM yInverseScalerWeight, casadi::DM yInverseScalerBias)
        : decoder_(std::move(decoder)),
          xDirectScalerWeight_(std::move(xDirectScalerWeight)),
          xDirectScalerBias_(std::move(xDirectScalerBias)),
          yInverseScalerWeight_(std::move(yInverseScalerWeight)),
          yInverseScalerBias_(std::move(yInverseScalerBias)) {}

    // Port of NeuralProcess.casadi_decoder.
    casadi::MX casadiDecoder(const casadi::MX& x, const casadi::MX& z) override;

private:
    CNPDecoder decoder_;
    casadi::DM xDirectScalerWeight_;
    casadi::DM xDirectScalerBias_;
    casadi::DM yInverseScalerWeight_;
    casadi::DM yInverseScalerBias_;
};

} // namespace npmpc::nps
