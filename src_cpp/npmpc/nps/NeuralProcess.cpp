#include "npmpc/nps/NeuralProcess.hpp"

namespace npmpc::nps {

casadi::MX NeuralProcess::casadiDecoder(const casadi::MX& x, const casadi::MX& z) {
    return decoder_.casadiForward(x, z, xDirectScalerWeight_, xDirectScalerBias_,
                                   yInverseScalerWeight_, yInverseScalerBias_);
}

} // namespace npmpc::nps
