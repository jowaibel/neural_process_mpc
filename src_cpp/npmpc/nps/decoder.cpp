#include "npmpc/nps/decoder.hpp"

namespace npmpc::nps {

casadi::MX CNPDecoder::casadiForward(const casadi::MX& x, const casadi::MX& z,
                                      const casadi::DM& xScalerWeight, const casadi::DM& xScalerBias,
                                      const casadi::DM& yInvScalerWeight, const casadi::DM& yInvScalerBias) const {
    casadi::MX xScaled = createCasadiScaler(x, xScalerWeight, xScalerBias);
    casadi::MX zRow = z.size1() > z.size2() ? z.T() : z;
    casadi::MX decoderInput = casadi::MX::horzcat({xScaled, zRow});

    casadi::MX yScaled = createCasadiMlp(decoderInput, muDecoderLayers_,
                                          getCasadiActivation(innerActivationName_),
                                          getCasadiActivation(muActivationName_));
    return createCasadiScaler(yScaled, yInvScalerWeight, yInvScalerBias);
}

} // namespace npmpc::nps
