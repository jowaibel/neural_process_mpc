#pragma once

#include <casadi/casadi.hpp>

#include "npmpc/dynamics/integrators/furuta_np.hpp"
#include "npmpc/mpc/problem/MPCBase.hpp"
#include "npmpc/mpc/problem/params.hpp"

namespace npmpc::mpc::problem {

// Port of FurutaNPMPC (FurutaNPMPC.py). Only the CasADi path is ported:
// integrationConstraints (via FurutaNPIntegration::casadiImplicit) and
// costFunction (via furutaCost). The torch-based integrate(), and
// linearize()/computeTerminalP() (torch autograd + scipy ARE), are not
// ported -- terminalP must be precomputed in Python and supplied via
// CostParams (see scripts/export_mpc_config.py). z is likewise a fixed
// latent supplied from outside (the NP encoder is not part of this port).
class FurutaNPMPC : public MPCBase {
public:
    FurutaNPMPC(npmpc::dynamics::integrators::INeuralProcessCasadi* np, casadi::DM zCasadi, CostParams cost)
        : integrator_(np), zCasadi_(std::move(zCasadi)), cost_(std::move(cost)) {}

    casadi::MX costFunction(const casadi::MX& x, const casadi::MX& u) const override;
    std::vector<casadi::MX> integrationConstraints(const casadi::MX& x, const casadi::MX& u, double dt) const override;

private:
    npmpc::dynamics::integrators::FurutaNPIntegration integrator_;
    casadi::DM zCasadi_;
    CostParams cost_;
};

} // namespace npmpc::mpc::problem
