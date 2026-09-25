#pragma once

#include <casadi/casadi.hpp>

#include "npmpc/dynamics/integrators/midpoint.hpp"
#include "npmpc/mpc/problem/MPCBase.hpp"
#include "npmpc/mpc/problem/params.hpp"

namespace npmpc::mpc::problem {

// Port of FurutaMPC (FurutaMPC.py), the equation-based counterpart of
// FurutaNPMPC. Only the CasADi path is ported: integrationConstraints (via
// MidpointIntegration::casadiImplicit on the analytical FurutaDynamics) and
// costFunction (via furutaCost). The torch-based integrate(), and
// linearize()/compute_terminal_P() (torch autograd + scipy ARE), are not
// ported -- terminalP must be precomputed in Python and supplied via
// CostParams (scripts/export_mpc_config.py --method equation).
class FurutaMPC : public MPCBase {
public:
    // p: plant parameters [lp, mp, lr, mr] (MPCParams::p).
    FurutaMPC(casadi::DM p, CostParams cost) : p_(std::move(p)), cost_(std::move(cost)) {}

    casadi::MX costFunction(const casadi::MX& x, const casadi::MX& u) const override;
    std::vector<casadi::MX> integrationConstraints(const casadi::MX& x, const casadi::MX& u, double dt) const override;

private:
    npmpc::dynamics::integrators::MidpointIntegration integrator_;
    casadi::DM p_;
    CostParams cost_;
};

} // namespace npmpc::mpc::problem
