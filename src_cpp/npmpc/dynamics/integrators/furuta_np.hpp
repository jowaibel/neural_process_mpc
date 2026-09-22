#pragma once

#include <casadi/casadi.hpp>
#include <vector>

namespace npmpc::dynamics::integrators {

// TODO: replace with a proper C++ port of NeuralProcess. Mirrors
// NeuralProcess.casadi_decoder(x, z) -> y_nn, a (1, y_dim) row vector.
class INeuralProcessCasadi {
public:
    virtual ~INeuralProcessCasadi() = default;
    virtual casadi::MX casadiDecoder(const casadi::MX& x, const casadi::MX& z) = 0;
};

// Port of FurutaNPIntegration (furuta_np.py). Only the CasADi-based
// integration methods are ported; the torch-based integrate() and
// montecarlo_integrate() are intentionally left out of this port.
class FurutaNPIntegration {
public:
    explicit FurutaNPIntegration(INeuralProcessCasadi* np) : np_(np) {}

    // CasADi implicit integration constraints: x[i+1] == f(x[i], u[i]).
    std::vector<casadi::MX> casadiImplicit(const casadi::MX& x, const casadi::MX& u,
                                            double dt, const casadi::MX& z = casadi::MX()) const;

    // CasADi explicit integration: returns the full predicted trajectory.
    casadi::MX casadiExplicit(const casadi::MX& x0, const casadi::MX& u,
                               double dt, const casadi::MX& z = casadi::MX()) const;

private:
    INeuralProcessCasadi* np_;
};

} // namespace npmpc::dynamics::integrators
