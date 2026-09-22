#include "furuta_np.hpp"

namespace npmpc::dynamics::integrators {

std::vector<casadi::MX> FurutaNPIntegration::casadiImplicit(
        const casadi::MX& x, const casadi::MX& u, double dt, const casadi::MX& z) const {
    std::vector<casadi::MX> constraints;
    constraints.reserve(static_cast<size_t>(u.size1()));

    for (casadi_int i = 0; i < u.size1(); ++i) {
        casadi::MX x_nn = casadi::MX::horzcat({
            casadi::MX::sin(casadi::MX(x(i, 0))),
            casadi::MX::cos(casadi::MX(x(i, 0))),
            x(i, casadi::Slice(2, 4)),
            u(i, casadi::Slice())
        });
        casadi::MX y_nn = np_->casadiDecoder(x_nn, z);
        casadi::MX x_next = x(i, casadi::Slice())
            + casadi::MX::horzcat({dt * (x(i, casadi::Slice(2, 4)) + y_nn / 2), y_nn});
        constraints.push_back(x(i + 1, casadi::Slice()) == x_next);
    }
    return constraints;
}

casadi::MX FurutaNPIntegration::casadiExplicit(
        const casadi::MX& x0, const casadi::MX& u, double dt, const casadi::MX& z) const {
    casadi_int n_steps = u.size1();
    casadi::MX x = casadi::MX::zeros(n_steps + 1, x0.size2());
    x(0, casadi::Slice()) = x0;

    for (casadi_int i = 0; i < n_steps; ++i) {
        casadi::MX x_nn = casadi::MX::horzcat({
            casadi::MX::sin(casadi::MX(x(i, 0))),
            casadi::MX::cos(casadi::MX(x(i, 0))),
            x(i, casadi::Slice(2, 4)),
            u(i, casadi::Slice())
        });
        casadi::MX y_nn = np_->casadiDecoder(x_nn, z);
        x(i + 1, casadi::Slice()) = x(i, casadi::Slice())
            + casadi::MX::horzcat({dt * (x(i, casadi::Slice(2, 4)) + y_nn / 2), y_nn});
    }
    return x;
}

} // namespace npmpc::dynamics::integrators
