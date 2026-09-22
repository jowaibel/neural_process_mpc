#include "npmpc/mpc/problem/furuta_cost.hpp"

namespace npmpc::mpc::problem {

casadi::MX furutaCost(const casadi::MX& x, const casadi::MX& u, const CostParams& cost) {
    casadi_int N = u.size1();
    casadi::Slice stage(0, N);
    casadi::Slice shifted(1, N + 1);

    casadi::MX stateStage =
        cost.x[0] * 2.0 * (1.0 - casadi::MX::cos(x(stage, 0)))
        + cost.x[1] * casadi::MX::pow(x(stage, 1), 2)
        + cost.x[2] * casadi::MX::pow(x(stage, 2), 2)
        + cost.x[3] * casadi::MX::pow(x(stage, 3), 2);
    casadi::MX stateCost = casadi::MX::sum1(stateStage);
    casadi::MX inputCost = casadi::MX::sum1(cost.u[0] * casadi::MX::pow(u(stage, 0), 2));

    casadi::MX dx = x(shifted, casadi::Slice()) - x(stage, casadi::Slice());
    casadi::MX diffStage =
        cost.xDiff[0] * casadi::MX::pow(dx(casadi::Slice(), 0), 2)
        + cost.xDiff[1] * casadi::MX::pow(dx(casadi::Slice(), 1), 2)
        + cost.xDiff[2] * casadi::MX::pow(dx(casadi::Slice(), 2), 2)
        + cost.xDiff[3] * casadi::MX::pow(dx(casadi::Slice(), 3), 2);
    casadi::MX diffCost = casadi::MX::sum1(diffStage);

    casadi::MX xN = x(x.size1() - 1, casadi::Slice());
    casadi::MX eN = casadi::MX::horzcat({
        2.0 * casadi::MX::sin(xN(0, 0) / 2.0),
        xN(0, 1), xN(0, 2), xN(0, 3)
    });
    casadi::MX terminalCost = casadi::MX::mtimes(casadi::MX::mtimes(eN, casadi::MX(cost.terminalP)), eN.T());

    return stateCost + inputCost + diffCost + terminalCost;
}

} // namespace npmpc::mpc::problem
