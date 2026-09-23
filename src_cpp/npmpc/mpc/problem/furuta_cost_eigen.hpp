#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "npmpc/mpc/problem/params.hpp"
#include "npmpc/nps/FurutaNPEigen.hpp"

namespace npmpc::mpc::problem {

// Fixed-size Eigen counterpart of CostParams (params.hpp), trimmed to the
// terms furutaCost uses (x_end is unused there too).
struct CostParamsEigen {
    static constexpr int kStateDim = npmpc::nps::FurutaNPEigen::kStateDim;
    static constexpr int kInputDim = npmpc::nps::FurutaNPEigen::kInputDim;

    Eigen::Vector<double, kStateDim> x;
    Eigen::Vector<double, kStateDim> xDiff;
    Eigen::Vector<double, kInputDim> u;
    Eigen::Matrix<double, kStateDim, kStateDim> terminalP;
};

// Converts the CasADi-path CostParams (as loaded by loadMPCParamsFromYaml).
// Throws if any size differs from the compile-time dimensions.
CostParamsEigen toCostParamsEigen(const CostParams& cost);

// Eigen port of furutaCost (furuta_cost.hpp), templated on the scalar type T
// for AD. x is the (kStateDim, N+1) state trajectory and u the (kInputDim, N)
// input trajectory, one time step per column (as FurutaNPEigen::rollout).
template <int N, typename T>
T furutaCostEigen(const npmpc::nps::FurutaNPEigen::StateTrajectory<T, N>& x,
                  const npmpc::nps::FurutaNPEigen::InputTrajectory<T, N>& u,
                  const CostParamsEigen& cost) {
    using std::cos;
    using std::sin;
    using State = npmpc::nps::FurutaNPEigen::State<T>;
    using Input = npmpc::nps::FurutaNPEigen::Input<T>;

    const State wX = cost.x.template cast<T>();
    const State wXDiff = cost.xDiff.template cast<T>();
    const Input wU = cost.u.template cast<T>();

    T stateCost(0.0);
    T inputCost(0.0);
    T diffCost(0.0);
    for (int k = 0; k < N; ++k) {
        // Half-angle lift on theta: 2 * (1 - cos(theta)) == (2 * sin(theta / 2))^2.
        State stage;
        stage << T(2.0) * (T(1.0) - cos(x(0, k))), x(1, k) * x(1, k), x(2, k) * x(2, k), x(3, k) * x(3, k);
        stateCost += wX.dot(stage);

        Input uk = u.col(k);
        inputCost += wU.dot(uk.cwiseProduct(uk));

        State dx = x.col(k + 1) - x.col(k);
        diffCost += wXDiff.dot(dx.cwiseProduct(dx));
    }

    State eN;
    eN << T(2.0) * sin(x(0, N) / T(2.0)), x(1, N), x(2, N), x(3, N);
    T terminalCost = eN.dot(cost.terminalP.template cast<T>() * eN);

    return stateCost + inputCost + diffCost + terminalCost;
}

} // namespace npmpc::mpc::problem
