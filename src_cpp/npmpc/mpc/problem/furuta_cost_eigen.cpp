#include "npmpc/mpc/problem/furuta_cost_eigen.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace npmpc::mpc::problem {

namespace {

template <int N>
Eigen::Vector<double, N> toFixedVector(const std::vector<double>& vals, const std::string& what) {
    if (vals.size() != static_cast<size_t>(N)) {
        throw std::runtime_error("toCostParamsEigen: cost." + what + " has size " + std::to_string(vals.size()) +
                                 ", expected " + std::to_string(N));
    }
    return Eigen::Map<const Eigen::Vector<double, N>>(vals.data());
}

} // namespace

CostParamsEigen toCostParamsEigen(const CostParams& cost) {
    constexpr int nx = CostParamsEigen::kStateDim;
    constexpr int nu = CostParamsEigen::kInputDim;

    CostParamsEigen out;
    out.x = toFixedVector<nx>(cost.x, "x");
    out.xDiff = toFixedVector<nx>(cost.xDiff, "x_diff");
    out.u = toFixedVector<nu>(cost.u, "u");

    if (cost.terminalP.size1() != nx || cost.terminalP.size2() != nx) {
        throw std::runtime_error("toCostParamsEigen: cost.terminal_p is " + std::to_string(cost.terminalP.size1()) +
                                 "x" + std::to_string(cost.terminalP.size2()) + ", expected " +
                                 std::to_string(nx) + "x" + std::to_string(nx));
    }
    for (int r = 0; r < nx; ++r) {
        for (int c = 0; c < nx; ++c) {
            out.terminalP(r, c) = static_cast<double>(cost.terminalP(r, c));
        }
    }
    return out;
}

} // namespace npmpc::mpc::problem
