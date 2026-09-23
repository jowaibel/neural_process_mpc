#pragma once

#include <cmath>

#include "laopt/laopt.hpp"

// erf for laopt's AD scalar, which (unlike sin/cos/exp/...) laopt does not
// provide. Needed by eigenGeluLayer (eigen_utils.hpp): the activation switch
// instantiates every branch for the AD scalar type.
//
// Written like laopt's own unary functions (EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY
// in laopt/autodiff/autodiff_scalar.hpp): exact value via erf, exact
// derivative d/dx erf(x) = 2/sqrt(pi) * exp(-x^2). The inner erf/exp calls are
// unqualified, so nested AD scalars (second derivatives) recurse into this
// overload via ADL.
//
// For the sparsity pass, the value type is laopt::TouchableDerivative. erf is
// odd with erf(0) = 0, so it forwards the "touched" state like sin/tanh do
// (LAOPT_TOUCHABLE_DERIVATIVE_DECLARE_GLOBAL_UNARY_FORWARD in
// laopt/autodiff/touchable_derivative.hpp).
namespace laopt {

template<typename Scalar>
inline TouchableDerivative<Scalar> erf(const TouchableDerivative<Scalar>& x)
{
    return x;
}

template<typename DerType>
inline const laopt::AutoDiffScalar<
    EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE_NS(typename Eigen::internal::remove_all<DerType>::type,
                                              typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar,
                                              product)>
erf(const laopt::AutoDiffScalar<DerType>& x)
{
    using namespace Eigen;
    typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar;
    using std::erf;
    using std::exp;
    const Scalar v = x.value();
    return laopt::laopt_make_autodiff_scalar(erf(v), x.derivatives() * (Scalar(M_2_SQRTPI) * exp(-v * v)));
}

} // namespace laopt
