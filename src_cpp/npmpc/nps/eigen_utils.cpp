#include "npmpc/nps/eigen_utils.hpp"

namespace npmpc::nps {

Activation activationFromName(const std::string& name) {
    if (name == "GELU") {
        return Activation::GELU;
    } else if (name == "ReLU") {
        return Activation::ReLU;
    } else if (name == "Sigmoid") {
        return Activation::Sigmoid;
    } else {
        return Activation::None;
    }
}

} // namespace npmpc::nps
