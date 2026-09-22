#include "npmpc/mpc/qube_client.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace npmpc::mpc {

namespace {

void sendAll(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, 0);
        if (n <= 0) {
            throw std::runtime_error("QubeClient: send() failed");
        }
        sent += static_cast<size_t>(n);
    }
}

// Reads exactly len bytes. Returns false if the peer closed the connection
// before any bytes of this read were received (a clean EOF); throws on a
// partial read cut short by EOF or an error.
bool recvAll(int fd, char* data, size_t len) {
    size_t received = 0;
    while (received < len) {
        ssize_t n = ::recv(fd, data + received, len - received, 0);
        if (n == 0) {
            if (received == 0) return false;
            throw std::runtime_error("QubeClient: connection closed mid-message");
        }
        if (n < 0) {
            throw std::runtime_error("QubeClient: recv() failed");
        }
        received += static_cast<size_t>(n);
    }
    return true;
}

// Extracts the numeric contents of a top-level JSON array field, e.g. for
// `key="x"` and payload `{"t": 1.0, "x": [1, 2, 3]}` returns [1, 2, 3].
// Only handles this specific flat shape (no nesting, no strings in values).
std::vector<double> extractArray(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t keyPos = json.find(needle);
    if (keyPos == std::string::npos) {
        throw std::runtime_error("QubeClient: missing field \"" + key + "\" in: " + json);
    }
    size_t open = json.find('[', keyPos);
    size_t close = json.find(']', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("QubeClient: malformed array for \"" + key + "\" in: " + json);
    }
    std::vector<double> values;
    std::stringstream ss(json.substr(open + 1, close - open - 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) values.push_back(std::stod(item));
    }
    return values;
}

double extractNumber(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t keyPos = json.find(needle);
    if (keyPos == std::string::npos) {
        throw std::runtime_error("QubeClient: missing field \"" + key + "\" in: " + json);
    }
    size_t colon = json.find(':', keyPos);
    size_t end = json.find_first_of(",}", colon);
    return std::stod(json.substr(colon + 1, end - colon - 1));
}

std::string encodeInput(const std::vector<double>& u) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "{\"u\":[";
    for (size_t i = 0; i < u.size(); ++i) {
        if (i > 0) oss << ",";
        oss << u[i];
    }
    oss << "]}";
    return oss.str();
}

} // namespace

QubeClient::QubeClient(const std::string& host, int port) : socketFd_(-1) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    std::string portStr = std::to_string(port);
    if (getaddrinfo(host.c_str(), portStr.c_str(), &hints, &result) != 0) {
        throw std::runtime_error("QubeClient: getaddrinfo failed for " + host + ":" + portStr);
    }

    int fd = -1;
    for (addrinfo* p = result; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
            socketFd_ = fd;
            break;
        }
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(result);

    if (socketFd_ < 0) {
        throw std::runtime_error("QubeClient: could not connect to " + host + ":" + portStr);
    }
}

QubeClient::~QubeClient() {
    if (socketFd_ >= 0) {
        ::close(socketFd_);
    }
}

bool QubeClient::receiveState(QubeState& state) {
    uint32_t lengthNet = 0;
    if (!recvAll(socketFd_, reinterpret_cast<char*>(&lengthNet), sizeof(lengthNet))) {
        return false;
    }
    uint32_t length = ntohl(lengthNet);

    std::string payload(length, '\0');
    if (!recvAll(socketFd_, payload.data(), length)) {
        return false;
    }

    state.x = extractArray(payload, "x");
    state.t = extractNumber(payload, "t");
    return true;
}

void QubeClient::sendInput(const std::vector<double>& u) {
    std::string payload = encodeInput(u);
    uint32_t lengthNet = htonl(static_cast<uint32_t>(payload.size()));
    sendAll(socketFd_, reinterpret_cast<const char*>(&lengthNet), sizeof(lengthNet));
    sendAll(socketFd_, payload.data(), payload.size());
}

} // namespace npmpc::mpc
