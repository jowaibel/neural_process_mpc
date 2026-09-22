#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npmpc::mpc {

// State message received from the Python Qube server: {"t": <time>, "x": [...]}.
struct QubeState {
    double t;
    std::vector<double> x;
};

// A minimal TCP client for the length-prefixed JSON wire protocol spoken by
// scripts/run_qube_server.py: each message is a 4-byte big-endian length
// prefix followed by that many bytes of JSON. Only the two flat message
// shapes used by that protocol are supported ({"t":...,"x":[...]} in,
// {"u":[...]} out) -- this is not a general JSON library.
class QubeClient {
public:
    QubeClient(const std::string& host, int port);
    ~QubeClient();

    QubeClient(const QubeClient&) = delete;
    QubeClient& operator=(const QubeClient&) = delete;

    // Blocks until a state message arrives. Returns false if the connection
    // was closed by the server.
    bool receiveState(QubeState& state);

    // Sends the first optimal input as {"u": [...]}.
    void sendInput(const std::vector<double>& u);

private:
    int socketFd_;
};

} // namespace npmpc::mpc
