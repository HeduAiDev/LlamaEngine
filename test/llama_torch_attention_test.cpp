#include "llama_torch.hpp"

int main() {
    using namespace llama_torch;
    auto attention = std::make_shared<Attention>(
        3,
        30,
        768,
        32,
        8
    );
    Tensor input = torch::randn({2, 20, 768});
    Tensor mask;
    RotaryEmbedding rotary(768 / 32, 30, 50000.0);

    auto out = attention->forward(input, 0, rotary, mask);
    std::cout << out.sizes() << std::endl;
    std::cout << out << std::endl;
    return 0;
}