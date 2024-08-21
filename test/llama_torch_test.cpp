#include "llama_torch.hpp"

int main() {
    using namespace llama_torch;


    auto args = ModelArgs({
    /*dim*/        4096,
    /*n_heads*/    32, 
    /*n_kv_heads*/ 8, 
    /*n_layers*/   32, 
    /*rope_theta*/ 500000.0, 
    // /*vocab_size*/ 128256, 
    /*vocab_size*/ 1024, 
    /*hidden_dim*/ 14336, 
    /*norm_eps*/   1e-05, 
    /*max_seq_len*/1, 
    /*max_batch_size*/ 1
    });
    auto transformer = std::make_shared<Transformer>(args);
    std::cout << transformer << std::endl;
    transformer -> to(torch::kCUDA);
    Tensor tokens = torch::randint(args.vocab_size, {1, 20});
    auto out = transformer->forward(tokens, 0);
    std::cout << out.sizes() << std::endl;
    std::cout << out << std::endl;
    return 0;
}