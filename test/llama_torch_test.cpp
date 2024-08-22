#include "llama_torch.hpp"

int main() {
    using namespace llama_torch;


    auto args = ModelArgs({
    /*dim*/        768,
    /*n_heads*/    32, 
    /*n_kv_heads*/ 8, 
    /*n_layers*/   32, 
    /*rope_theta*/ 500000.0, 
    // /*vocab_size*/ 128256, 
    /*vocab_size*/ 1024, 
    /*hidden_dim*/ 1024, 
    /*norm_eps*/   1e-05, 
    /*max_seq_len*/30,
    /*max_batch_size*/ 2
    });
    auto transformer = Transformer(args);
    std::cout << (transformer) << std::endl;
    Tensor tokens = torch::randint(args.vocab_size, {1, 20});
    transformer -> to(torch::kCUDA);
    tokens = tokens.to(torch::kCUDA);
    auto out = transformer->forward(tokens, 0);
    std::cout << out.sizes() << std::endl;
    // std::cout << out << std::endl;
    return 0;
}