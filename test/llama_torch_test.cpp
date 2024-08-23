#include "llama_torch.hpp"
#include <torch/torch.h>
#include <iostream>
using namespace llama_torch;



int main() {

    
    auto args = ModelArgs({
    /*dim*/        4096,
    // /*dim*/        128,
    /*n_heads*/    32, 
    /*n_kv_heads*/ 8, 
    /*n_layers*/   32, 
    /*rope_theta*/ 500000.0, 
    /*vocab_size*/ 128256, 
    // /*vocab_size*/ 1024, 
    /*hidden_dim*/ 14336, // hidden_dim = ((int((dim * 4 * 2) // 3 * ffn_dim_multiplier) + multiple_of - 1) // multiple_of) * multiple_of
    /*norm_eps*/   1e-05, 
    /*max_seq_len*/30,
    /*max_batch_size*/ 1
    });
    torch::manual_seed(0);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    auto transformer = Transformer(args);
    transformer -> to(torch::kFloat16);
    transformer -> to(device);
    transformer -> load_parameters("E:/Laboratory/LlamaEngine/model/Llama-3.1-8B-Instruct/consolidated.00.pth");
    std::cout << (transformer) << std::endl;
    Tensor tokens = torch::randint(args.vocab_size, {1, 20});
    tokens = tokens.to(device);
    auto out = transformer->forward(tokens, 0);
    std::cout << out.sizes() << std::endl;
    std::cout << out.argmax(-1) << std::endl;
    // std::cout << out << std::endl;
    return 0;
}