#include <llama_torch.hpp>

int main() {
    Attention* attn = new Attention(512, 8, 2048);
	torch::Tensor x = torch::randn({1, 20, 512});
	std::cout << "wq(x)" << attn -> wq(x).sizes() << std::endl;
	Tensor output = attn -> forward(x);
	std::cout << *attn << std::endl;
	std::cout << output.sizes() << std::endl;
    return 0;
}