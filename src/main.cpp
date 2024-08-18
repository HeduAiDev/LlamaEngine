#include<torch/torch.h>
#include<torch/script.h>
#include<iostream>

using namespace torch;
class Attention: public torch::nn::Module {
public:
	Attention(int d_model, int nhead, int ffd, int dropout = 0.1):
	 wq(d_model, nhead * d_model), wk(d_model, nhead * d_model), wv(d_model, nhead * d_model), wo(nhead * d_model, d_model) {
		register_module("wq", wq);
		register_module("wk", wk);
		register_module("wv", wv);
		register_module("wo", wo);

	}
	Tensor forward(Tensor &x) {
		// x: [batch_size, seq_len, d_model]
		// q: [batch_size, seq_len, nhead * d_model]
		// k: [batch_size, seq_len, nhead * d_model]
		// v: [batch_size, seq_len, nhead * d_model]
		Tensor q = wq(x);
		Tensor k = wk(x);
		Tensor v = wv(x);
		Tensor score = torch::matmul(q, k.transpose(1, 2)) / sqrt(q.size(2));
		score = torch::softmax(score, 2);
		score = torch::matmul(score, v);
		return wo(score);
	}
	nn::Linear wq, wk, wv, wo;
};
 
 
int main() {
 
	Attention* attn = new Attention(512, 8, 2048);
	torch::Tensor x = torch::randn({1, 20, 512});
	std::cout << "wq(x)" << attn -> wq(x).sizes() << std::endl;
	Tensor output = attn -> forward(x);
	std::cout << *attn << std::endl;
	std::cout << output.sizes() << std::endl;
	// std::cout << output << std::endl;
	std::cout << "support cuda:" << torch::cuda::is_available() << std::endl;

	std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
	std::cout << "torch::cuda::cudnn_is_available():" << torch::cuda::cudnn_is_available() << std::endl;
	std::cout << "torch::cuda::device_count():" << torch::cuda::device_count() << std::endl;
 
	torch::Device device(torch::kCUDA);
	torch::Tensor tensor1 = torch::eye(3); // (A) tensor-cpu
	torch::Tensor tensor2 = torch::eye(3, device); // (B) tensor-cuda
	std::cout << tensor1 << std::endl;
	std::cout << tensor2 << std::endl;
}