#include <torch/torch.h>
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