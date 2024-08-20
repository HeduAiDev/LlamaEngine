#include <torch/torch.h>
#include<iostream>

using namespace torch;
#define DATA_TYPE (torch::kFloat16)


class TokenEmbedding: public torch::nn::Module {
public:
	TokenEmbedding(int vocab_size, int d_model):
	 embedding(vocab_size, d_model, DATA_TYPE) {
		register_module("embedding", embedding);
	}
	Tensor forward(Tensor &x) {
		return embedding(x);
	}
	nn::Embedding embedding;
};

class RMSNorm: public torch::nn::Module {
    public:
	RMSNorm(int _dim, float _eps = 1e-6): weight(register_parameter("weight", torch::ones(_dim, DATA_TYPE))), eps(_eps) {}

	Tensor forward(Tensor &x) {
		return x * torch::rsqrt(torch::mean(x * x, -1, true) + eps) * weight;
	}
	float eps;
	torch::Tensor weight;
};

class RotaryEmbedding: public torch::nn::Module {
	public:
	RotaryEmbedding(float _theta): theta(_theta) {}
	Tensor forward(Tensor &x) {
		int seq_len = x.size(-2);
		int dim 	= x.size(-1);
		Tensor pos = torch::arange(seq_len, DATA_TYPE);
		Tensor freqs = 1./torch::pow(theta, torch::arange(0, dim, 2, DATA_TYPE) / dim);
		// (seq_len, d_model // 2)
		freqs = torch::outer(pos, freqs);
		std::initializer_list<at::indexing::TensorIndex> p1m = {torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)};
		std::initializer_list<at::indexing::TensorIndex> p2m = {torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)};
		// [:, 0::2] -> cos, [:, 1::2] -> sin
		Tensor cos_mt = torch::cos(freqs.index(p1m));
		Tensor sin_mt = torch::sin(freqs.index(p2m));
		
		Tensor out = torch::empty_like(x);
		// [:, 0::2] = cos mθ * x1m - sin mθ * x2m
		out.index(p1m) = cos_mt * x.index(p1m) - sin_mt * x.index(p2m);
		// [:, 1::2] = sin mθ * x1m + cos mθ * x2m
		out.index(p2m) = sin_mt * x.index(p1m) + cos_mt * x.index(p2m);
		return out;
	}	
	float theta;
};
