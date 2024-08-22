#include <torch/torch.h>
#include<iostream>

using namespace torch;
#define DATA_TYPE (torch::kFloat32)

namespace llama_torch {
	struct ModelArgs {
		int dim;
		int n_heads;
		int n_kv_heads;
		int n_layers;
		float rope_theta;
		int vocab_size;
		int hidden_dim;
		double norm_eps;
		int max_seq_len;
		int max_batch_size;
	};
	class TokenEmbedding: public torch::nn::Module {
	public:
		TokenEmbedding(int vocab_size, int dim):
		embedding(vocab_size, dim) {
			register_module("embedding", embedding);
		}
		Tensor forward(Tensor x) {
			return embedding(x);
		}
		nn::Embedding embedding;
	};

	struct RMSNorm : torch::nn::Module {
		RMSNorm(int64_t dim, double eps = 1e-6)
			: eps(eps) {
			weight = register_parameter("weight", torch::ones({dim}));
		}

		torch::Tensor _norm(const torch::Tensor& x) {
			auto mean_square = x.pow(2).mean(-1, true);
			return x * torch::rsqrt(mean_square + eps);
		}

		torch::Tensor forward(const torch::Tensor& x) {
			auto output = _norm(x.to(torch::kFloat)).to(x.scalar_type());
			return output * weight;
		}

		torch::Tensor weight;
		double eps;
	};

	class RotaryEmbedding: public torch::nn::Module {
		public:
		RotaryEmbedding(int _dim, int _max_seq_len, float _theta): dim(_dim), max_seq_len(_max_seq_len), theta(_theta) {
			Tensor pos = torch::arange(max_seq_len, DATA_TYPE);
			Tensor freqs = 1./torch::pow(theta, torch::arange(0, dim, 2, DATA_TYPE) / dim);
			// (seq_len, d_model // 2)
			freqs = torch::outer(pos, freqs);
			// [..., 0::2] -> cos, [..., 1::2] -> sin
			Tensor cos_mt = torch::cos(freqs);
			Tensor sin_mt = torch::sin(freqs);
			cached_cos_mt = register_buffer("cached_cos_mt", cos_mt);
			cached_sin_mt = register_buffer("cached_sin_mt", sin_mt);
		}
		Tensor forward(Tensor x) {
			// x: (batch_size, seq_len, n_heads, head_dim)
			int64_t seq_len = x.size(1);
			// (seq_len, d_model // 2) -> (1, seq_len, 1, d_model // 2)
			Tensor cos_mt = this -> cached_cos_mt.index({torch::indexing::Slice(0, seq_len)}).unsqueeze(0).unsqueeze(2);
			Tensor sin_mt = this -> cached_sin_mt.index({torch::indexing::Slice(0, seq_len)}).unsqueeze(0).unsqueeze(2);
			std::initializer_list<at::indexing::TensorIndex> p1m = {"...", torch::indexing::Slice(0, torch::indexing::None, 2)};
			std::initializer_list<at::indexing::TensorIndex> p2m = {"...", torch::indexing::Slice(1, torch::indexing::None, 2)};
			Tensor out = torch::empty_like(x);
			// [..., 0::2] = cos mθ * x1m - sin mθ * x2m
			out.index(p1m) = cos_mt * x.index(p1m) - sin_mt * x.index(p2m);
			// [..., 1::2] = sin mθ * x1m + cos mθ * x2m
			out.index(p2m) = sin_mt * x.index(p1m) + cos_mt * x.index(p2m);
			return out;
		}	
		float theta;
		int dim;
		int max_seq_len;
		torch::Tensor cached_cos_mt;
		torch::Tensor cached_sin_mt;
	};


	class Attention: public torch::nn::Module {
	    public:
		Attention(int max_batch_size, int max_seq_len, int dim, int n_heads, int n_kv_heads):
			head_dim(dim / n_heads),
			n_heads(n_heads),
			n_kv_heads(n_kv_heads),
			n_rep(n_heads / n_kv_heads),
			wq(nn::LinearOptions(dim, n_heads * head_dim).bias(false)),
			wk(nn::LinearOptions(dim, n_kv_heads * head_dim).bias(false)),
			wv(nn::LinearOptions(dim, n_kv_heads * head_dim).bias(false)),
			wo(nn::LinearOptions(n_heads * head_dim, dim).bias(false)),
			cache_k(torch::zeros({max_batch_size, max_seq_len, n_kv_heads, head_dim})),
			cache_v(torch::zeros({max_batch_size, max_seq_len, n_kv_heads, head_dim}))
			{
				register_module("wq", wq);
				register_module("wk", wk);
				register_module("wv", wv);
				register_module("wo", wo);
				register_buffer("cache_k", cache_k);
				register_buffer("cache_v", cache_v);
			}
		Tensor forward(Tensor x, int start_pos, RotaryEmbedding &rotary_emb, Tensor &mask) {
			// x: (batch_size, seq_len, d_model)
			int64_t batch_size = x.size(0);
			int64_t seq_len = x.size(1);
			Tensor xq = this -> wq(x).reshape({batch_size, seq_len, this -> n_heads, this -> head_dim});
			Tensor xk = this -> wk(x).reshape({batch_size, seq_len, this -> n_kv_heads, this -> head_dim});
			Tensor xv = this -> wv(x).reshape({batch_size, seq_len, this -> n_kv_heads, this -> head_dim});

			// rotary_embedding
			xq = rotary_emb.forward(xq);
			xk = rotary_emb.forward(xk);

			this -> cache_k = this -> cache_k.to(xq);
			this -> cache_v = this -> cache_v.to(xq);
			// [:batch_size, start_pos: start_pos + seq_len]
			this -> cache_k.index({torch::indexing::Slice(0, batch_size), torch::indexing::Slice(start_pos, start_pos + seq_len), torch::indexing::Slice(), torch::indexing::Slice()}) = xk;
			this -> cache_v.index({torch::indexing::Slice(0, batch_size), torch::indexing::Slice(start_pos, start_pos + seq_len), torch::indexing::Slice(), torch::indexing::Slice()}) = xv;
			// [:batch_size, :start_pos + seq_len]
			Tensor keys = 	this -> cache_k.index({torch::indexing::Slice(0, batch_size), torch::indexing::Slice(0, start_pos + seq_len), torch::indexing::Slice(), torch::indexing::Slice()});
			Tensor values = this -> cache_v.index({torch::indexing::Slice(0, batch_size), torch::indexing::Slice(0, start_pos + seq_len), torch::indexing::Slice(), torch::indexing::Slice()});
			// repeat kv (batch_size, seq_len, n_kv_heads, head_dim)-> (batch_size, seq_len, n_heads, head_dim)
			keys = keys.unsqueeze(-2).expand({batch_size, seq_len, this -> n_kv_heads, this -> n_rep, this -> head_dim}).reshape_as(xq);
			values = values.unsqueeze(-2).expand({batch_size, seq_len, this -> n_kv_heads, this -> n_rep, this -> head_dim}).reshape_as(xq);
			
			// (batch_size, n_heads, seq_len, head_dim)
			xq = xq.transpose(1, 2);
			// (batch_size, n_heads, cache_len + seq_len, head_dim)
			keys = keys.transpose(1, 2);
			// (batch_size, n_heads, cache_len + seq_len, head_dim)
			values = values.transpose(1, 2);
			// scale dot product attention
			// (batch_size, n_heads, seq_len, cache_len + seq_len)
			Tensor scores = torch::matmul(xq, keys.transpose(-2, -1)) / sqrt(this -> head_dim);
			if (mask.defined()) {
				scores = scores + mask;
			}
			scores = torch::softmax(scores.to(torch::kFloat), -1).type_as(xq);
			// (batch_size, n_heads, seq_len, head_dim)
			Tensor output = torch::matmul(scores, values);
			output = output.transpose(1, 2).contiguous().reshape({batch_size, seq_len, this -> n_heads * this -> head_dim});
			return this -> wo(output);
		}
		int head_dim, n_heads, n_kv_heads, n_rep;
		nn::Linear wq, wk, wv, wo;
		Tensor cache_k, cache_v;
	};

	class FeedForward : public torch::nn::Module {
	public:
		FeedForward(int dim, int hidden_dim):
			w1(nn::LinearOptions(dim, hidden_dim).bias(false)),
			w2(nn::LinearOptions(hidden_dim, dim).bias(false)),
			w3(nn::LinearOptions(dim, hidden_dim).bias(false))
			{
				register_module("w1", w1);
				register_module("w2", w1);
				register_module("w3", w1);
			}
		Tensor forward(Tensor x) {
		   return this -> w2(torch::silu(this -> w1(x) * this -> w3(x)));
		}
		nn::Linear w1, w2, w3;
	};


	class TransformerBlock : public torch::nn::Module {
	    public:
		TransformerBlock(int layer_id, ModelArgs &args):
			layer_id(layer_id),
			attention(args.max_batch_size, args.max_seq_len, args.dim, args.n_heads, args.n_kv_heads),
			feed_forward(args.dim, args.hidden_dim),
			attention_norm(args.dim, args.norm_eps),
			ffn_norm(args.dim, args.norm_eps)
			{}
		Tensor forward(Tensor x, int start_pos, RotaryEmbedding &rotary_embedding, Tensor &mask) {
		   Tensor h = x + this -> attention.forward(this -> attention_norm.forward(x), start_pos, rotary_embedding, mask);
		   return h + this -> feed_forward.forward(this -> ffn_norm.forward(h));
		}
		Attention attention;
		FeedForward feed_forward;
		RMSNorm attention_norm, ffn_norm;
		int layer_id;
	};

	class Transformer : public torch::nn::Module {
	    public:
		Transformer(ModelArgs &args):
			token_embedding(args.vocab_size, args.dim),
			rotary_embedding(args.dim / args.n_heads, args.max_seq_len, args.rope_theta),
			norm(args.dim, args.norm_eps),
			layers(),
			lm_head(nn::LinearOptions(args.dim, args.vocab_size).bias(false))
			{
				for (int i = 0; i < args.n_layers; i++) {
					layers.push_back(TransformerBlock(i, args));
				}
				register_module("lm_head", lm_head);
		}
		Tensor forward(Tensor tokens, int start_pos) {
		    torch::NoGradGuard no_grad;
			int64_t seq_len = tokens.size(1);
			Tensor h = this -> token_embedding.forward(tokens);
			this -> rotary_embedding.to(h.device());

			Tensor mask(nullptr);
			if (seq_len > 1) {
				mask = torch::full({seq_len, seq_len}, -1e9, torch::device(tokens.device()));
				mask = torch::triu(mask, 1);
				// When performing key-value caching, we compute the attention scores
				// only for the new sequence. Thus, the matrix of scores is of size
				// (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
				// j > cache_len + i, since row i corresponds to token cache_len + i.
				mask = torch::hstack({torch::zeros({seq_len, start_pos}, torch::device(tokens.device())), mask}).type_as(h);
			}

			for (auto &layer : layers) {
				h = layer.forward(h, start_pos, rotary_embedding, mask);
			}
			h = this -> norm.forward(h);
			return this -> lm_head(h);


		}
		TokenEmbedding token_embedding;
		RotaryEmbedding rotary_embedding;
		std::vector<TransformerBlock> layers;
		RMSNorm norm;
		nn::Linear lm_head;
	};
}