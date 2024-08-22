#include <llama_torch.hpp>
#include <tuple>
using namespace llama_torch;

// Function to precompute frequency cis
torch::Tensor precompute_freqs_cis(int dim, int end, float theta = 10000.0) {
    auto freqs = 1.0 / torch::pow(theta, torch::arange(0, dim, 2, torch::kFloat32) / dim);
    auto t = torch::arange(end, torch::TensorOptions().dtype(torch::kFloat32));
    freqs = torch::ger(t, freqs);  // Equivalent to torch.outer in PyTorch
    auto freqs_cis = torch::polar(torch::ones_like(freqs), freqs);
    return freqs_cis;
}

// Function to reshape for broadcast
torch::Tensor reshape_for_broadcast(const torch::Tensor& freqs_cis, const torch::Tensor& x) {
    int ndim = x.dim();
    std::cout << x.sizes() << std::endl;
    std::cout << freqs_cis.sizes() << std::endl;
    assert(0 <= 1 && 1 < ndim);
    assert(freqs_cis.sizes() == torch::IntArrayRef({x.size(1), x.size(-1)}));

    std::vector<int64_t> shape;
    for (int i = 0; i < ndim; ++i) {
        shape.push_back((i == 1 || i == ndim - 1) ? x.size(i) : 1);
    }
    return freqs_cis.view(shape);
}

// Function to apply rotary embedding
std::tuple<torch::Tensor, torch::Tensor> apply_rotary_emb(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs_cis
) {
    auto xq_ = torch::view_as_complex(xq.to(torch::kFloat32).reshape({xq.size(0), xq.size(1), xq.size(2), -1, 2}));
    auto xk_ = torch::view_as_complex(xk.to(torch::kFloat32).reshape({xk.size(0), xk.size(1), xk.size(2), -1, 2}));
    auto freqs_cis_reshaped = reshape_for_broadcast(freqs_cis, xq_);
    auto xq_out = torch::view_as_real(xq_ * freqs_cis_reshaped).flatten(3);
    auto xk_out = torch::view_as_real(xk_ * freqs_cis_reshaped).flatten(3);
    return std::make_tuple(xq_out.to(xq.options()), xk_out.to(xk.options()));
}

int main() {
    torch::manual_seed(0);
	Tensor input = torch::randn({3, 20, 8, 768}, torch::kFloat32);
	auto rope_embed = RotaryEmbedding(768, 30, 500000.0);
	rope_embed->to(DATA_TYPE);
	Tensor rope_out = rope_embed->forward(input);

    Tensor freqs_cis = precompute_freqs_cis(768, 30, 500000.0);
    Tensor ground_truth, _;
    std::tie(ground_truth, _) = apply_rotary_emb(input, input, freqs_cis.index({torch::indexing::Slice(0, 20)}));
    ground_truth = ground_truth.reshape({3, 20, 8, 768});
	std::cout << "input:" << input.sizes() << std::endl;

    std::cout << "ground_truth:" << ground_truth.sizes()  << std::endl;
	std::cout << "ground_truth:" << ground_truth.dtype()  << std::endl;
	std::cout << "rope_out:" << rope_out.sizes()  << std::endl;
	std::cout << "rope_out:" << rope_out.dtype()  << std::endl;

    std::cout << "max diff:" << (rope_out - ground_truth).abs().max().item<float>() << std::endl;
    std::cout << "mean diff:" << (rope_out - ground_truth).abs().mean().item<float>() << std::endl;
    std::cout << "num diff:" << (rope_out - ground_truth).abs().gt(1e-5).sum().item<float>() << std::endl;

    return (rope_out - ground_truth).abs().gt(1e-5).sum().item<float>() > 0;
}