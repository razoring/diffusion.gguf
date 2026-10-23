#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace diffusion {

struct GgjtHparams {
    int32_t n_vocab = 0;
    int32_t n_embd = 0;
    int32_t n_mult = 0;
    int32_t n_head = 0;
    int32_t n_layer = 0;
    int32_t n_rot = 0;
    int32_t ftype = 0;
};

struct GgjtVocabItem {
    std::string text;
    float score = 0.0f;
};

struct GgjtTensor {
    std::string name;
    int32_t n_dims = 0;
    int32_t ggml_type = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    size_t data_offset = 0;
    size_t data_size = 0;
};

class GgjtReader {
public:
    GgjtReader();
    ~GgjtReader();

    bool open(const std::string& path);
    void close();

    uint32_t version() const { return version_; }
    const GgjtHparams& hparams() const { return hparams_; }
    const std::vector<GgjtVocabItem>& vocab() const { return vocab_; }
    const std::vector<GgjtTensor>& tensors() const { return tensors_; }
    const uint8_t* tensor_data(const GgjtTensor& t) const;

private:
    uint32_t version_ = 0;
    GgjtHparams hparams_;
    std::vector<GgjtVocabItem> vocab_;
    std::vector<GgjtTensor> tensors_;

    uint8_t* mapped_data_ = nullptr;
    size_t file_size_ = 0;

#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* map_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

} // namespace diffusion
