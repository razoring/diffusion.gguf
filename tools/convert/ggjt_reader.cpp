#include "ggjt_reader.h"
#include "ggml.h"
#include <cstring>
#include <iostream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace diffusion {

static constexpr uint32_t GGJT_MAGIC = 0x67676a74; // 'ggjt'

GgjtReader::GgjtReader() = default;

GgjtReader::~GgjtReader() {
    close();
}

void GgjtReader::close() {
    if (mapped_data_) {
#ifdef _WIN32
        UnmapViewOfFile(mapped_data_);
        mapped_data_ = nullptr;
        if (map_handle_) {
            CloseHandle(map_handle_);
            map_handle_ = nullptr;
        }
        if (file_handle_) {
            CloseHandle(file_handle_);
            file_handle_ = nullptr;
        }
#else
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
    }
    file_size_ = 0;
    tensors_.clear();
    vocab_.clear();
}

const uint8_t* GgjtReader::tensor_data(const GgjtTensor& t) const {
    if (!mapped_data_ || t.data_offset + t.data_size > file_size_) {
        return nullptr;
    }
    return mapped_data_ + t.data_offset;
}

static size_t compute_ggjt_data_size(uint32_t version, int32_t ggml_type, const int64_t* ne, int32_t n_dims) {
    int64_t n_elements = 1;
    for (int i = 0; i < n_dims; ++i) {
        n_elements *= ne[i];
    }

    if (version == 3) {
        size_t type_size = ggml_type_size(static_cast<enum ggml_type>(ggml_type));
        int64_t blck_size = ggml_blck_size(static_cast<enum ggml_type>(ggml_type));
        if (blck_size == 0) return 0;
        return (n_elements * type_size) / blck_size;
    }

    // Version 1 or 2
    if (ggml_type == GGML_TYPE_F32) {
        return n_elements * 4;
    } else if (ggml_type == GGML_TYPE_F16) {
        return n_elements * 2;
    } else if (ggml_type == GGML_TYPE_Q4_0) {
        // v1/v2: 20 bytes per 32 elements (float32 scale + 16 bytes nibbles)
        return (n_elements / 32) * 20;
    } else if (ggml_type == GGML_TYPE_Q4_1) {
        // v1/v2: 24 bytes per 32 elements (2x float32 + 16 bytes nibbles)
        return (n_elements / 32) * 24;
    } else if (ggml_type == GGML_TYPE_Q8_0) {
        // v1/v2: 36 bytes per 32 elements (float32 scale + 32 bytes)
        return (n_elements / 32) * 36;
    }

    // Fallback to ggml
    size_t type_size = ggml_type_size(static_cast<enum ggml_type>(ggml_type));
    int64_t blck_size = ggml_blck_size(static_cast<enum ggml_type>(ggml_type));
    return (n_elements * type_size) / blck_size;
}

bool GgjtReader::open(const std::string& path) {
    close();

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) return false;
    file_handle_ = hFile;

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size)) return false;
    file_size_ = static_cast<size_t>(size.QuadPart);
    if (file_size_ < 36) return false;

    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) return false;
    map_handle_ = hMap;

    void* ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) return false;
    mapped_data_ = static_cast<uint8_t*>(ptr);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    fd_ = fd;

    struct stat st;
    if (fstat(fd, &st) != 0) return false;
    file_size_ = static_cast<size_t>(st.st_size);
    if (file_size_ < 36) return false;

    void* ptr = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) return false;
    mapped_data_ = static_cast<uint8_t*>(ptr);
#endif

    const uint8_t* p = mapped_data_;
    const uint8_t* end = mapped_data_ + file_size_;

    // 1. Magic
    uint32_t magic = 0;
    std::memcpy(&magic, p, 4);
    p += 4;
    if (magic != GGJT_MAGIC) {
        std::cerr << "Invalid GGJT magic: 0x" << std::hex << magic << std::dec << "\n";
        return false;
    }

    // 2. Version
    std::memcpy(&version_, p, 4);
    p += 4;
    if (version_ != 1 && version_ != 2 && version_ != 3) {
        std::cerr << "Unsupported GGJT version: " << version_ << "\n";
        return false;
    }

    // 3. Hparams
    if (p + 28 > end) return false;
    std::memcpy(&hparams_.n_vocab, p + 0, 4);
    std::memcpy(&hparams_.n_embd,  p + 4, 4);
    std::memcpy(&hparams_.n_mult,  p + 8, 4);
    std::memcpy(&hparams_.n_head,  p + 12, 4);
    std::memcpy(&hparams_.n_layer, p + 16, 4);
    std::memcpy(&hparams_.n_rot,   p + 20, 4);
    std::memcpy(&hparams_.ftype,   p + 24, 4);
    p += 28;

    // 4. Vocab
    vocab_.reserve(hparams_.n_vocab);
    for (int32_t i = 0; i < hparams_.n_vocab; ++i) {
        if (p + 4 > end) return false;
        uint32_t len = 0;
        std::memcpy(&len, p, 4);
        p += 4;

        if (p + len + 4 > end) return false;
        std::string str(reinterpret_cast<const char*>(p), len);
        p += len;

        float score = 0.0f;
        std::memcpy(&score, p, 4);
        p += 4;

        vocab_.push_back(GgjtVocabItem{std::move(str), score});
    }

    // 5. Tensors
    while (p < end) {
        if (p + 12 > end) break; // Trailing alignment padding or done

        int32_t n_dims = 0;
        int32_t name_len = 0;
        int32_t ggml_type = 0;
        std::memcpy(&n_dims, p + 0, 4);
        std::memcpy(&name_len, p + 4, 4);
        std::memcpy(&ggml_type, p + 8, 4);
        p += 12;

        if (n_dims <= 0 || n_dims > 4 || name_len <= 0 || name_len > 256) {
            break;
        }

        if (p + n_dims * 4 + name_len > end) return false;

        GgjtTensor t;
        t.n_dims = n_dims;
        t.ggml_type = ggml_type;
        for (int i = 0; i < n_dims; ++i) {
            int32_t d = 0;
            std::memcpy(&d, p, 4);
            p += 4;
            t.ne[i] = d;
        }
        for (int i = n_dims; i < 4; ++i) {
            t.ne[i] = 1;
        }

        t.name = std::string(reinterpret_cast<const char*>(p), name_len);
        p += name_len;

        // 32-byte alignment for tensor data
        size_t offset = static_cast<size_t>(p - mapped_data_);
        offset = (offset + 31) & ~31;
        p = mapped_data_ + offset;

        t.data_offset = offset;
        t.data_size = compute_ggjt_data_size(version_, ggml_type, t.ne, n_dims);

        if (t.data_offset + t.data_size > file_size_) {
            std::cerr << "Tensor data exceeds file boundary: " << t.name << "\n";
            return false;
        }

        p += t.data_size;
        tensors_.push_back(std::move(t));
    }

    return true;
}

} // namespace diffusion
