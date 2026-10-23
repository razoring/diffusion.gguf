#include "ggjt_reader.h"
#include "gguf_emit.h"
#include <iostream>
#include <string>
#include <string_view>

static void print_convert_help() {
    std::cout << "Usage: diffusion-convert <input.ggjt> <output.gguf> [options]\n\n"
              << "Arguments:\n"
              << "  input.ggjt              Path to source GGJT v1, v2, or v3 file\n"
              << "  output.gguf             Destination GGUF file path\n\n"
              << "Options:\n"
              << "  --name NAME             Model name in GGUF metadata (default: filename)\n"
              << "  --ctx N                 Context length (default: 2048)\n"
              << "  --eps FLOAT             RMS norm epsilon (default: 1e-6, use 1e-5 for LLaMA-2)\n"
              << "  --gqa N                 Grouped Query Attention ratio (default: 1, 8 for 70B)\n"
              << "  -v, --version           Show version\n"
              << "  -h, --help              Show this help message\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_convert_help();
        return 1;
    }

    std::string input_path;
    std::string output_path;
    diffusion::ConvertOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-v" || arg == "--version") {
            std::cout << "diffusion-convert (diffusion.gguf) version 0.1.0\n";
            return 0;
        }
        if (arg == "-h" || arg == "--help") {
            print_convert_help();
            return 0;
        }

        if (arg == "--name" && i + 1 < argc) {
            options.name = argv[++i];
        } else if (arg == "--ctx" && i + 1 < argc) {
            options.ctx_len = std::stoi(argv[++i]);
        } else if (arg == "--eps" && i + 1 < argc) {
            options.eps = std::stof(argv[++i]);
        } else if (arg == "--gqa" && i + 1 < argc) {
            options.gqa = std::stoi(argv[++i]);
        } else if (!arg.empty() && arg[0] != '-') {
            if (input_path.empty()) {
                input_path = arg;
            } else if (output_path.empty()) {
                output_path = arg;
            }
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: Both input.ggjt and output.gguf paths are required.\n\n";
        print_convert_help();
        return 1;
    }

    std::cout << "Loading GGJT file: " << input_path << "\n";
    diffusion::GgjtReader reader;
    if (!reader.open(input_path)) {
        std::cerr << "Error: Failed to open or parse GGJT file: " << input_path << "\n";
        return 1;
    }

    std::cout << "Detected GGJT version " << reader.version()
              << " with " << reader.vocab().size() << " vocab entries and "
              << reader.tensors().size() << " tensors.\n";

    if (!diffusion::emit_gguf_file(reader, output_path, options)) {
        std::cerr << "Error: Failed to convert and emit GGUF file: " << output_path << "\n";
        return 1;
    }

    std::cout << "Successfully converted " << input_path << " to " << output_path << "\n";
    return 0;
}
