#include "render_final.h"
#include <fstream>
#include <iostream>

namespace diffusion {

RenderFinal::RenderFinal(const CliArgs& args) : args_(args) {}

bool RenderFinal::on_step(const StepEvent& /*event*/) {
    // Non-streaming final mode outputs nothing during execution
    return true;
}

void RenderFinal::finish(const GenResult& result) {
    std::cout << result.text << "\n";
    std::cout.flush();

    if (!args_.tokens_out_file.empty()) {
        std::ofstream f(args_.tokens_out_file);
        if (f) {
            for (size_t i = 0; i < result.tokens.size(); ++i) {
                if (i > 0) f << ",";
                f << result.tokens[i];
            }
            f << "\n";
        }
    }

    if (args_.show_stats) {
        std::cerr << "{\n"
                  << "  \"tokens\": " << result.tokens.size() << ",\n"
                  << "  \"iterations\": " << result.iterations << ",\n"
                  << "  \"tokens_per_iteration\": " << result.tok_per_iter << ",\n"
                  << "  \"tokens_per_second\": " << result.tok_per_s << ",\n"
                  << "  \"total_time_ms\": " << result.total_time_ms << "\n"
                  << "}\n";
    }
}

} // namespace diffusion
