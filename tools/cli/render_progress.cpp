#include "render_progress.h"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace diffusion {

RenderProgress::RenderProgress(const CliArgs& args)
    : args_(args), last_render_time_(std::chrono::steady_clock::now()) {}

bool RenderProgress::on_step(const StepEvent& ev) {
    int n_predict = args_.params.n_predict > 0 ? args_.params.n_predict : 128;
    float pct = ev.done ? 100.0f : ((n_predict > 0) ? (100.0f * ev.committed_upto / n_predict) : 0.0f);
    if (pct > 100.0f) pct = 100.0f;

    int bar_width = 25;
    int filled = static_cast<int>(pct / 100.0f * bar_width);
    if (filled > bar_width) filled = bar_width;

    std::cerr << "\r[";
    for (int i = 0; i < filled; ++i) std::cerr << "=";
    if (filled < bar_width) std::cerr << ">";
    for (int i = filled + 1; i < bar_width; ++i) std::cerr << " ";
    std::cerr << "] " << std::fixed << std::setprecision(1) << pct << "%"
              << " (" << ev.committed_upto << "/" << n_predict << " tok)"
              << " · it: " << ev.step
              << " · " << std::fixed << std::setprecision(2) << ev.tok_per_iter << " tok/it"
              << " · " << std::fixed << std::setprecision(1) << ev.tok_per_s << " tok/s";
    std::cerr.flush();

    if (ev.done) {
        std::cerr << "\n";
    }
    return true;
}

void RenderProgress::finish(const GenResult& result) {
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
