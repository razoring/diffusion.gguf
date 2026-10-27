#pragma once

namespace diffusion {
namespace os {

// Applies OS-specific low-latency scheduling, timer resolution, and core affinity
bool tune_process_for_inference();

// Restores OS defaults upon process exit
void restore_process_tuning();

} // namespace os
} // namespace diffusion
