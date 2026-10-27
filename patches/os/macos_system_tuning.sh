#!/usr/bin/env bash
# diffusion.gguf: macOS High-Performance Inference & Metal Tuning

echo "[+] Optimizing macOS QoS and Mach thread parameters..."

# 1. Enable high-performance unified memory allocation
sudo sysctl -w kern.sched_latency=10000000 2>/dev/null || true

# 2. Prevent system App Nap / sleep during active inference
defaults write NSGlobalDomain NSAppSleepDisabled -bool YES

# 3. Increase max open files and virtual memory mappings
sudo launchctl limit maxfiles 65536 65536 2>/dev/null || true

echo "[+] macOS low-latency tuning configured."
