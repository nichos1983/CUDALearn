#pragma once

#include <cstdint>

namespace constants {

inline constexpr uint32_t RANDOM_SEED1 = 42;
inline constexpr uint32_t RANDOM_SEED2 = 1314;
inline constexpr float ATOL = 1e-3f;
inline constexpr float RTOL = 1e-4f;
inline constexpr float SGEMM_ALPHA = 1.0f;
inline constexpr float SGEMM_BETA = 0.0f;

}  // namespace constants
