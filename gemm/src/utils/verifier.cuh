#pragma once

#include <cuda_runtime.h>

#include "constants.cuh"

struct VerifyResult {
    float max_abs_error = 0.0f;
    // We use EPSILON as part of denominator when calculating max_rel_error to
    // avoid division by zero. So the value of max_rel_error in occasional cases
    // might be large enough to scare you. But max_rel_error is just for logical
    // integrity. We only use max_abs_error to check if the test is passed.
    float max_rel_error = 0.0f;
    int error_count = 0;
    int total_count = 0;

    [[nodiscard]] bool is_passed() const {
        return error_count == 0;
    };
};

class SGEMMVerifier {
public:
    SGEMMVerifier() = delete;

    static VerifyResult verify(const float* h_test, const float* h_ref, int M,
                               int N, float rtol = constants::RTOL,
                               float atol = constants::ATOL);
};
