#include <algorithm>
#include <cmath>
#include <limits>

#include "verifier.cuh"

VerifyResult SGEMMVerifier::verify(const float* h_test, const float* h_ref,
                                   int M, int N, float rtol, float atol) {
    VerifyResult result;
    result.error_count = 0;
    result.total_count = M * N;

    for (int i = 0; i < M * N; i++) {
        float test_val = h_test[i];
        float ref_val = h_ref[i];

        if (std::isnan(test_val) || std::isnan(ref_val) ||
            std::isinf(test_val) || std::isinf(ref_val)) {
            result.error_count++;
            result.max_abs_error = std::numeric_limits<float>::infinity();
            result.max_rel_error = std::numeric_limits<float>::infinity();
            continue;
        }

        float abs_error = std::abs(test_val - ref_val);
        float rel_error = abs_error / (std::abs(ref_val) +
                                       std::numeric_limits<float>::epsilon());

        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);

        float tolerance = atol + rtol * std::abs(ref_val);
        if (abs_error > tolerance)
            result.error_count++;
    }

    return result;
}