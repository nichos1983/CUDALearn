#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "verifier.cuh"

// Enable it to debug max error logic and calcuating precision
// #define DEBUG_PRINT_MAX_ERROR

VerifyResult SGEMMVerifier::verify(const float* h_test, const float* h_ref,
                                   int M, int N, float rtol, float atol) {
    VerifyResult result;
    result.max_abs_error = 0.0f;
    result.max_rel_error = 0.0f;
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
        float rel_error = abs_error / (std::abs(ref_val) + constants::EPSILON);

        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);

#ifdef DEBUG_PRINT_MAX_ERROR
        if (abs_error > constants::EPSILON &&
            result.max_abs_error == abs_error) {
            printf("\nabs_error reached max. i: %d, test_val: %f ref_val: %f\n",
                   i, test_val, ref_val);
            printf("abs_error: %f, rel_error: %f\n", abs_error, rel_error);
        }

        if (rel_error > constants::EPSILON &&
            result.max_rel_error == rel_error) {
            printf("\nrel_error reached max. i: %d, test_val: %f ref_val: %f\n",
                   i, test_val, ref_val);
            printf("abs_error: %f, rel_error: %f\n", abs_error, rel_error);
        }
#endif

        float tolerance = atol + rtol * std::abs(ref_val);
        if (abs_error > tolerance) {
            printf("\nVerify fail. i: %d, test_val: %f ref_val: %f\n", i,
                   test_val, ref_val);
            printf("abs_error: %f, rel_error: %f tolerance: %f\n", abs_error,
                   rel_error, tolerance);
            result.error_count++;
        }
    }

    return result;
}