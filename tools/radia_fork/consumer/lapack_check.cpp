// Installed-SDK consumer via find_package(NGSolve) + target ngsolve only.
// Real and complex LapackInverse; oracle = scalar product A*inv(A) vs I computed
// here (no LAPACK in the check). Prints JSON lines.
#include <bla.hpp>
#include <cstdio>
#include <complex>
#include <cmath>
using namespace ngbla;
static_assert(sizeof(integer) == 8, "SDK must expose 64-bit LAPACK integers");
template <typename T> T entry(size_t i, size_t j, size_t n);
template <> double entry<double>(size_t i, size_t j, size_t n)
{ return (i == j) ? double(n) : std::sin(1.0 + i * 31 + j * 17); }
template <> Complex entry<Complex>(size_t i, size_t j, size_t n)
{ return (i == j) ? Complex(n, 0.5 * n) : Complex(std::sin(1.0 + i * 31 + j * 17), std::cos(2.0 + i * 7 + j * 13)); }
static bool nonfinite = false;
template <typename T> void run(const char *kind)
{
  for (size_t n : {7, 64, 300}) {
    Matrix<T> a(n, n), ainv(n, n);
    for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) a(i, j) = entry<T>(i, j, n);
    ainv = a;
    LapackInverse(ainv);
    double err = 0;
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < n; j++) {
        T s = 0;
        for (size_t k = 0; k < n; k++) s += a(i, k) * ainv(k, j);
        double d = std::abs(s - T(i == j ? 1.0 : 0.0));
        if (!std::isfinite(d)) { nonfinite = true; d = 1e300; }  // never let NaN vanish in max
        err = std::max(err, d);
      }
    std::printf("{\"kind\": \"%s\", \"n\": %zu, \"max_abs_AinvA_minus_I\": %.3e}\n", kind, n, err);
  }
}
int main() { run<double>("real"); run<Complex>("complex"); return nonfinite ? 1 : 0; }
