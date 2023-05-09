#include <benchmark/benchmark.h>
#include <Eigen/Core>
using EigenMapXt = Eigen::Map<Eigen::MatrixXf>;

#define NOFREE_EIGEN_MMAP(name, rows, cols)             \
	alignas(16) std::vector<float> mem##name(rows *cols); \
    EigenMapXt name(mem##name.data(), rows, cols);

static void BM_EigenAbat_addc(benchmark::State& state) {
	Eigen::Matrix<float, 12, 12> covII;
	Eigen::Matrix<float, 12, 12> PHI;
	Eigen::Matrix<float, 12, 12> Qd;
	srand((unsigned int) time(0));

	PHI.setRandom();
	Qd.setRandom();
	for(auto _ : state) {
		covII = PHI * covII * PHI.transpose() + Qd;
	}
}

// Register the function as a benchmark
BENCHMARK(BM_EigenAbat_addc);

// Define another benchmark
static void BM_EigenAbat_addc2(benchmark::State& state) {
	Eigen::Matrix<float, 12, 12> covII;
	Eigen::Matrix<float, 12, 12> PHI;
	Eigen::Matrix<float, 12, 12> Qd;
	srand((unsigned int) time(0));

	PHI.setRandom();
	Qd.setRandom();
	NOFREE_EIGEN_MMAP(PHIt, 12, 12);
	NOFREE_EIGEN_MMAP(covIIPHIt, 12, 12);
	NOFREE_EIGEN_MMAP(PHIcovIIPHIt, 12, 12);
	for(auto _ : state) {
	    PHIt = PHI.transpose();
        covIIPHIt = covII * PHIt;
        PHIcovIIPHIt = PHI * covIIPHIt;
        covII = PHIcovIIPHIt + Qd;
	}
}

BENCHMARK(BM_EigenAbat_addc2);

// Define another benchmark
static void BM_EigenAbat_addc3(benchmark::State& state) {
	Eigen::Matrix<float, 12, 12> covII;
	Eigen::Matrix<float, 12, 12> PHI;
	Eigen::Matrix<float, 12, 12> Qd;
	srand((unsigned int) time(0));

	PHI.setRandom();
	Qd.setRandom();
	NOFREE_EIGEN_MMAP(PHIt, 12, 12);
	NOFREE_EIGEN_MMAP(covIIPHIt, 12, 12);
	for(auto _ : state) {
	    Eigen::Matrix<float, 12, 12> PHIt = PHI.transpose();
         covIIPHIt = covII * PHIt;
        covII.noalias() = PHI * covIIPHIt + Qd;
	}
}
BENCHMARK(BM_EigenAbat_addc3);


BENCHMARK_MAIN();
