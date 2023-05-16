#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>

#define TMatrixXt Eigen::MatrixXf
using EigenMapXt = Eigen::Map<Eigen::MatrixXf>;

#define NOFREE_EIGEN_MMAP(name, rows, cols)             \
	alignas(16) std::vector<float> mem##name(rows *cols); \
    EigenMapXt name(mem##name.data(), rows, cols);

static void BM_EigenLltSolver(benchmark::State& state) {
	NOFREE_EIGEN_MMAP(S, 90, 90);
	NOFREE_EIGEN_MMAP(PHt, 182, 90);
	std::ifstream fin("test.txt", std::ios::in | std::ios::binary );
	fin.read((char*)PHt.data(), 90 * 182 * 4);

	NOFREE_EIGEN_MMAP(Kt, S.cols(), PHt.rows());
	S.setRandom();

	for(auto _ : state) {
		Eigen::LLT<Eigen::Ref<TMatrixXt>> llt(S);
		Kt.noalias() = llt.solve(PHt.transpose());
	}
}

BENCHMARK(BM_EigenLltSolver);

static void BM_EigenLdltSolver(benchmark::State& state) {
	NOFREE_EIGEN_MMAP(S, 90, 90);
	NOFREE_EIGEN_MMAP(PHt, 182, 90);
	NOFREE_EIGEN_MMAP(Kt, S.cols(), PHt.rows());
	std::ifstream fin("test.txt", std::ios::in | std::ios::binary );
	fin.read((char*)PHt.data(), 90 * 182 * 4);
	S.setRandom();

	for(auto _ : state) {
		Eigen::LDLT<Eigen::Ref<TMatrixXt>> ldlt(S);
		Kt.noalias() = ldlt.solve(PHt.transpose());
	}
}

// Register the function as a benchmark
BENCHMARK(BM_EigenLdltSolver);

static void BM_EigenLuSolver(benchmark::State& state) {
	NOFREE_EIGEN_MMAP(S, 90, 90);
	NOFREE_EIGEN_MMAP(PHt, 182, 90);
	NOFREE_EIGEN_MMAP(Kt, S.cols(), PHt.rows());
	std::ifstream fin("test.txt", std::ios::in | std::ios::binary );
	fin.read((char*)PHt.data(), 90 * 182 * 4);
	S.setRandom();

	for(auto _ : state) {
		Eigen::FullPivLU<Eigen::Ref<TMatrixXt>> lu(S);
		Kt.noalias() = lu.solve(PHt.transpose());
	}
}

// Register the function as a benchmark
BENCHMARK(BM_EigenLuSolver);



BENCHMARK_MAIN();

