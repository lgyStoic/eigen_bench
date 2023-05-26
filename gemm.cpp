#define EIGEN_GEMM_TO_COEFFBASED_THRESHOLD 1
#define EIGEN_NO_MALLOC
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#define TMatrixXt Eigen::MatrixXf
using EigenMapXt = Eigen::Map<Eigen::MatrixXf>;

#define NOFREE_EIGEN_MMAP(name, rows, cols)             \
	alignas(16) std::vector<float> mem##name(rows *cols); \
    EigenMapXt name(mem##name.data(), rows, cols);


static void BM_EigenGemm() {
	std::vector<float> adata(2 * 6);
	Eigen::Map<Eigen::Matrix<float, -1, -1, 0>, Eigen::Unaligned> a(adata.data(), 2, 6);
	//NOFREE_EIGEN_MMAP(a, 2, 6);
	std::vector<float> bdata(6 * 102);
	Eigen::Map<Eigen::Matrix<float, -1, -1, 0>, Eigen::Unaligned> b(bdata.data(), 6, 102);
	//NOFREE_EIGEN_MMAP(b, 6, 102);
	std::vector<float> abdata(2 * 102);
	Eigen::Map<Eigen::Matrix<float, -1, -1, 0>, Eigen::Unaligned> a_b(abdata.data(), 2, 102);
	//NOFREE_EIGEN_MMAP(a_b, 2, 102);
	a.setRandom();
	b.setRandom();
	a_b.noalias() = a * b;
	std::cout << a_b << std::endl;
}

// Register the function as a benchmark
//BENCHMARK(BM_EigenLuSolver);



//BENCHMARK_MAIN();

int main () {
	BM_EigenGemm();	
}

//int main()
//{
//    using Scalar = float;
//    using namespace Eigen;
//    std::vector<Scalar> aDat = {1, 2, 3, 4};
//    std::vector<Scalar> bDat = {1, 2, 3, 4};
//    std::vector<Scalar> cDat = {1, 2, 3, 4};
//    Map<Matrix<Scalar, -1, -1, RowMajor>, Unaligned> a(aDat.data(), 2, 2);
//    Map<Matrix<Scalar, -1, -1, RowMajor>, Unaligned> b(bDat.data(), 2, 2);
//    Map<Matrix<Scalar, -1, -1, RowMajor>, Unaligned> c(cDat.data(), 2, 2);
//
//    //Ok
//    c.noalias() += a * b;
//
//    //Assertion `false && "heap allocation is forbidden.....
//    c.noalias() += 2 * a * b;
//
//    return 0;
//}
