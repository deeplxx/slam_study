#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <opencv2/core.hpp>

/* 顶点模型 */

class CurveFittingVertex:
        public g2o::BaseVertex<3, Eigen::Vector3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl(){  // 重置
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double* update){  // 更新
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(istream& is){}
    virtual bool write(ostream& os) const{}
};

/* 边模型 */

class CurveFittingEdge:
        public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x): BaseUnaryEdge(), _x(x) {}

    void computeError(){  // 计算误差，取出边所链接的顶点来计算误差
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();  // 待优化参数的最新估计
        _error(0, 0) = _measurement - exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));  // 更新误差
    }

    virtual bool read(istream &is){}
    virtual bool write(ostream &os) const{}

private:
    double _x;  // 对应的y值为measurement
};

void g2oTest(){
    double a = 1.0, b = 2.0, c = 1.0;  // 真实参数值
    int N = 100;  // 数据点数量
    double w_sigma = 1.0;  // 噪声参数
    cv::RNG rng;  // 随机数产生器
    double abc[3] = {0, 0, 0};  // 初始化参数估计值
    vector<double> x_data, y_data;

    // generate data
    cout << "Generating data..." << endl;
    for (int i = 0; i != N; i++){
        double x = i / 100.0;
        double y = exp(a * x * x + b * x + c);
        x_data.push_back(x);
        y_data.push_back(y);
        cout << "x_data: " << x_data[i] << "    y_data: " << y_data[i] << endl;
    }

    // 构建图优化
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> Block;  // 矩阵块，优化变量维度3，误差项维度1
//    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 求解器：稠密增量方程
    unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverDense<Block::PoseMatrixType>());
//    Block* solver_ptr = new Block(move(linearSolver));
    unique_ptr<Block> solver_ptr(new Block(move(linearSolver)));

    g2o::OptimizationAlgorithmLevenberg* solver_levenberg = new g2o::OptimizationAlgorithmLevenberg(move(solver_ptr));
    g2o::OptimizationAlgorithmGaussNewton* solver_gaussnewton = new g2o::OptimizationAlgorithmGaussNewton(move(solver_ptr));
    g2o::OptimizationAlgorithmDogleg* solver_dogleg = new g2o::OptimizationAlgorithmDogleg(move(solver_ptr));

    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver_levenberg);
    optimizer.setVerbose(true);

    // 向图中添加节点,节点是待优化参数a, b, c
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);

    // 向图中添加边,边是观测数据的误差项
    for (int i = 0; i != N; i++){
        CurveFittingEdge* e = new CurveFittingEdge(x_data[i]);
        e->setId(i);
        e->setVertex(0, v);  // 设置链接的顶点
        e->setMeasurement(y_data[i]);
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge(e);
    }

    // 执行优化
    cout << "start optimization..." << endl;
    clock_t t1 = clock();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    double dur_t = 1000 * (clock() - t1) / (double)CLOCKS_PER_SEC;
    cout << "solver time cost = " << dur_t << "ms" << endl;

    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;
}

