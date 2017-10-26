#include <iostream>

#include <opencv2/core.hpp>
#include <ceres/ceres.h>
#include <ctime>

using namespace std;

/*定义cost模型*/
struct CurveFittingCost
{
    const double _x, _y;  // 标签数据

    CurveFittingCost(double x, double y): _x(x), _y(y){}

    // 计算参差，abc为三维模型参数（是一个顶层+底层指针）
    template <typename T>
    bool operator() (const T* const abc, T* residual) const{
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }
};


void ceresTest(){
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

    // define a problem
    ceres::Problem problem_1;
    for (int i = 0; i != N; i++){
        problem_1.AddResidualBlock(  // 问题中添加误差项
                    new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3>(new CurveFittingCost(x_data[i], y_data[i])),
                    nullptr,
                    abc);
    }

    // config solver
    ceres::Solver::Options options;  // 配置项
    options.linear_solver_type = ceres::DENSE_QR;  // 如何求解
    options.minimizer_progress_to_stdout = true;  // 输出到cout

    ceres::Solver::Summary summary;  // 优化信息

    // 优化！！！
    clock_t t1 = clock();
    ceres::Solve(options, &problem_1, &summary);
    double dur_t = 1000 * (clock() - t1) / (double)CLOCKS_PER_SEC;
    cout << "solve cost time: " << dur_t << "ms" << endl;

    cout << summary.BriefReport() << endl;
    cout << "estimated a, b, c = ";
    for (auto i: abc){
        cout << i << " ";
    }
    cout << endl;

}
