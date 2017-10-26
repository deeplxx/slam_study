#include <iostream>
#include <vector>
#include <ctime>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

#include <Eigen/Core>

#include "pose_estimation_2d2d.cpp"

void pnpTest(const char* path1, cv::Mat& K,
             vector<cv::DMatch>& matches,
             vector<cv::KeyPoint>& kp_1, vector<cv::KeyPoint>& kp_2,
             cv::Mat& R, cv::Mat& t, vector<cv::Point3d>& pts_3d, vector<cv::Point2d>& pts_2d)
{
    cv::Mat img = cv::imread(path1, CV_LOAD_IMAGE_UNCHANGED);  // img是深度图，每个元素代表深度

    for (cv::DMatch m: matches) {
        ushort d = img.ptr<ushort>(int(kp_1[m.queryIdx].pt.y))[int(kp_1[m.queryIdx].pt.x)];
        if (d == 0)
            continue;
        double dd = d / 1000.0;
        cv::Point2d p1 = pixel2cam(kp_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3d(p1.x * dd, p1.y * dd, dd));  // 第一幅图的3D信息(世界坐标)
        pts_2d.push_back(kp_2[m.trainIdx].pt);  // 第二幅图的2D信息(像素坐标)
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    // PnP
    cv::Mat r;
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    cv::Rodrigues(r, R);  // 将旋转向量r转换为旋转矩阵R
    cout << "R = \n" << R << "\nr = \n" << t << endl;
}


/*目标是对之前估计的R和t进行优化！！！*/
void bundleAdjustment(const vector<cv::Point3d>& pts_3d,
                      const vector<cv::Point2d>& pts_2d,
                      const cv::Mat& K, cv::Mat& R, cv::Mat& t)
{
    // 构建图优化
    // **初始化一个优化器
    g2o::SparseOptimizer optimizer;  // 创建一个图优化

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;  // pose维度6，路标维度3
    unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCSparse<Block::PoseMatrixType>());
    unique_ptr<Block> solver_ptr(new Block(move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver_lb = new g2o::OptimizationAlgorithmLevenberg(move(solver_ptr));
    optimizer.setAlgorithm(solver_lb);

    // **图优化中添加节点
    // ****添加相机位姿
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
          R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
          R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId(0);
    pose->setEstimate(
                g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))));
    optimizer.addVertex(pose);
    // ****添加特征点(landmark)位置
    int idx = 1;  // 节点ID
    for (const cv::Point3d p: pts_3d){
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(idx++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    // **图优化中添加参数
    g2o::CameraParameters* cam_param = new g2o::CameraParameters(
                K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    cam_param->setId(0);
    optimizer.addParameter(cam_param);

    // **图优化中添加边
    idx = 1;
    for (const cv::Point2d p: pts_2d){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(idx);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(idx)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        idx++;
    }

    // 执行优化
    clock_t t1 = clock();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    double dur_t = (double)(1000 * (clock() - t1) / CLOCKS_PER_SEC);
    cout << "optimization cost time: " << dur_t << "ms" << endl;
    cout << "after optimization, T = \n" << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}
