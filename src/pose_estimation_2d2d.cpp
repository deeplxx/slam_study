#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

/*--对极约束计算旋转矩阵和平移向量*/
void poseEstimation_2d2d(vector<cv::KeyPoint> keypoints_1, vector<cv::KeyPoint> keypoints_2,
                         vector<cv::DMatch> matches, cv::Mat& R, cv::Mat& t)
{
    // 将匹配点转换为 vector<point2f>的形式
    vector<cv::Point2f> point1;
    vector<cv::Point2f> point2;
    for (int i = 0; i != (int)matches.size(); i++) {
        point1.push_back(keypoints_1[matches[i].queryIdx].pt);
        point2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // 计算基础矩阵
    cv::Mat fundamental_mat;
    fundamental_mat = cv::findFundamentalMat(point1, point2, CV_FM_8POINT);
//    cout << "fundamental_mat is \n" << fundamental_mat << endl;

    // 计算本质矩阵
    cv::Point2d principle_point(325.1, 249.7);  // 光心
    int focal_lenght = 521;  // 焦距
    cv::Mat essential_mat;
    essential_mat = cv::findEssentialMat(point1, point2, focal_lenght, principle_point);
//    cout << "essential_mat is \n" << essential_mat << endl;

    // 计算单应矩阵
    cv::Mat homography_mat;
    homography_mat = cv::findHomography(point1, point2, cv::RANSAC, 3);
//    cout << "homography_mat is \n" << homography_mat << endl;

    // 从本质矩阵中恢复旋转和平移信息
    cv::recoverPose(essential_mat, point1, point2, R, t, focal_lenght, principle_point);
//    cout << "R is \n" << R << endl;
//    cout << "t is \n" << t << endl;
}


/*像素坐标转相机归一化坐标*/
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
    return cv::Point2d(
               (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
               (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

/*三角测量*/
void triangulation(
    const vector<cv::KeyPoint>& kp_1,
    const vector<cv::KeyPoint>& kp_2,
    const vector<cv::DMatch>& matches,
    const cv::Mat& R, const cv::Mat& t, const cv::Mat& K,
    vector<cv::Point3d>& points)
{
    cv::Mat T1 = (cv::Mat_<double>(3, 4) <<
                  1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<double>(3, 4) <<
                  R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                  R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                  R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    vector<cv::Point2d> ptcam_1, ptcam_2; // 相机坐标
    for (cv::DMatch m: matches) {
        ptcam_1.push_back(pixel2cam(kp_1[m.queryIdx].pt, K));
        ptcam_2.push_back(pixel2cam(kp_2[m.trainIdx].pt, K));
    }

    // 三角测量
    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, ptcam_1, ptcam_2, pts_4d);

    // 转换为非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++) {
        cv::Mat x = pts_4d.col(i);
//        cout << "pts_4d: " << x << endl;
        x /= x.at<double>(3, 0);
//        cout << "pts_: " << x << endl;
        cv::Point3d p(x.at<double>(0, 0), x.at<double>(1, 0), x.at<double>(2, 0));
        points.push_back(p);
    }
}
