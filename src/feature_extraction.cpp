#include <iostream>
#include <vector>
#include <unistd.h>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include "../include/featureVO.h"

FeatureMatch* featureExtration(const char* path1, const char* path2){
    if (access(path1, F_OK) == -1 || access(path2, F_OK) == -1){
        throw runtime_error("import error: can't find source img");
    }

    cv::Mat img1 = cv::imread(path1);
    cv::Mat img2 = cv::imread(path2);

    // 初始化
    vector<cv::KeyPoint> key_point_1, key_point_2;
    cv::Mat descriptor_1, descriptor_2;
    // nfeatures, scaleFactor, nleves, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    // 检测角点位置(固定500个特征点)
    cout << "检测角点位置..." << endl;
    orb->detect(img1, key_point_1);
    orb->detect(img2, key_point_2);

    // 根据位置计算BRIEF描述子
    cout << "计算描述子..." << endl;
    orb->compute(img1, key_point_1, descriptor_1);
    orb->compute(img2, key_point_2, descriptor_2);

    cv::Mat outImg1;
    cv::drawKeypoints(img1, key_point_1, outImg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
//    cv::imshow("ORB特征点", outImg1);

    // 对两幅图的描述子进行匹配,使用汉明距离
    vector<cv::DMatch> matches;  // D是descriptor
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptor_1, descriptor_2, matches);

    // 匹配点对 筛选（找出所有匹配之间的最小和最大距离, 当匹配的距离大于最小距离的两倍时，则认为匹配有误）
    double min_dist = 1e4, max_dist = 0;
    for (int i = 0; i != descriptor_1.rows; i++){
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf("最大和最小距离分别是：%f, %f\n", max_dist, min_dist);

    vector<cv::DMatch> good_match;
    for (int i = 0; i != descriptor_1.rows; i++){
        if (matches[i].distance <= max(2 * min_dist, 30.0)){
            good_match.push_back(matches[i]);
        }
    }

    // 绘制
    cv::Mat img_match, img_goodmatch;
    cv::drawMatches(img1, key_point_1, img2, key_point_2, matches, img_match);
    cv::drawMatches(img1, key_point_1, img2, key_point_2, good_match, img_goodmatch);

    cv::imwrite("orb特征.jpg", outImg1);
    cv::imwrite("全部匹配.jpg", img_match);
    cv::imwrite("good匹配.jpg", img_goodmatch);

    FeatureMatch* fm = new FeatureMatch();
    fm->kp_1 = key_point_1;
    fm->kp_2 = key_point_2;
    fm->matches = good_match;

//    cv::imshow("matches", img_match);
//    cv::imshow("goodmatch", img_goodmatch);
//    cv::waitKey(0);

    return fm;
}
