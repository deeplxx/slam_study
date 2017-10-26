#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

string getName(int idx, int frame_count);

void video2pic(const char* video_path) {
	string prefix("..");
	cv::VideoCapture capture(prefix + video_path);
	if (!capture.isOpened()) {
		throw runtime_error("error path!");
	}

	double rate = capture.get(cv::CAP_PROP_FPS);
	int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
	cout << "the frame rate is " << rate;
	cout << " and thr frame count is " << frame_count << endl;

	cout << "starting write frame..." << endl;
	cv::Mat frame;
	int idx = 0;
	ofstream fou("../rgb_data/rgb.txt");
	while (capture.read(frame)) {
		string name = "rgb/" + getName(idx, frame_count);
		cv::imwrite("../rgb_data/" + name, frame);
		cout << "already write " << idx << endl;
		fou << "123 " + name << endl;
		idx++;
	}
}

string getName(int idx, int frame_count) {
	// 帧数转换成字符串并获得长度

	char count[10000];
	sprintf(count, "%d", frame_count);
	string str_frame_count(count);
	int str_length = str_frame_count.size();

	// 生成name，前缀补0
	char stridx[10000];
	sprintf(stridx, "%d", idx);
	string name(stridx);
	while (name.size() < str_length) {
		name = "0" + name;
	}
	name = name + ".png";
	return name;
}

