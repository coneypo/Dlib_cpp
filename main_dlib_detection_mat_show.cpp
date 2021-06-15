#include <iostream>
#include <fstream>
#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main() {

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::array2d<unsigned char> img;

    dlib::load_image(img, "/home/hddl/code/cpp/dlib_cpp/faces_1.jpg");
    pyramid_up(img);

    std::vector<dlib::rectangle> dets = detector(img);
    cout << "faces: " << dets.size() << endl;

    namedWindow("display win");

    cv::Mat img_Mat = dlib::toMat(img);

    cv::imshow("display win", img_Mat);
    waitKey(0);

    return 0;
}
