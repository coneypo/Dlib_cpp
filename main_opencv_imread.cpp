#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using  namespace cv;

int main() {

    Mat img_cv = imread("/home/hddl/code/cpp/dlib_cpp/faces_1.jpg", 0);
    if (img_cv.empty())
    {
        std::cout << "cannot read via cv";
        return -1;
    }

    namedWindow("display win");
    imshow("display win", img_cv);
    waitKey(0);

    return 0;
}
