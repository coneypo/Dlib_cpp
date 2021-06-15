#include <iostream>
#include <fstream>
#include <vector>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/dnn/loss_abstract.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/string.h>
#include <dlib/clustering.h>


#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dlib;

//using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
//                                             alevel0<
//                                             alevel1<
//                                             alevel2<
//                                             alevel3<
//                                             alevel4<
//                                             max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
//        input_rgb_image_sized<150>
//>>>>>>>>>>>>;

int main() {
    dlib::shape_predictor sp;
    deserialize("/home/hddl/code/cpp/Dlib_cpp/data/data_dlib/shape_predictor_68_face_landmarks.dat") >> sp;
//    anet_type net;
//    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::array2d<unsigned char> img;

    dlib::load_image(img, "/home/hddl/code/cpp/Dlib_cpp/data/data_images/faces_1.jpg");
    pyramid_up(img);

    std::vector<dlib::rectangle> dets = detector(img);
    cout << "faces: " << dets.size() << endl;
    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img)) {
        auto shape = sp(img, face);

        auto fod_num_parts = shape.num_parts();
        std::cout << "xxx: " << fod_num_parts << "\n";  // 68

        auto fod_part = shape.part(0);
        std::cout << "xxx: " << fod_part << "\n";       //
    }

//    namedWindow("display win");
//
//    cv::Mat img_Mat = dlib::toMat(img);
//
//    cv::imshow("display win", img_Mat);
//    waitKey(0);

    return 0;
}
