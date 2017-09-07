//
//
//
//
//

#include <opencv2/opencv.hpp>
#include <fstream>
#include <thread>
#include <mutex>
#include <chrono>

using namespace cv;
using namespace std;

bool gRunning = true;
bool gSaveNext = false;
std::mutex gMutex;

void mouseCallback(int event, int x, int y, int flags, void* userdata);

bool imagesFromCamera(int _argc, char **_argv,  std::vector<std::vector<cv::Point2f>> &_leftPoints, std::vector<std::vector<cv::Point2f>> &_rightPoints);
bool imagesFromDataset(int _argc, char **_argv, std::vector<std::vector<Point2f> > &_leftPoints, std::vector<std::vector<Point2f> > &_rightPoints);

Size boardSize;
int width, height;

int main(int _argc, char ** _argv){
    std::vector<std::vector<cv::Point2f>> pointsLeft2D, pointsRight2D;
    //imagesFromCamera(_argc, _argv, pointsLeft2D, pointsRight2D);

    imagesFromDataset(_argc, _argv, pointsLeft2D, pointsRight2D);

    //// Calibration!
    // Calibrate single cameras.
    float squareSize = atof(_argv[6]);
    std::cout << "Computing parameters of individual cameras." << std::endl;
    Mat matrixLeft = Mat::eye(3, 3, CV_64F), matrixRight = Mat::eye(3, 3, CV_64F);
    Mat distCoefLeft = Mat::zeros(8, 1, CV_64F), distCoefRight= Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f>> pointsLeft(1), pointsRight(1);
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            pointsLeft[0].push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
            pointsRight[0].push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
        }
    }
    pointsLeft.resize(pointsLeft2D.size(),pointsLeft[0]);
    pointsRight.resize(pointsRight2D.size(),pointsRight[0]);

    std::vector<cv::Mat> rotVectors, transVectors;

    double rmsLeft = calibrateCamera(pointsLeft, pointsLeft2D, Size(width/2, height), matrixLeft, distCoefLeft,rotVectors, transVectors, CALIB_FIX_K4|CALIB_FIX_K5);
    double rmsRight = calibrateCamera(pointsRight, pointsRight2D, Size(width/2, height), matrixRight, distCoefRight, rotVectors, transVectors, CALIB_FIX_K4|CALIB_FIX_K5);

    std::cout << "Rms left calibration: " << rmsLeft << ". Rms right calibration: " << rmsRight << std::endl;

    // Calibrate stereo.
    std::cout << "Computing parameters of stereo system." << std::endl;
    vector<vector<Point3f>> pointsBoard(1);
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            pointsBoard[0].push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
        }
    }
    pointsBoard.resize(pointsLeft2D.size(),pointsBoard[0]);

    Mat rot, trans, essential, fundamental;
    stereoCalibrate(pointsBoard, pointsLeft2D, pointsRight2D, matrixLeft, distCoefLeft, matrixRight, distCoefRight,Size(width/2, height), rot, trans, essential, fundamental);

    Mat rectificationLeft, rectificationRight, projectionLeft, projectionRight, disparityToDepth;
    cv::Rect roiLeft, roiRight;
    stereoRectify(matrixLeft, distCoefLeft, matrixRight, distCoefRight, Size(width/2, height), rot, trans,
                        rectificationLeft, rectificationRight, projectionLeft, projectionRight, disparityToDepth,
                  CALIB_ZERO_DISPARITY, -1, Size(), &roiLeft, &roiRight);


    std::cout << "Stereo Camera calibrated! Saving files." << std::endl;
    FileStorage fs(_argv[3], FileStorage::WRITE);

    fs << "MatrixLeft" << matrixLeft;
    fs << "DistCoeffsLeft" << distCoefLeft;
    fs << "MatrixRight" << matrixRight;
    fs << "DistCoeffsRight" << distCoefRight;
    fs << "Rotation" << rot;
    fs << "Translation" << trans;
    fs << "Essential" << essential;
    fs << "Fundamental" << fundamental;
    fs << "RectificationLeft" << rectificationLeft;
    fs << "RectificationRight" << rectificationRight;
    fs << "ProjectionLeft" << projectionLeft;
    fs << "ProjectionRight" << projectionRight;
    fs << "DisparityToDepth" << disparityToDepth;
    fs << "RoiLeft" << roiLeft;
    fs << "RoiRight" << roiRight;

    std::cout << "Saved files." << std::endl;

    return 1;
}
//---------------------------------------------------------------------------------------------------------------------
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN){
        gSaveNext = true;
    } else if (event == EVENT_RBUTTONDOWN){
        std::cout << "Received stop signal" << std::endl;
        gRunning = false;
    }
}


bool imagesFromCamera(int _argc, char **_argv, std::vector<std::vector<cv::Point2f>> &_leftPoints, std::vector<std::vector<cv::Point2f>> &_rightPoints){
    if (_argc != 7) {
        std::cout << "Bad input arguments. Usage: \n ./featureModelCreator [cameraIdx] [frameWidth] [calibFile] [boardWidth] [boardHeight] [squareSize]" << std::endl;
        return false;
    }

    // Start camera and set resolution
    width =  atoi(_argv[2]);
    height = -1;
    if(width == 1280){
        height = 480;
    }else if(width == 2560){
        height = 720;
    }else{
        std::cout << "Not allowed resolution, set it to 1280 or 2560" << std::endl;
        return false;
    }

    VideoCapture zedCamera(atoi(_argv[1]));
    if(!zedCamera.isOpened()){
        std::cout << "Couldn't open camera\n";
        return false;
    }

    zedCamera.set(CV_CAP_PROP_FRAME_WIDTH, width);
    zedCamera.set(CV_CAP_PROP_FRAME_HEIGHT, height);
    if(zedCamera.get(CV_CAP_PROP_FRAME_WIDTH) !=  width){
       std::cout << "Couldn't set camera resolution "<< width << ". Current resolution is " << zedCamera.get(CV_CAP_PROP_FRAME_WIDTH) << std::endl;
        return false;
    }

    boardSize.height = atoi(_argv[4]);
    boardSize.width = atoi(_argv[5]);

    // Start!
    std::string displayerName = "Image displayer";
    namedWindow(displayerName, CV_WINDOW_FREERATIO);
    setMouseCallback(displayerName, mouseCallback);

    cv::Mat left, right;
    std::thread captureThread([&](){
        Mat lLeft, lRight, sideBySide;
        while(gRunning){
            zedCamera >> sideBySide;
            left = sideBySide(Rect(0,0, sideBySide.cols/2, sideBySide.rows)).clone();
            right= sideBySide(Rect(sideBySide.cols/2,0, sideBySide.cols/2, sideBySide.rows)).clone();
            gMutex.lock();
            lLeft.copyTo(left);
            lRight.copyTo(right);
            gMutex.unlock();
            //std::this_thread::sleep_for(std::chrono::milliseconds(10);
        }
    });

    while(gRunning){
        Mat lLeft, lRight;
        gMutex.lock();
        left.copyTo(lLeft);
        right.copyTo(lRight);
        gMutex.unlock();

        if(lLeft.rows == 0 || lRight.rows == 0){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        Mat leftGray, rightGray;
        cvtColor(lLeft, leftGray, COLOR_BGR2GRAY);
        cvtColor(lRight, rightGray, COLOR_BGR2GRAY);
        vector<Point2f> pointBufLeft, pointBufRight;
        bool foundLeft = findChessboardCorners( leftGray, boardSize, pointBufLeft, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK);
        bool foundRight = findChessboardCorners( rightGray, boardSize, pointBufRight, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK);

        if(gSaveNext){
            if (foundLeft && foundRight) {
                cornerSubPix(leftGray, pointBufLeft, Size(7, 7), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
                cornerSubPix(rightGray, pointBufRight, Size(7, 7), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
                _leftPoints.push_back(pointBufLeft);
                _rightPoints.push_back(pointBufRight);
                gSaveNext = false;
                std::cout << "Added a new pair of images for the calibration." << std::endl;
            }
        }

        drawChessboardCorners(lLeft, boardSize, pointBufLeft, foundLeft);
        drawChessboardCorners(lRight, boardSize, pointBufRight, foundRight);

        hconcat(lLeft, lRight, lLeft);
        imshow(displayerName, lLeft);
        waitKey(3);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Chosen " << _leftPoints.size() << " images. Performing calibration with chosen images! Wait...." << std::endl;

    captureThread.join();
    return true;
}

bool imagesFromDataset(int _argc, char **_argv, std::vector<std::vector<cv::Point2f>> &_leftPoints, std::vector<std::vector<cv::Point2f>> &_rightPoints){
    if (_argc != 7) {
        std::cout << "Bad input arguments. Usage: \n ./featureModelCreator [leftPattern] [rightPattern] [calibFile] [boardWidth] [boardHeight] [squareSize]" << std::endl;
        return false;
    }

    cv::VideoCapture leftCamera(_argv[1]);
    cv::VideoCapture rightCamera(_argv[2]);

    if(!leftCamera.isOpened()){
         std::cout << "could't open left images" << std::endl;
         return false;
    }

    if(!rightCamera.isOpened()){
         std::cout << "could't open right images" << std::endl;
         return false;
    }


    boardSize.height = atoi(_argv[4]);
    boardSize.width = atoi(_argv[5]);

    cv::Mat left, right;
    for(;;){
        leftCamera >> left;
        rightCamera >> right;
        if(left.rows == 0 || right.rows == 0){
            break;
        }

        height = left.rows;
        width = left.cols;

        Mat leftGray, rightGray;
        cvtColor(left, leftGray, COLOR_BGR2GRAY);
        cvtColor(right, rightGray, COLOR_BGR2GRAY);
        vector<Point2f> pointBufLeft, pointBufRight;
        bool foundLeft = findChessboardCorners( leftGray, boardSize, pointBufLeft, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK);
        bool foundRight = findChessboardCorners( rightGray, boardSize, pointBufRight, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK);

        if (foundLeft && foundRight) {
            cornerSubPix(leftGray, pointBufLeft, Size(7, 7), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            cornerSubPix(rightGray, pointBufRight, Size(7, 7), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            _leftPoints.push_back(pointBufLeft);
            _rightPoints.push_back(pointBufRight);

            drawChessboardCorners(left, boardSize, pointBufLeft, foundLeft);
            drawChessboardCorners(right, boardSize, pointBufRight, foundRight);

            hconcat(left, right, left);
            imshow("displayerName", left);
            waitKey(100);
        }
    }

    std::cout << "Chosen " << _leftPoints.size() << " images. Performing calibration with chosen images! Wait...." << std::endl;
    return true;
}
