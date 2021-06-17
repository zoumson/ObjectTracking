// HARRIS CORNERS TRACKER

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    String keys =
        "{v video |<none>           | video image path}"    
        "{b block |5  | block size}"                                  
        "{help h usage ?    |      | show help message}";      
  
    CommandLineParser parser(argc, argv, keys);
    parser.about("Harris Corners Tracker");
    if (parser.has("help")) 
    {
        parser.printMessage();
        return 0;
    }

    String video = parser.get<String>("video"); 
    int blockSize = parser.get<int>("block");
 
    if (!parser.check()) 
    {
        parser.printErrors();
        return -1;
    }

    // Check if 'blockSize' is smaller than 2
    if(blockSize < 2)
    {
        blockSize = 2;
    }
    
    // Detector parameters
    int apertureSize = 5;
    double k = 0.04;

    int thresh = 200;

    RNG rng(12345);
    string windowName = "Harris Corner Detector";
    
    // Current frame
    Mat frame, frameGray;
    
    char ch;
    
    // Create the capture object
    // 0 -> input arg that specifies it should take the input from the webcam
    VideoCapture cap(video);
    
    if(!cap.isOpened())
    {
        cerr << "Unable to open the webcam. Exiting!" << endl;
        return -1;
    }
    
    // Scaling factor to resize the input frames from the webcam
    float scalingFactor = 0.75;
    
    Mat dst, dst_norm, dst_norm_scaled;
    
    // Iterate until the user presses the Esc key
    while(true)
    {
        // Capture the current frame
        cap >> frame;
        
        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);
        
        dst = Mat::zeros(frame.size(), CV_32FC1);
        
        // Convert to grayscale
        cvtColor(frame, frameGray, COLOR_BGR2GRAY );
        
        // Detecting corners
        cornerHarris(frameGray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
        
        // Normalizing
        normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        convertScaleAbs(dst_norm, dst_norm_scaled);
        
        // Drawing a circle around corners
        for(int j = 0; j < dst_norm.rows ; j++)
        {
            for(int i = 0; i < dst_norm.cols; i++)
            {
                if((int)dst_norm.at<float>(j,i) > thresh)
                {
                    circle(frame, Point(i, j), 8,  Scalar(0,255,0), 2, 8, 0);
                }
            }
        }
        
        // Showing the result
        imshow(windowName, frame);
        
        // Get the keyboard input and check if it's 'Esc'
        // 27 -> ASCII value of 'Esc' key
        ch = waitKey(10);
        if (ch == 27) {
            break;
        }
    }
    
    // Release the video capture object
    cap.release();
    
    // Close all windows
    destroyAllWindows();
    
    return 1;
}
