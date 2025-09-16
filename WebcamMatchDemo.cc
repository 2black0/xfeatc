#include <iostream>
#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"
#include "XFeat.h"
#include "Matcher.h"

// Function to initialize webcam with fallback options
cv::VideoCapture initializeWebcam() {
    cv::VideoCapture cap;
    
    // Try different backends
    std::vector<int> backends = {
        cv::CAP_V4L2,
        cv::CAP_GSTREAMER,
        cv::CAP_ANY
    };
    
    for (int backend : backends) {
        std::cout << "Trying to open webcam with backend: " << backend << std::endl;
        if (backend == cv::CAP_GSTREAMER) {
            // For GStreamer, try with specific pipeline
            cap.open(0, cv::CAP_GSTREAMER);
        } else {
            cap.open(0, backend);
        }
        
        if (cap.isOpened()) {
            std::cout << "Successfully opened webcam with backend: " << backend << std::endl;
            
            // Test if we can grab a frame
            cv::Mat testFrame;
            if (cap.read(testFrame) && !testFrame.empty()) {
                std::cout << "Webcam is working correctly with resolution: " 
                         << testFrame.cols << "x" << testFrame.rows << std::endl;
                return cap;
            } else {
                std::cout << "Could not read frame from webcam, trying next backend" << std::endl;
                cap.release();
            }
        } else {
            std::cout << "Failed to open webcam with backend: " << backend << std::endl;
        }
    }
    
    return cap;
}

// Function to resize image to exact dimensions (with padding if necessary)
cv::Mat resizeToExactSize(const cv::Mat& input, int targetWidth, int targetHeight) {
    cv::Mat output(targetHeight, targetWidth, input.type(), cv::Scalar(0, 0, 0));
    
    // Calculate scaling factor to fit within target dimensions while maintaining aspect ratio
    double scale = std::min(static_cast<double>(targetWidth) / input.cols, 
                           static_cast<double>(targetHeight) / input.rows);
    
    int newWidth = static_cast<int>(input.cols * scale);
    int newHeight = static_cast<int>(input.rows * scale);
    
    // Resize the image
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(newWidth, newHeight));
    
    // Center the resized image in the output
    int xOffset = (targetWidth - newWidth) / 2;
    int yOffset = (targetHeight - newHeight) / 2;
    
    resized.copyTo(output(cv::Rect(xOffset, yOffset, newWidth, newHeight)));
    
    return output;
}

int main(int argc, char** argv) {
    // Parse arguments
    const std::string argKeys =
            "{model | ../model/xfeat_640x640.onnx | model file path}"
            "{ransac | 0 | use RANSAC to refine matches}";
    cv::CommandLineParser parser(argc, argv, argKeys);
    auto modelFile = parser.get<std::string>("model");
    auto useRansac = parser.get<int>("ransac");
    std::cout << "model file: " << modelFile << std::endl;
    std::cout << "use RANSAC: " << (useRansac ? "true" : "false") << std::endl;

    // Create XFeat object
    std::cout << "creating XFeat...\n";
    XFeat xfeat(modelFile);

    // Initialize webcam
    std::cout << "Initializing webcam...\n";
    cv::VideoCapture cap = initializeWebcam();
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam\n";
        return -1;
    }

    // Model dimensions
    const int modelWidth = 640;
    const int modelHeight = 640;

    // Capture first frame as reference
    std::cout << "Capturing reference frame...\n";
    cv::Mat refFrame, refGray, refResized;
    bool refCaptured = false;
    
    // Reference features
    std::vector<cv::KeyPoint> refKeys;
    cv::Mat refDescs;
    cv::Mat refDisplay;  // Color version for display
    
    // Current frame variables
    cv::Mat currentFrame, currentGray, currentResized;
    std::vector<cv::KeyPoint> currentKeys;
    cv::Mat currentDescs;
    cv::Mat currentDisplay;  // Color version for display
    
    // Matching variables
    std::vector<cv::DMatch> matches;
    
    std::cout << "Instructions:\n";
    std::cout << "1. First frame will be captured as reference\n";
    std::cout << "2. Press 'q' to quit\n";
    std::cout << "3. Press 'r' to recapture reference frame\n";

    // Define panel sizes - all panels will be the same size
    const int panelWidth = 640;
    const int panelHeight = 480;
    
    // Create combined display (2x2 grid, but we'll only use 3 panels)
    cv::Mat combinedDisplay(panelHeight * 2, panelWidth * 2, CV_8UC3, cv::Scalar(30, 30, 30));

    while (true) {
        // Capture current frame
        cv::Mat rawFrame;
        cap >> rawFrame;
        if (rawFrame.empty()) {
            std::cerr << "Error: Blank frame grabbed\n";
            continue;
        }

        // Convert to grayscale for processing
        cv::Mat rawGray;
        if (rawFrame.channels() == 3) {
            cv::cvtColor(rawFrame, rawGray, cv::COLOR_BGR2GRAY);
        } else {
            rawGray = rawFrame;
        }

        // Resize for model processing
        cv::resize(rawGray, currentResized, cv::Size(modelWidth, modelHeight));
        
        // Keep color version for display
        if (rawFrame.channels() == 3) {
            currentFrame = rawFrame;
        } else {
            cv::cvtColor(rawFrame, currentFrame, cv::COLOR_GRAY2BGR);
        }

        // Capture reference frame on first run
        if (!refCaptured) {
            refFrame = currentFrame.clone();
            refGray = currentResized.clone();
            
            // Detect features on reference
            xfeat.DetectAndCompute(refGray, refKeys, refDescs, 1000);
            
            // Create color display version
            cv::cvtColor(refGray, refDisplay, cv::COLOR_GRAY2BGR);
            cv::drawKeypoints(refDisplay, refKeys, refDisplay, cv::Scalar(0, 0, 255));
            cv::putText(refDisplay, "Reference: " + std::to_string(refKeys.size()) + " features", 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
            
            refCaptured = true;
            std::cout << "Reference frame captured with " << refKeys.size() << " features\n";
        }

        // Detect features on current frame
        xfeat.DetectAndCompute(currentResized, currentKeys, currentDescs, 1000);
        
        // Create color display version for current frame
        cv::cvtColor(currentResized, currentDisplay, cv::COLOR_GRAY2BGR);
        cv::drawKeypoints(currentDisplay, currentKeys, currentDisplay, cv::Scalar(0, 0, 255));
        cv::putText(currentDisplay, "Current: " + std::to_string(currentKeys.size()) + " features", 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        // Match features
        Matcher::Match(refDescs, currentDescs, matches, 0.82f);

        // Apply RANSAC if requested
        if (useRansac && matches.size() > 8) {
            std::vector<cv::Point2f> pts1, pts2;
            for (auto& m : matches) {
                pts1.push_back(refKeys[m.queryIdx].pt);
                pts2.push_back(currentKeys[m.trainIdx].pt);
            }
            Matcher::RejectBadMatchesF(pts1, pts2, matches, 4.0f);
        }

        // Draw matches
        cv::Mat matchesDisplay;
        cv::Mat refDisplayGray;
        cv::cvtColor(refGray, refDisplayGray, cv::COLOR_GRAY2BGR);
        cv::drawKeypoints(refDisplayGray, refKeys, refDisplayGray, cv::Scalar(0, 0, 255));
        cv::putText(refDisplayGray, "Reference: " + std::to_string(refKeys.size()) + " features", 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        
        cv::drawMatches(refDisplayGray, refKeys, currentDisplay, currentKeys, matches, matchesDisplay);
        cv::putText(matchesDisplay, "Matches: " + std::to_string(matches.size()), 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        // Resize all panels to exact same dimensions
        cv::Mat refPanel = resizeToExactSize(refFrame, panelWidth, panelHeight);
        cv::Mat currentPanel = resizeToExactSize(currentFrame, panelWidth, panelHeight);
        cv::Mat matchesPanel = resizeToExactSize(matchesDisplay, panelWidth * 2, panelHeight);
        
        // Place reference image (top-left)
        refPanel.copyTo(combinedDisplay(cv::Rect(0, 0, panelWidth, panelHeight)));
        
        // Place current image (top-right)
        currentPanel.copyTo(combinedDisplay(cv::Rect(panelWidth, 0, panelWidth, panelHeight)));
        
        // Place matches (bottom - spans both columns)
        matchesPanel.copyTo(combinedDisplay(cv::Rect(0, panelHeight, panelWidth * 2, panelHeight)));
        
        // Draw borders around panels for clarity
        cv::rectangle(combinedDisplay, cv::Rect(0, 0, panelWidth, panelHeight), cv::Scalar(100, 100, 100), 2);
        cv::rectangle(combinedDisplay, cv::Rect(panelWidth, 0, panelWidth, panelHeight), cv::Scalar(100, 100, 100), 2);
        cv::rectangle(combinedDisplay, cv::Rect(0, panelHeight, panelWidth * 2, panelHeight), cv::Scalar(100, 100, 100), 2);
        
        // Add labels
        cv::putText(combinedDisplay, "REFERENCE", cv::Point(10, panelHeight - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(combinedDisplay, "LIVE", cv::Point(panelWidth + 10, panelHeight - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(combinedDisplay, "MATCHES", cv::Point(10, panelHeight * 2 - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // Display combined view
        cv::imshow("Webcam Match Demo - Reference | Live | Matches", combinedDisplay);

        // Handle key presses
        int key = cv::waitKey(30) & 0xFF;
        if (key == 'q') {
            break;
        } else if (key == 'r') {
            // Recapture reference frame
            refCaptured = false;
            refKeys.clear();
            refDescs.release();
            std::cout << "Ready to capture new reference frame\n";
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();
    return 0;
}