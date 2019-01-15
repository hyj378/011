//이미지 비교해서 원하는 대상 찾기

#include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
	//검출 -> 기술 -> 매칭
	Mat img1; 
//*****************각자 검출하고싶은 물체의 이미지 파일 위치를 대입
	img1 = imread("C:/opencv_source/image/model3.jpg", IMREAD_GRAYSCALE);
	if (!(img1.data))
	{
		printf("no img");
		return 0;
	}
//*****************1.검출 with sift (SiftFeatureDetector)
	Ptr<xfeatures2d::SIFT> instance_FeatureDetector = xfeatures2d::SIFT::create();//검출을 위한 인스턴스 생성
	std::vector<KeyPoint> img1keypoint;
	instance_FeatureDetector->detect(img1, img1keypoint);//Feature2D를 상속받았으므로 사용가능

	VideoCapture cap;
	cap.open(0);
	//자기장치에 달려있는 기본카메라 open하려면 open(0)
	//아니면  https://docs.opencv.org/3.4.3/d8/dfe/classcv_1_1VideoCapture.html 참조
	if (!cap.isOpened()) //1 + CAP_MSMF
	{
		std::cout << "카메라 작동 불가" << std::endl;
		return -1;
	}

	UMat cam;
	namedWindow("camera", 1);

	for (;;) {
		cap.read(cam); // retrieve
		imshow("camera", cam);
//*****************이미지 준비 완료 1.검출 with sift (SiftFeatureDetector)
		std::vector<KeyPoint> camkeypoint;
		instance_FeatureDetector->detect(cam, camkeypoint);

		std::cout << "특징점 검출 완료" << std::endl;

		//제공된 모든 이미지의 키포인트 검출 완료
//*****************1.기술 with sift (SiftDescriptorExtractor)
		Ptr<xfeatures2d::SIFT> instance_Descriptor = xfeatures2d::SIFT::create();
		Mat img1outputarray, camoutputarray;
		instance_Descriptor->compute(img1, img1keypoint, img1outputarray);
		instance_Descriptor->compute(cam, camkeypoint, camoutputarray);

		std::cout << "기술 완료" << std::endl;



		Mat displayOfImg1;
		drawKeypoints(img1, img1keypoint, displayOfImg1, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//빨간색 DRAW_RICH_KEYPOINTS로 나타냄
		namedWindow("img1의 키포인트"); imshow("img1의 키포인트", displayOfImg1);

		Mat displayOfImg2;
		drawKeypoints(cam, camkeypoint, displayOfImg2, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//빨간색 DRAW_RICH_KEYPOINTS로 나타냄
		namedWindow("img2의 키포인트"); imshow("img2의 키포인트", displayOfImg2);
		//키포인트 검출 성공을 확인했으므로 키포인트 기술하기
		waitKey();
//*****************1.매칭 with sift (SiftDescriptorExtractor)
		FlannBasedMatcher FLANNmatcher;
		std::vector<DMatch> match;
		FLANNmatcher.match(camoutputarray, img1outputarray, match);

		if (!(match.size())) 
		{
			Mat tt;
			drawKeypoints(cam, camkeypoint, tt, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			namedWindow("키포인트 매칭불가..."); imshow("키포인트 매칭불가...", tt);
			waitKey();
			return -1;
		}

		double maxd = 0; double mind = match[0].distance;
		for (int i = 0; i < match.size() ; i++)
		{
			double dist = match[i].distance;
			if (dist < mind) mind = dist;
			if (dist > maxd) maxd = dist;
		}

		std::vector<DMatch> good_match;
		for (int i = 0; i < match.size(); i++) 
			if (match[i].distance <= max(2 * mind, 0.02)) good_match.push_back(match[i]);		


		Mat finalOutputImg;
		std::cout << "good match 의 갯수는: "<<good_match.size() << std::endl;

		std::cout << "여기까지는 괜찮아여" << std::endl;
		drawMatches(img1, img1keypoint, cam, camkeypoint, good_match, finalOutputImg, Scalar(150, 30, 200), Scalar(0, 0, 255), std::vector< char >(), DrawMatchesFlags::DEFAULT);
		namedWindow("매칭 결과"); imshow("매칭결과", finalOutputImg);
		std::cout << "매칭 완료" << std::endl;

		if (waitKey(30) >= 0) break;
	}

	return 0;
}