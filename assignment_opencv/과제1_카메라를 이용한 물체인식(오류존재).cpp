//�̹��� ���ؼ� ���ϴ� ��� ã��

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
	//���� -> ��� -> ��Ī
	Mat img1; 
//*****************���� �����ϰ���� ��ü�� �̹��� ���� ��ġ�� ����
	img1 = imread("C:/opencv_source/image/model3.jpg", IMREAD_GRAYSCALE);
	if (!(img1.data))
	{
		printf("no img");
		return 0;
	}
//*****************1.���� with sift (SiftFeatureDetector)
	Ptr<xfeatures2d::SIFT> instance_FeatureDetector = xfeatures2d::SIFT::create();//������ ���� �ν��Ͻ� ����
	std::vector<KeyPoint> img1keypoint;
	instance_FeatureDetector->detect(img1, img1keypoint);//Feature2D�� ��ӹ޾����Ƿ� ��밡��

	VideoCapture cap;
	cap.open(0);
	//�ڱ���ġ�� �޷��ִ� �⺻ī�޶� open�Ϸ��� open(0)
	//�ƴϸ�  https://docs.opencv.org/3.4.3/d8/dfe/classcv_1_1VideoCapture.html ����
	if (!cap.isOpened()) //1 + CAP_MSMF
	{
		std::cout << "ī�޶� �۵� �Ұ�" << std::endl;
		return -1;
	}

	UMat cam;
	namedWindow("camera", 1);

	for (;;) {
		cap.read(cam); // retrieve
		imshow("camera", cam);
//*****************�̹��� �غ� �Ϸ� 1.���� with sift (SiftFeatureDetector)
		std::vector<KeyPoint> camkeypoint;
		instance_FeatureDetector->detect(cam, camkeypoint);

		std::cout << "Ư¡�� ���� �Ϸ�" << std::endl;

		//������ ��� �̹����� Ű����Ʈ ���� �Ϸ�
//*****************1.��� with sift (SiftDescriptorExtractor)
		Ptr<xfeatures2d::SIFT> instance_Descriptor = xfeatures2d::SIFT::create();
		Mat img1outputarray, camoutputarray;
		instance_Descriptor->compute(img1, img1keypoint, img1outputarray);
		instance_Descriptor->compute(cam, camkeypoint, camoutputarray);

		std::cout << "��� �Ϸ�" << std::endl;



		Mat displayOfImg1;
		drawKeypoints(img1, img1keypoint, displayOfImg1, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//������ DRAW_RICH_KEYPOINTS�� ��Ÿ��
		namedWindow("img1�� Ű����Ʈ"); imshow("img1�� Ű����Ʈ", displayOfImg1);

		Mat displayOfImg2;
		drawKeypoints(cam, camkeypoint, displayOfImg2, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//������ DRAW_RICH_KEYPOINTS�� ��Ÿ��
		namedWindow("img2�� Ű����Ʈ"); imshow("img2�� Ű����Ʈ", displayOfImg2);
		//Ű����Ʈ ���� ������ Ȯ�������Ƿ� Ű����Ʈ ����ϱ�
		waitKey();
//*****************1.��Ī with sift (SiftDescriptorExtractor)
		FlannBasedMatcher FLANNmatcher;
		std::vector<DMatch> match;
		FLANNmatcher.match(camoutputarray, img1outputarray, match);

		if (!(match.size())) 
		{
			Mat tt;
			drawKeypoints(cam, camkeypoint, tt, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			namedWindow("Ű����Ʈ ��Ī�Ұ�..."); imshow("Ű����Ʈ ��Ī�Ұ�...", tt);
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
		std::cout << "good match �� ������: "<<good_match.size() << std::endl;

		std::cout << "��������� �����ƿ�" << std::endl;
		drawMatches(img1, img1keypoint, cam, camkeypoint, good_match, finalOutputImg, Scalar(150, 30, 200), Scalar(0, 0, 255), std::vector< char >(), DrawMatchesFlags::DEFAULT);
		namedWindow("��Ī ���"); imshow("��Ī���", finalOutputImg);
		std::cout << "��Ī �Ϸ�" << std::endl;

		if (waitKey(30) >= 0) break;
	}

	return 0;
}