/*
 *--------------------------------------------------------------------------------------------------
 * DS-SLAM: A Semantic Visual SLAM towards Dynamic Environments
　*　Author(s):
 * Chao Yu, Zuxin Liu, Xinjun Liu, Fugui Xie, Yi Yang, Qi Wei, Fei Qiao qiaofei@mail.tsinghua.edu.cn
 * Created by Yu Chao@2018.12.03
 * --------------------------------------------------------------------------------------------------
 * DS-SLAM is a optimized SLAM system based on the famous ORB-SLAM2. If you haven't learn ORB_SLAM2 code, 
 * you'd better to be familiar with ORB_SLAM2 project first. Compared to ORB_SLAM2, 
 * we add anther two threads including semantic segmentation thread and densemap creation thread. 
 * You should pay attention to Frame.cc, ORBmatcher.cc, Pointcloudmapping.cc and Segment.cc.
 * 
 *　@article{murORB2,
 *　title={{ORB-SLAM2}: an Open-Source {SLAM} System for Monocular, Stereo and {RGB-D} Cameras},
　*　author={Mur-Artal, Ra\'ul and Tard\'os, Juan D.},
　* journal={IEEE Transactions on Robotics},
　*　volume={33},
　* number={5},
　* pages={1255--1262},
　* doi = {10.1109/TRO.2017.2705103},
　* year={2017}
 *　}
 * --------------------------------------------------------------------------------------------------
 * Copyright (C) 2018, iVip Lab @ EE, THU (https://ivip-tsinghua.github.io/iViP-Homepage/) and 
 * Advanced Mechanism and Roboticized Equipment Lab. All rights reserved.
 *
 * Licensed under the GPLv3 License;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * https://github.com/ivipsourcecode/DS-SLAM/blob/master/LICENSE
 *--------------------------------------------------------------------------------------------------
 */

#include "Segment.h"
#include "Tracking.h"
//#include "Camera.h"
#include <fstream>
#define SKIP_NUMBER 1
using namespace std;

namespace ORB_SLAM2
{
Segment::Segment(const string &pascal_prototxt, const string &pascal_caffemodel, const string &pascal_png, const string &strSettingPath)
                :mbFinishRequested(false),mSkipIndex(SKIP_NUMBER),mSegmentTime(0),imgIndex(0)
{

    model_file = pascal_prototxt;
    trained_file = pascal_caffemodel;
    LUT_file = pascal_png;

    label_colours = cv::imread(LUT_file,1);
    cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    bool bParse = ParseSegmentationParamFile(fSettings);

    if(!bParse)
    {
        std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
        try
        {
            throw -1;
        }
        catch(exception &e)
        {

        }
    }
    // std::cout << "Img width: " << s_width << " Img height: " << s_height << std::endl;
    mImgSegmentLatest=cv::Mat(s_height,s_width,CV_8UC1);
    
    mImgSegment_color_final=cv::Mat(s_height,s_width,CV_8UC3);
    
    mbNewImgFlag=false;

}

void Segment::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

bool Segment::isNewImgArrived()
{
    unique_lock<mutex> lock(mMutexGetNewImg);
    if(mbNewImgFlag)
    {
        mbNewImgFlag=false;
        return true;
    }
    else
    return false;
}

void Segment::Run()
{

    classifier=new Classifier(model_file, trained_file);
    cout << "Load model ..."<<endl;
    vector<float> vTimesDetectSegNet;
    while(1)
    {
        usleep(1);
        if(!isNewImgArrived())
            continue;
        if(!mImg.empty())
        {
            //cout << "Wait for new RGB img time =" << endl;
            if(mSkipIndex==SKIP_NUMBER)
            {
                std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
                // Recognise by Semantin segmentation
                mImgSegment=classifier->Predict(mImg, label_colours);

                mImgSegment_color = mImgSegment.clone();
                cv::cvtColor(mImgSegment,mImgSegment_color, CV_GRAY2BGR);

                LUT(mImgSegment_color, label_colours, mImgSegment_color_final);
                cv::resize(mImgSegment, mImgSegment, cv::Size(s_width,s_height) );
                cv::resize(mImgSegment_color_final, mImgSegment_color_final, cv::Size(s_width,s_height) );
                cv::Mat temp = mImgSegment_color_final;
                int morph_size = 15;
                cv::Mat kernel = getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
                cv::morphologyEx(mImgSegment_color_final, temp, 2, kernel);
                std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
                mSegmentTime = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
                vTimesDetectSegNet.push_back(mSegmentTime);
                mSkipIndex=0;
                imgIndex++;
                mImgSegment_color_final = temp.clone();
                cv::imshow("SegNet", mImgSegment_color_final); cv::waitKey(1);
            }
            mSkipIndex++;
            ProduceImgSegment();
            
        }
        if(CheckFinish())
        {
            break;
        }
    }

    // // SegNet time statistics
    sort(vTimesDetectSegNet.begin(),vTimesDetectSegNet.end());
    float totaltime = 0;
    for(int ni=0; ni<vTimesDetectSegNet.size(); ni++)
    {
        totaltime+=vTimesDetectSegNet[ni];
    }
    ofstream tStatics;
    cout << "-------" << endl << endl;
    cout << "total KFs SegNet processed: " << vTimesDetectSegNet.size() << endl;
    cout << "median SegNet time: " << vTimesDetectSegNet[vTimesDetectSegNet.size()/2] << endl;
    cout << "mean SegNet time: " << totaltime/vTimesDetectSegNet.size() << endl;
    tStatics.open("time_statistic_SegNet.txt");
    tStatics << "total KFs SegNet processed: " << vTimesDetectSegNet.size() << endl;
    tStatics << "median SegNet time: " << vTimesDetectSegNet[vTimesDetectSegNet.size()/2] << endl;
    tStatics << "mean SegNet time: " << totaltime/vTimesDetectSegNet.size() << endl;
    tStatics.close();
    SetFinish();

}

bool Segment::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}
  
void Segment::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested=true;
}

bool Segment::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Segment::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
}

void Segment::ProduceImgSegment()
{
    std::unique_lock <std::mutex> lock(mMutexNewImgSegment);
    mImgTemp=mImgSegmentLatest;
    mImgSegmentLatest=mImgSegment;
    mImgSegment=mImgTemp;
    
}

bool Segment::ParseSegmentationParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Camera.width"];
    if(!node.empty())
    {
        s_width = node.real();
    }
    else
    {
        std::cerr << "*Camera.width parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Camera.height"];
    if(!node.empty())
    {
        s_height = node.real();
    }
    else
    {
        std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }
}

}   //ORB_SLAM2
