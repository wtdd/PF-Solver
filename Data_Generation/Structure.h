#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include<iostream>
#include <fstream>
#include <string>
#include<vector>
#include<map>
#include<list>
#include<set>
#include <windows.h>
#include<algorithm>
#include<complex>
#include <iomanip>
using namespace std;
typedef struct s0 {
    int bus_number = 0;
    int bus_type = 0;
    long double Pd = 0.0;         //结点注入有功
    long double Qd = 0.0;         //结点注入无功
    long double Gs = 0.0;
    long double Bs = 0.0;
    long double area = 0.0;
    long double Vm = 0.0;
    long double Va = 0.0;
    long double baseKV = 0.0;
    long double zone = 0.0;
    long double maxVm = 0.0;
    long double minVm = 0.0;
}Bus;
typedef struct s6 {
    double a[50];
    double all;
}cal_time;

typedef struct s1 {
    int fr = 0;
    int to = 0;
    long double r = 0.0;         //结点注入有功
    long double x = 0.0;         //结点注入无功
    long double b = 0.0;
    long double rateA = 0.0;
    long double rateB = 0.0;
    long double rateC = 0.0;
    long double _ratio = 0.0;
    long double angel = 0.0;
    long double status = 0.0;
    long double angmin = 0.0;
    long double angmax = 0.0;
}Line;

typedef struct gendata {
    long double Pg;
    long double Qg;
    long double Qmax;
    long double Qmin;
}Gen;

typedef struct s2 {
    int type = 0;
    long double mes_value = 0.0;
    long double add_error = 0.0;
    long double true_value = 0.0;
    long double weight = 0.0;
    int fr = 0;
    int to = 0;
    long double mes_index = 0.0;
}Mdata;

typedef struct s3 {
    Bus bus;
    vector<Line> lines;
    Gen gendata;
    pair<double, double> ans;
}node_Physics;

typedef struct s4 {
    //第一步初始化：得到Ya Yr 和 邻接结点
    //vector< pair <int, complex<double>>> Ya; //to,value的形式
    vector< pair <int, complex<double>>> Yr; //存放到vector中
    // 自导纳数据
    complex<double> Y;
    // 测量节点
    long double p = 0.0, q = 0.0, v = 0.0;
    long double ori_p = 0.0, ori_q = 0.0, ori_v = 0.0;
    map<int, double>H,M,K,L;
    //右边向量
    long double right_hp = 0.0;
    long double right_hq = 0.0;
}Node;

class stop_watch
{
public:
    stop_watch()
        : elapsed_(0)
    {
        QueryPerformanceFrequency(&freq_);
        QueryPerformanceCounter(&begin_time_);
    }
    ~stop_watch() {}
public:
    void start()
    {
        QueryPerformanceCounter(&begin_time_);
    }
    void stop()
    {
        LARGE_INTEGER end_time;
        QueryPerformanceCounter(&end_time);
        elapsed_ += (end_time.QuadPart - begin_time_.QuadPart) * 1000000 / freq_.QuadPart;
    }
    void restart()
    {
        elapsed_ = 0;
        start();
    }
    //微秒
    long double elapsed()
    {
        return static_cast<double>(elapsed_);
    }
    //毫秒
    long double elapsed_ms()
    {
        return elapsed_ / 1000.0;
    }
    //秒
    long double elapsed_second()
    {
        return elapsed_ / 1000000.0;
    }

private:
    LARGE_INTEGER freq_;
    LARGE_INTEGER begin_time_;
    long long elapsed_;
};