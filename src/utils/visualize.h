//
// Created by ziyin on 10/25/18.
//

#ifndef DEEPSIM_VISUALIZE_H
#define DEEPSIM_VISUALIZE_H

#include "../include/vec.h"

struct color_bar{
    template<class T>
    inline T lerp(T a, T b, float c)
    {
        return (1-c)*a + c*b;
    }
    inline Vec3uc color(const float &point)
    {
        float x = std::min(std::max(point,0.0f), .99f);
        int i = x*10.0;
        float fx = x*10.0-i;
        Vec3f color = lerp(colorBar[i], colorBar[i+1], fx);
        return Vec3uc(color[0]*255, color[1]*255, color[2]*255);
    }
    float src_max;
    std::vector<Vec3f> colorBar;
    color_bar(){}
    color_bar(const color_bar & b)
    {
        src_max = b.src_max;
        colorBar.resize(11);
        colorBar[0] = Vec3f(0,0.007195,0.2590);
        colorBar[1] = Vec3f(0, 0, 0.5);
        colorBar[2] = Vec3f(0, 0.3375, 0.9);
        colorBar[3] = Vec3f(0, 0.57, 0.9);
        colorBar[4] = Vec3f(0.0032514, 0.735, 0.181);
        colorBar[5] = Vec3f(0.0065028, 0.9, 0.100473);
        colorBar[6] = Vec3f(0.228251, 0.9, 0.0502);
        colorBar[7] = Vec3f(0.45, 0.9, 0.0);
        colorBar[8] = Vec3f(0.9, 0.45, 0.0);
        colorBar[9] = Vec3f(0.9, 0, 0.0);
        colorBar[10] = Vec3f(0.9, 0, 0.0);
    }

    explicit color_bar(const float &max_val)
    {
        src_max = max_val;
        colorBar.resize(11);
        colorBar[0] = Vec3f(0,0.007195,0.2590);
        colorBar[1] = Vec3f(0, 0, 0.5);
        colorBar[2] = Vec3f(0, 0.3375, 0.9);
        colorBar[3] = Vec3f(0, 0.57, 0.9);
        colorBar[4] = Vec3f(0.0032514, 0.735, 0.181);
        colorBar[5] = Vec3f(0.0065028, 0.9, 0.100473);
        colorBar[6] = Vec3f(0.228251, 0.9, 0.0502);
        colorBar[7] = Vec3f(0.45, 0.9, 0.0);
        colorBar[8] = Vec3f(0.9, 0.45, 0.0);
        colorBar[9] = Vec3f(0.9, 0, 0.0);
        colorBar[10] = Vec3f(0.3, 0, 0.0);
    }



    Vec3uc toRGB(float val)
    {
        return color(val/10);
    }
};


#endif //DEEPSIM_VISUALIZE_H
