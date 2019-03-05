#include <iostream>
#include "vector3.h"

using namespace std;

int main() {
    //test constructor
    Vector3 v1,v2,v3,v4,v5,v6,v7;
    v2 = Vector3(1.0f);
    v3 = Vector3(1.0,2.0,3.0);
    v4 = Vector3(4.0,3.6,2.7);
    v5 = Vector3(2.0,3.0,4.0);
    v6 = Vector3(2.0,3.0,4.0);
    v7 = Vector3(3.0,4.0,5.0);
    v1.PrintVector3();
    v2.PrintVector3();
    v3.PrintVector3();
    v4.PrintVector3();
    v5.PrintVector3();
    v6.PrintVector3();
    v7.PrintVector3();

    //test component-wise operations
    Vector3 v_x(2.0,3.0,4.0);
    Vector3 v_y(3.0,5.0,5.0);
    Vector3 v_add = v_x + v_y;
    Vector3 v_sub = v_x - v_y;
    Vector3 v_mul = v_x * v_y;
    Vector3 v_div = v_x / v_y;
    v_add.PrintVector3();
    v_sub.PrintVector3();
    v_mul.PrintVector3();
    v_div.PrintVector3();

    //test operations with each component by rhs
    float a = 2.0;
    v1 = v1 + a;
    v2 = v2 - a;
    v3 = v3 * a;
    v4 = v4 / a;
    v1.PrintVector3();
    v2.PrintVector3();
    v3.PrintVector3();
    v4.PrintVector3();

    //test dot and cross product
    float v_dot = v1 | v2;
    Vector3 v_cross = v1 ^ v2;
    std:cout << "v_dot= " << v_dot <<std::endl;
    v_cross.PrintVector3();

    //test component-wise operations
    v1 += v5;
    v2 -= v5;
    v3 *= v5;
    v4 /= v5;
    v1.PrintVector3();
    v2.PrintVector3();
    v3.PrintVector3();
    v4.PrintVector3();

    //test rotation operation
    Vector3 v_right1 = v1++;
    Vector3 v_right2 = ++v2;
    Vector3 v_left1 = v3--;
    Vector3 v_left2 = --v4;
    v_right1.PrintVector3();
    v1.PrintVector3();
    v_right2.PrintVector3();
    v_left1.PrintVector3();
    v3.PrintVector3();
    v_left2.PrintVector3();

    //test component-wise equality and inequality
    if (v5 == v6)
    {
        std::cout << "v5=v6" <<std::endl;
    }
    else
    {
        std::cout << "v5!=v6" <<std::endl;
    }

    if (v5 == v7)
    {
        std::cout << "v5=v7" <<std::endl;
    }
    else
    {
        std::cout << "v5!=v7" <<std::endl;
    }
    if (v5 != v6)
    {
        std::cout << "v5!=v6" <<std::endl;
    }
    else
    {
        std::cout << "v5=v6" <<std::endl;
    }

    if (v5 != v7)
    {
        std::cout << "v5!=v7" <<std::endl;
    }
    else
    {
        std::cout << "v5=v7" <<std::endl;
    }

}