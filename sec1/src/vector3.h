//
// Created by Peng Guo on 2018/9/8.
//

#ifndef SE1_VECTOR3_H
#define SE1_VECTOR3_H

#include <iostream>
#include <vector>

struct Vector3{
    float x;
    float y;
    float z;

    Vector3();//default constructor

    Vector3(float xyz);//set x, y, and z to xyz

    Vector3(float x, float y, float z);//set component by name


    Vector3 operator+(const Vector3& rhs)const;//component-wise add

    Vector3 operator-(const Vector3& rhs)const;//component-wise subtract

    Vector3 operator*(const Vector3& rhs)const;//component-wise multiplication

    Vector3 operator/(const Vector3& rhs)const;//component-wise division


    Vector3 operator+(float rhs)const;//add rhs to each component

    Vector3 operator-(float rhs)const;//subtract rhs from each component

    Vector3 operator*(float rhs)const;//multiply each component by rhs

    Vector3 operator/(float rhs)const;//divide each component by rhs


    float operator|(const Vector3& rhs)const;// dot product

    Vector3 operator^(const Vector3& rhs)const;// cross product


    Vector3 operator+=(const Vector3& rhs);//component-wise add

    Vector3 operator-=(const Vector3& rhs);//component-wise subtract

    Vector3 operator*=(const Vector3& rhs);//component-wise multiplication

    Vector3 operator/=(const Vector3& rhs);//component-wise division

// Vector3++ and ++Vector3 rotate xyz to the right
// i.e. make x = z, y = x, z = y
    Vector3& operator++();//++v

    Vector3 operator++(int _unused);//v++

// Vector3-- and --Vector3 rotate xyz to the left
// i.e. make x = y, y = z, z = x
    Vector3& operator--();//--v

    Vector3 operator--(int _unused);//v--


    bool operator==(const Vector3& rhs)const;//component-wise equality

    bool operator!=(const Vector3& rhs)const;//component-wise inequality


    void PrintVector3();//print out vectors

};

#endif //SEC1_VECTOR3_H
