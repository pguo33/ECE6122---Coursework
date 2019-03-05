//
// Created by Peng Guo on 2018/9/9.
//
#include "vector3.h"
#include <iostream>
#include <cmath>
using namespace std;

Vector3::Vector3()//default constructor
{
    x = 0;
    y = 0;
    z = 0;
}

Vector3::Vector3(float xyz)//set x, y, and z to xyz
{
    x = xyz;
    y = xyz;
    z = xyz;
}

Vector3::Vector3(float x, float y, float z)//set component by name
{
    this->x = x;
    this->y = y;
    this->z = z;
}

Vector3 Vector3::operator+(const Vector3& rhs)const//component-wise add
{
    return Vector3(x+rhs.x, y+rhs.y, z+rhs.z);
}

Vector3 Vector3::operator-(const Vector3& rhs)const//component-wise subtract
{
    return Vector3(x-rhs.x, y-rhs.y, z-rhs.z);
}

Vector3 Vector3::operator*(const Vector3& rhs)const//component-wise multiplication
{
    return Vector3(x*rhs.x, y*rhs.y, z*rhs.z);
}

Vector3 Vector3::operator/(const Vector3& rhs)const//component-wise division
{
    return Vector3(x/rhs.x, y/rhs.y, z/rhs.z);
}

Vector3 Vector3::operator+(float rhs)const//add rhs to each component
{
    return Vector3(x + rhs,y + rhs,z + rhs);
}

Vector3 Vector3::operator-(float rhs)const//subtract rhs from each component
{
    return Vector3(x - rhs,y - rhs,z - rhs);
}

Vector3 Vector3::operator*(float rhs)const//multiply each component by rhs
{
    return Vector3(x * rhs,y * rhs,z * rhs);
}

Vector3 Vector3::operator/(float rhs)const//divide each component by rhs
{
    return Vector3(x / rhs,y / rhs,z / rhs);
}

float Vector3::operator|(const Vector3& rhs)const// dot product
{
    return x * rhs.x + y * rhs.y + z * rhs.z;
}

Vector3 Vector3::operator^(const Vector3& rhs)const// cross product
{
    return Vector3(
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
            );
}

Vector3 Vector3::operator+=(const Vector3& rhs)//component-wise add
{
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return Vector3(x,y,z);
}

Vector3 Vector3::operator-=(const Vector3& rhs)//component-wise subtract
{
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return Vector3(x,y,z);
}

Vector3 Vector3::operator*=(const Vector3& rhs)//component-wise multiplication
{
    x *= rhs.x;
    y *= rhs.y;
    z *= rhs.z;
    return Vector3(x,y,z);
}

Vector3 Vector3::operator/=(const Vector3& rhs)//component-wise division
{
    x /= rhs.x;
    y /= rhs.y;
    z /= rhs.z;
    return Vector3(x,y,z);
}

// Vector3++ and ++Vector3 rotate xyz to the right
// i.e. make x = z, y = x, z = y
Vector3& Vector3::operator++()//++v
{
    float temp;
    temp = x;
    x = z;
    z = y;
    y = temp;
    return *this;
}

Vector3 Vector3::operator++(int unused)//++v
{
    Vector3 vector = *this;
    float temp = x;
    x = z;
    z = y;
    y = temp;
    return vector;
}

// Vector3-- and --Vector3 rotate xyz to the left
// i.e. make x = y, y = z, z = x
Vector3& Vector3::operator--()//--v
{
    float temp;
    temp = x;
    x = y;
    y = z;
    z = temp;
    return *this;
}

Vector3 Vector3::operator--(int unused)//v--
{
    Vector3 vector = *this;
    float temp;
    temp = x;
    x = y;
    y = z;
    z = temp;
    return vector;
}

bool Vector3::operator==(const Vector3& rhs)const//component-wise equality
{
    return x == rhs.x && y == rhs.y && z == rhs.z;
}

bool Vector3::operator!=(const Vector3& rhs)const//component-wise inequality
{
    return x != rhs.x || y != rhs.y || z != rhs.z;
}

void Vector3::PrintVector3()//print out vectors
{
    std::cout << "(" << x << "," << y << "," << z << ")" << std::endl;
}

//SE1_VECTOR3_H

