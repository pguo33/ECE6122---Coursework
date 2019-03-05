//
// Created by bpswe on 9/24/2018.
//

//implement your array code here
#include <iostream>
#include <array.h>
using namespace std;

//default constructor
template<class T>
array<T>::array()
{
    m_size = 0;
    m_reserved_size = 0;
    m_elements = nullptr;
}

//initialize array with elements in initializer
template<class T>
array<T>::array(std::initializer_list<T> init_list)
{
    m_size = init_list.size();
    m_reserved_size = m_size;
    m_elements = (T*)malloc(sizeof(T) * m_size);
    auto i = init_list.begin();
    for (size_t j = 0; j < m_size; j++)
    {
        new (&m_elements[j]) T(*(i + j));
    }
}

//copy constructor
template<class T>
array<T>::array(const array& rhs)
{
    m_reserved_size = rhs.m_reserved_size;
    m_size = rhs.m_size;
    m_elements = (T*)malloc(sizeof(T) * m_size);
    for (size_t i = 0; i < m_size; i++)
    {
        new (&m_elements[i]) T(rhs.m_elements[i]);
    }
}

//move constructor
template<class T>
array<T>::array(array&& rhs)
{
    m_reserved_size = rhs.m_reserved_size;
    m_size = rhs.m_size;
    m_elements = rhs.m_elements;
    rhs.m_elements = NULL;
    rhs.m_reserved_size = 0;
    rhs.m_size = 0;
}

//construct with initial "reserved" size
template<class T>
array<T>::array(size_t nReserved)
{
    m_size = 0;
    m_reserved_size = nReserved;
    m_elements = (T*)malloc(sizeof(T) * nReserved);
}

//construct with n copies of t
template<class T>
array<T>::array(size_t n, const T& t)
{
    m_size = n;
    m_reserved_size = 0;
    m_elements = (T*)malloc(sizeof(T) * n);
    for (size_t i = 0; i < n; ++i)
    {
        new (&m_elements[i]) T(t);
    }
}

//destructor
template<class T>
array<T>::~array()
{
    if(m_elements != NULL)
    {
        clear();
        free(m_elements);
    }
}

//ensure enough memory for n elements
template<class T>
void array<T>::reserve(size_t n)
{
    T* newElements = (T*)malloc(sizeof(T) * n);
    for (size_t i = 0; i < m_size; ++i)
    {
        new (&newElements[i]) T(m_elements[i]);
        m_elements[i].~T();
    }
    free(m_elements);
    m_reserved_size = n;
    m_elements = newElements;
}

//add to end of vector
template<class T>
void array<T>::push_back(const T& rhs)
{
    if(m_reserved_size > m_size)
    {
        new(&m_elements[m_size]) T(std::move(rhs));
    }
    else
    {
        T* newElements = (T*)malloc(sizeof(T) * (m_size+1));
        m_reserved_size = m_size + 1;
        //Copy elements to new array
        for (size_t i = 0; i < m_size; ++i)
        {
            new (&newElements[i]) T(std::move(m_elements[i]));
            //call destructor for the old element
            m_elements[i].~T();
        }
        //Copy new pushed element
        new(&newElements[m_size]) T(std::move(rhs));
        //free old memory
        free(m_elements);
        m_elements = newElements;
    }
    m_size++;
}

//add to front of vector
template<class T>
void array<T>::push_front(const T& rhs)
{
    if(m_reserved_size > m_size)
    {
        for(size_t i = m_size; i > 0; --i)
        {
            new (&m_elements[i]) T(std::move(m_elements[i - 1]));
        }
        new (&m_elements[0]) T(std::move(rhs));
        m_size++;
    }
    else
    {
        T* newElements = (T*)malloc(sizeof(T) * (m_size+1));
        for (size_t i = m_size; i > 0; --i)
        {
            new (&newElements[i]) T(std::move(m_elements[i - 1]));
            m_elements[i - 1].~T();
        }
        //free old memory
        free(m_elements);
        m_elements = newElements;
        new (&m_elements[0]) T(std::move(rhs));
        m_size++;
        m_reserved_size = m_size;
    }
}

//remove last element
template<class T>
void array<T>::pop_back()
{
    m_elements[(m_size - 1)].~T();
    m_size--;
}

//remove first element
template<class T>
void array<T>::pop_front()
{
    m_elements[0].~T();
    for (size_t i = 0; i < m_size - 1; ++i)
    {
        m_elements[i] = std::move(m_elements[i+1]);
    }
    m_size--;
}

//return reference to first element
template<class T>
T& array<T>::front() const
{
    return m_elements[0];
}

//return reference to last element
template<class T>
T& array<T>::back() const
{
    return m_elements[m_size - 1];
}

//return reference to specified element
template<class T>
const T& array<T>::operator[](size_t i) const
{
    return m_elements[i];
}

//return reference to specified element
template<class T>
T& array<T>::operator[](size_t i)
{
    return m_elements[i];
}

//return number of elements
template<class T>
size_t array<T>::length() const
{
    return m_size;
}

//returns true if empty
template<class T>
bool array<T>::empty() const
{
    return (m_size == 0);
}

//remove all elements
template<class T>
void array<T>::clear()
{
    m_reserved_size = m_size;
    for(size_t i = 0; i < m_size; i++)
    {
        m_elements[m_size - i - 1].~T();
    }
    m_size = 0;
}

//obtain iterator to first element
template<class T>
array_iterator<T> array<T>::begin() const
{
    return array_iterator<T>(m_elements);
}

//obtain iterator to one beyond element
template<class T>
array_iterator<T> array<T>::end() const
{
    return array_iterator<T>(m_elements + m_size);
}

//remove specified element
template<class T>
void array<T>::erase(const array_iterator<T>& it)
{
    size_t ind;
    for(size_t i = 0; i < m_size; ++i)
    {
        if(it.m_current == &m_elements[i])
        {
            ind = i;
            break;
        }
    }
    m_elements[ind].~T();
    for(size_t i = ind; i < m_size -1; i++)
    {
        //m_elements[i].~T();
        m_elements[i] = std::move(m_elements[i+1]);
    }
    //m_elements[m_size - 1].~T();
    m_size--;
}

//insert element right before it
template<class T>
void array<T>::insert(const T& rhs, const array_iterator<T>& it)
{
    if (m_elements != NULL) {
        size_t ind = 0;
        for (size_t i = 0; i < m_size; ++i) {
            if (it.m_current == &m_elements[i]) {
                ind = i;
                break;
            }
        }
        if (m_size < m_reserved_size) {
            T *newElements = (T *) malloc(sizeof(T) * 1);
            new (&newElements[0]) T(rhs);
            new (&m_elements[m_size]) T(std::move(m_elements[m_size - 1]));
            for (size_t i = m_size - 1; i > ind; i--) {
                m_elements[i] = std::move(m_elements[i - 1]);
            }
            m_elements[ind] = std::move(newElements[0]);
            newElements[0].~T();
            free(newElements);
            m_size++;
        }
        else
            {
            m_reserved_size += 2;
            T *newElements = (T *) malloc(sizeof(T) * (m_reserved_size));
            for (size_t i = 0; i < ind; ++i) {
                new(&newElements[i]) T(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            new(&newElements[ind]) T(rhs);
            for (size_t i = ind; i < m_size; i++)
            {
                new(&newElements[i+1]) T(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            free(m_elements);
            //new(&newElements[ind]) T(std::move(rhs));
            m_elements = newElements;
            m_size++;
            }
    }
    else
        {
            std::cout << "NULL pointer! Cannot erase element." << std::endl;
            abort();
        }
    /*if (m_elements != NULL)
    {
        size_t i = 0;
        bool find = false;
        auto it1 = this -> begin();
        T* newElements = (T*)malloc((m_size + 1) * sizeof(T));
        m_reserved_size = m_size + 1;
        while (it1 != this -> end())
        {
            if (it1 == it)
            {
                new (&newElements[i]) T(std::move(rhs));
                find = true;
            }
            if (find == true)
            {
                new (&newElements[i + 1]) T(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            else
            {
                new (&newElements[i]) T(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            it1++;
            i++;
        }
        free(m_elements);
        m_elements = newElements;
        m_size = m_size + 1;
    }
    else
    {
        std::cout << "NULL pointer! Cannot erase element." << std::endl;
        abort();
    }*/
}

template <class T>
array_iterator<T>::array_iterator()
{
    m_current = 0;
}

template <class T>
array_iterator<T>::array_iterator(T* c)
{
    m_current = c;
}

template <class T>
array_iterator<T>::array_iterator(const array_iterator& rhs)
{
    m_current = &*rhs;
}

template <class T>
T& array_iterator<T>::operator*() const
{
    return *m_current;
}

template <class T>
array_iterator<T> array_iterator<T>::operator++()
{
    m_current = m_current + 1;
    return m_current;
}

template <class T>
array_iterator<T> array_iterator<T>::operator++(int __unused)
{
    T* tmp = m_current;
    m_current = m_current + 1;
    return tmp;
}

template <class T>
bool array_iterator<T>::operator != (const array_iterator& rhs) const
{
    bool tmp = (m_current == &*rhs);
    return !tmp;
}

template <class T>
bool array_iterator<T>::operator == (const array_iterator& rhs) const
{
    return (m_current == &*rhs);
}