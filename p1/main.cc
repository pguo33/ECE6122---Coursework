#include <iostream>
#include <vector>
#include "simple_string.h"
#include "array.h"
#include "array.cc"

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//NOTE THIS IS NOT A COMPLETE LISTING OF TESTS THAT WILL BE RUN ON YOUR CODE
//Just a sample to help get you started and give you an idea of how i'll be testing
//Above each test gives the counts for std::vector and the solution i've written for your array
//As well as checking totals ensure your array doesn't leak memory
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//comment/uncomment these lines to enable tests
#define TEST_PUSH_BACK_NEW_VEC
#define TEST_CLEAR
#define TEST_PUSH_FRONT_VEC
#define TEST_PUSH_FRONT_WITH_RESERVE
#define TEST_POP_BACK
#define TEST_INITIALIZER_LIST
#define TEST_POP_FRONT
#define TEST_COPY_CONSTRUCTOR
#define TEST_ITERATOR
#define TEST_ERASE1
#define TEST_INSERT
#define TEST_INSERT1
#define TEST_ERASE

using std::vector;
//test your code here

int main() {

#ifdef TEST_CLEAR
    //Vector                    Array
    //Default: 0                Default: 0
    //Create: 0                 Create: 0
    //Copy: 0                   Copy: 0
    //Assign: 0                 Assign: 0
    //Destruct: 2               Destruct: 2
    //Move Construct: 0         Move Construct: 0
    //Move Assign: 0            Move Assign: 0

    {
        std::cout << "Vector" << std::endl;
        simple_string a("Goober");
        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(a);
        simple_string::initialize_counts();
        vec.clear();
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.clear();
        simple_string::print_counts();
    }
#endif

#ifdef TEST_POP_FRONT
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 0               Copy: 0
    //Assign: 0             Assign: 0
    //Destruct: 1           Destruct: 1
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 2        Move Assign: 2

    {
        simple_string a("Goober");
        simple_string b("Gabber");
        simple_string c("Gupper");

        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(b);
        vec.push_back(c);
        simple_string::initialize_counts();
        //note: std::vec does not have pop_front
        vec.erase(vec.begin());
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(b);
        arr.push_back(c);
        simple_string::initialize_counts();
        arr.pop_front();
        simple_string::print_counts();
    }

#endif

#ifdef TEST_POP_BACK
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 0               Copy: 0
    //Assign: 0             Assign: 0
    //Destruct: 1           Destruct: 1
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0
    {
        simple_string a("Goober");


        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.push_back(a);
        simple_string::initialize_counts();
        vec.pop_back();
        simple_string::print_counts();


        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.pop_back();
        simple_string::print_counts();
    }
#endif

#ifdef TEST_PUSH_FRONT_WITH_RESERVE
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 0           Destruct: 0
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0

    {
        simple_string a("Goober");

        simple_string::initialize_counts();
        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.reserve(2);
        vec.insert(vec.begin(), a);
        simple_string::print_counts();

        simple_string::initialize_counts();
        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.reserve(2);
        arr.push_front(a);
        simple_string::print_counts();
    }
#endif

#ifdef TEST_PUSH_FRONT_VEC
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 2           Destruct: 2
    //Move Construct: 2     Move Construct: 2
    //Move Assign: 0        Move Assign: 0

    {
        simple_string a;
        simple_string b("Foob");
        std::cout << "Vector" << std::endl;

        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(a);
        simple_string::initialize_counts();
        //note std::vector doesn't have a push_front
        vec.insert(vec.begin(), a);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.push_front(b);
        std::cout << arr.front() << std::endl;
        simple_string::print_counts();
    }

#endif

#ifdef TEST_PUSH_BACK_NEW_VEC

    //Push back new vec with no reserve
    //
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 0           Destruct: 0
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0
    {
        simple_string a;

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec;
        vec.push_back(a);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr;
        arr.push_back(a);
        simple_string::print_counts();
    }
#endif

#ifdef TEST_INITIALIZER_LIST

    //Test initializer list
    //
    //Vector                  Array
    //Default: 0              Default: 0
    //Create: 0               Create: 0
    //Copy: 4                 Copy: 4
    //Assign: 0               Assign: 0
    //Destruct: 2             Destruct: 2
    //Move Construct: 0       Move Construct: 0
    //Move Assign: 0          Move Assign: 0

    {
        simple_string a;
        simple_string b;
        simple_string c;

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec({ a, b });
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr({ a, b });
        simple_string::print_counts();
    }
#endif

#ifdef TEST_COPY_CONSTRUCTOR
    {
        simple_string a("Goober");
        simple_string b("Gabber");
        simple_string c("Gupper");

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec({ a, b, c });
        vector<simple_string> vec1(vec);
        for (auto i = vec1.begin(); i != vec1.end(); ++i) {
            cout << *i << endl;
        }
        simple_string::print_counts();


        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr({ a, b, c });
        array<simple_string> arr1(arr);
        for (size_t i = 0; i < arr1.length(); ++i) {
            std::cout << arr1[i] << std::endl;
        }
        for (auto i = arr.begin(); i != arr.end(); ++i) {
            cout << *i << endl;
        }
        simple_string::print_counts();
    }
#endif // TEST_COPY_CONSTRUCTOR

#ifdef TEST_ITERATOR
    {
        simple_string a("Goober");
        simple_string b("Gabber");
        simple_string c("Gupper");
        simple_string::initialize_counts();
        array<simple_string> arr({ a, b, c });
        for (size_t i = 0; i < arr.length(); ++i)
        {
            std::cout << arr[i] << std::endl;
        }
        for (auto i = arr.begin(); i != arr.end(); ++i) {
            cout << *i << endl;
        }
        auto iter = arr.begin();
        cout << *iter << '\n';
        cout << *(iter++) << '\n';
        iter = arr.begin();
        cout << *(++iter) << '\n';
        cout << *iter << endl;
    }

#endif // TEST_ITERATOR

#ifdef TEST_ERASE1
    {
        simple_string a("Goober");
        simple_string b("Gabber");
        simple_string c("Gupper");

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec({ a, b, c, c, b, a });
        vector<simple_string> vec1(std::move(vec));
        auto iter_vec = vec1.begin();
        iter_vec++;
        vec1.erase(iter_vec);
        for (auto i = vec1.begin(); i != vec1.end(); ++i) {
            cout << *i << endl;
        }
        iter_vec = vec1.begin();
        iter_vec += 2;
        vec1.erase(iter_vec);
        for (auto i = vec1.begin(); i != vec1.end(); ++i) {
            cout << *i << endl;
        }
        simple_string::print_counts();


        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr({ a, b, c, c, b, a });
        array<simple_string> arr1(std::move(arr));
        //array<simple_string> arr1 = arr;
        auto iter_arr = arr1.begin();
        iter_arr++;
        arr1.erase(iter_arr);
        for (auto i = arr1.begin(); i != arr1.end(); ++i) {
            cout << *i << endl;
        }
        iter_arr = arr1.begin();
        iter_arr++;
        iter_arr++;
        arr1.erase(iter_arr);
        for (auto i = arr1.begin(); i != arr1.end(); ++i) {
            cout << *i << endl;
        }
        simple_string::print_counts();
    }
#endif // TEST_ERASE

#ifdef TEST_INSERT
    {
        simple_string a("Goober");
        simple_string b("Gabber");
        simple_string c("Gupper");

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec({ a, b, c, c });
        auto iter_vec(vec.begin());
        iter_vec += 2;
        vec.insert(iter_vec, a);
        iter_vec = vec.begin();
        iter_vec += 2;
        vec.insert(iter_vec, b);
        for (auto i = vec.begin(); i != vec.end(); ++i) {
            cout << *i << endl;
        }
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr({ a, b, c, c });
        auto iter_arr = arr.begin();
        iter_arr++;
        iter_arr++;
        arr.insert(a, iter_arr);
        iter_arr = arr.begin();
        iter_arr++;
        iter_arr++;
        arr.insert(b, iter_arr);
        for (auto i = arr.begin(); i != arr.end(); ++i) {
            cout << *i << endl;
        }
        simple_string::print_counts();
    }
#endif // TEST_INSERT

#ifdef TEST_INSERT1
    //Vector                    Array
    //Default: 0                Default: 0
    //Create: 0                 Create: 0
    //Copy: 1                   Copy: 1
    //Assign: 0                 Assign: 0
    //Destruct: 2               Destruct: 2
    //Move Construct: 2         Move Construct: 2
    //Move Assign: 0            Move Assign: 0

    {
        std::cout << "Vector" << std::endl;
        simple_string a("Goober");
        simple_string b("Uber");
        simple_string c("Amazon");
        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(b);
        simple_string::initialize_counts();
        vec.insert(vec.begin(), c);
        vec.insert(vec.begin(), b);
        for (int i = 0; i < vec.size(); ++i)
        {
            std::cout << vec.at(i) << std::endl;
        }
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(b);
        simple_string::initialize_counts();
        arr.insert(c, arr.begin());
        arr.insert(b, arr.begin());
        for (size_t i = 0; i < arr.length(); ++i)
        {
            std::cout << arr[i] << std::endl;
        }
        simple_string::print_counts();
    }
#endif

#ifdef TEST_ERASE
    //Vector                    Array
    //Default: 0                Default: 0
    //Create: 0                 Create: 0
    //Copy: 0                   Copy: 0
    //Assign: 0                 Assign: 0
    //Destruct: 2               Destruct: 2
    //Move Construct: 0         Move Construct: 0
    //Move Assign: 2            Move Assign: 2

    {
        std::cout << "Vector" << std::endl;
        simple_string a("Goober");
        simple_string b("Uber");
        simple_string c("Amazon");
        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(b);
        vec.push_back(c);
        simple_string::initialize_counts();
        vec.erase(vec.begin());
        vec.erase(vec.begin()+1);
        for (int i = 0; i < vec.size(); ++i)
        {
            std::cout << vec.at(i) << std::endl;
        }
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(b);
        arr.push_back(c);
        simple_string::initialize_counts();
        arr.erase(arr.begin());
        arr.erase(++arr.begin());
        for (size_t i = 0; i < arr.length(); ++i)
        {
            std::cout << arr[i] << std::endl;
        }
        simple_string::print_counts();
    }
#endif
    return 0;
}
