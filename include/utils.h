#ifndef DGALERKIN_UTILS_H
#define DGALERKIN_UTILS_H

#include <iomanip>
#include <iostream>

//! added by Sofiane KHELLADI in 01/04/2022 /////////////////////
#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */
#define CLEAR "\033[2J"               // clear screen escape code

#define TRI 2
#define TETRA 4

#define VTK_TRI 5
#define VTK_TETRA 10

//////////////////////////////////////////////////////////////
// using namespace std;

namespace screen_display
{
    void write_string(std::string text, std::string color = YELLOW);

    void write_value(std::string name, double value, std::string unit, std::string color);
    
    void write_value_r(std::string name, double value, std::string unit, std::string color, size_t precision);
}
/////////////////////////

namespace lapack
{

    void inverse(double *A, int &N);

    void solve(double *A, double *B, int &N);

    void normalize(double *A, int &N);

    double dot(double *A, double *b, int N);

    void linEq(double *A, double *X, double *Y, double &alpha, double beta, int &N);

    void minus(double *A, double *B, int N);

    void plus(double *A, double *B, int N);

    void plusTimes(double *A, double *B, double c, int N);
}

namespace eigen
{

    void inverse(double *A, int &N);

    void solve(double *A, double *B, int &N);

    void normalize(double *A, int &N);

    double dot(double *A, double *b, int N);

    void linEq(double *A, double *X, double *Y, double &alpha, double beta, int &N);

    void minus(double *A, double *B, int N);

    void plus(double *A, double *B, int N);

    void plusTimes(double *A, double *B, double c, int N);

    void cross(double *A, double *B, double *OUT);
}

namespace display
{
    template <typename Container>
    void print(const Container &cont, int row = 1, bool colMajor = false)
    {

        if (colMajor)
        {
            for (int rowIt = 0; rowIt < row; ++rowIt)
            {
                int colIt = 0;
                for (auto const &x : cont)
                {
                    if (colIt % row == rowIt)
                    {
                        std::cout << std::setprecision(4) << std::left << std::setw(10) << x << " ";
                    }
                    colIt++;
                }
                std::cout << std::endl;
            }
        }
        else
        {
            int colIt = 0;
            for (auto const &x : cont)
            {
                std::cout << std::setprecision(4) << std::left << std::setw(10) << x << " ";
                colIt++;
                if (colIt % row == 0)
                    std::cout << std::endl;
            }
        }
    }
}

#endif // DGALERKIN_UTILS_H
