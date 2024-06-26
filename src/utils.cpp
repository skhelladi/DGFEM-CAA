#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "utils.h"

extern "C"
{
    // LU decomoposition of a general matrix
    void dgetrf_(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);

    // Generate inverse of a matrix given its LU decomposition
    void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork, int *INFO);

    // Computes the solution to a real system of linear equations
    void dgesv_(int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO);

    // Returns the value of the norm
    double dlange_(char *NORM, int *M, int *N, double *A, int *LDA, double *WORK);

    // Dot product
    double ddot_(int *N, double *DX, int *INCX, double *DY, int *INCY);

    // Matrix/vector product:  y := alpha*A*x + beta*y,
    void dgemv_(char &TRANS, int &M, int &N, double &a, double *A,
                int &LDA, double *X, int &INCX, double &beta, double *Y, int &INCY);
}

//! added by Sofiane KHELLADI in 11/03/2022 /////////////////////

std::string fileExtension(std::string file)
{

    std::size_t found = file.find_last_of(".");
    return file.substr(found + 1);
}

namespace screen_display
{
    void write_string(std::string text, std::string color)
    {
#ifdef _WIN64
        std::cout << text << std::endl
                  << std::flush;
//#endif
#else
        //#ifdef _linux_
        std::cout << color << text << RESET << std::endl
                  << std::flush;
#endif // _linux_
    }

    void write_value(std::string name, double value, std::string unit, std::string color)
    {
#ifdef _WIN64
        if (fabs(value) >= 1.0e-3)
        {
            std::cout << std::setw(50) << std::left << name << std::right << std::fixed << std::setprecision(6) << std::setw(30) << value << std::right << " [" << unit << "]" << std::endl
                      << std::flush;
        }
        else
        {
            std::cout << std::setw(50) << std::left << name << std::right << std::scientific << std::setw(30) << value << std::right << " [" << unit << "]" << std::endl
                      << std::flush;
        }
//#endif
#else
        //#ifdef _linux_
        if (fabs(value) >= 1.0e-3)
        {
            std::cout << RESET << std::setw(50) << std::left << name << std::right << color << std::fixed << std::setprecision(6) << std::setw(30) << value << RESET << std::right << " [" << unit << "]" << std::endl
                      << std::flush;
        }
        else
        {
            std::cout << RESET << std::setw(50) << std::left << name << std::right << color << std::scientific << std::setw(30) << value << RESET << std::right << " [" << unit << "]" << std::endl
                      << std::flush;
        }
#endif
    }

    void write_value_r(std::string name, double value, std::string unit, std::string color, size_t precision)
    {
#ifdef _WIN64
        if (fabs(value) >= 1.0e-3)
        {
            std::cout << "\r" << std::setw(50) << std::left << name << std::right << std::fixed << std::setprecision(precision) << std::setw(30) << value << std::right << " [" << unit << "]";
        }
        else
        {
            std::cout << "\r" << std::setw(50) << std::left << name << std::right << std::scientific << std::setw(30) << value << std::right << " [" << unit << "]";
        }
//#endif
#else
        //#ifdef _linux_
        if (fabs(value) >= 1.0e-3)
        {
            std::cout << "\r" << RESET << std::setw(50) << std::left << name << std::right << color << std::fixed << std::setprecision(precision) << std::setw(30) << value << RESET << std::right << " [" << unit << "]";
        }
        else
        {
            std::cout << "\r" << RESET << std::setw(50) << std::left << name << std::right << color << std::scientific << std::setw(30) << value << RESET << std::right << " [" << unit << "]";
        }
#endif
    }
    bool write_if_false(const bool assertion, const char *msg)
    {
        if (!assertion)
        {
            // endl to flush
            write_string(msg, RED);
        }
        return assertion;
    }

    // void write_vector_to_file(std::string file_name, std::vector<auto> vec, size_t offset)
    // {
    //     std::ofstream outfile(file_name.c_str());
    //     write_value("writing value in "+file_name+" - vector size",vec.size()/offset,"",BLUE);
    //     for(size_t i=0;i<vec.size();i+=offset)
    //     {
    //        for(size_t j=0;j<offset;j++)
    //             outfile<<vec[i+j]<<"\t";
    //        outfile<<std::endl;
    //     }

    //     outfile.close();
    // }
}
///////////////////////
namespace io
{
    /**
     * Reads csv file into table, exported as a vector of vector of doubles.
     * @param inputFileName input file name (full path).
     * @return data as vector of vector of doubles.
     */
    std::vector<std::vector<double>> parseCSVFile(std::string inputFileName, char separator)
    {
        std::vector<std::vector<double>> data;
        std::ifstream inputFile(inputFileName);
        int l = 0;

        while (inputFile)
        {
            l++;
            std::string s;
            if (!getline(inputFile, s))
                break;

            if (s[0] != '#' && l != 1)
            {
                std::istringstream ss(s);
                std::vector<double> record;

                while (ss)
                {
                    std::string line;
                    if (!getline(ss, line, separator))
                        break;

                    double value = stof(line);
                    record.push_back(value);
                }

                data.push_back(record);
            }
        }

        if (!inputFile.eof())
        {
            std::cerr << "Could not read file " << inputFileName << "\n";
            std::__throw_invalid_argument("File not found.");
        }

        return data;
    }

    void writeWave(std::vector<float> V, std::string filename, uint32_t sample_rate, uint16_t bits_per_sample, uint16_t channel_number, size_t nb_sequence)
    {
        screen_display::write_string("Write WAVE file: '" + filename + "'");

        wave::File write_file;
        // uint32_t sample_rate = 10000;//V.size();
        // uint16_t bits_per_sample = 16;
        // uint16_t channel_number = 1;
        // filename += ".wav";

        std::vector<float> content;

        for (size_t i = 0; i < nb_sequence; i++)
            for (auto v : V)
            {
                content.push_back(v);
            }

        wave::Error err = write_file.Open(filename, wave::kOut);
        if (err)
        {
            Fatal_Error("Something went wrong in out open")
        }
        write_file.set_sample_rate(sample_rate);
        write_file.set_bits_per_sample(bits_per_sample);
        write_file.set_channel_number(channel_number);

        err = write_file.Write(content);
        if (err)
        {
            Fatal_Error("Something went wrong in write")
        }
    }

    void readWave(std::string filename, std::vector<float> &V, uint32_t &sample_rate)
    {
       
        screen_display::write_string("Read WAVE source file: '" + filename + "'");
        wave::File read_file;
        wave::Error err = read_file.Open(filename, wave::kIn);
        if (err)
        {
            Fatal_Error("Something went wrong in out open")
        }

        // ASSERT_EQ(read_file.sample_rate(), 44100);
        // ASSERT_EQ(read_file.bits_per_sample(), 16);
        // ASSERT_EQ(read_file.channel_number(), 2);
        sample_rate = read_file.sample_rate();
        err = read_file.Read(&V);
        if (err)
        {
            Fatal_Error("Something went wrong in read")
        }

        std::cout << "source file=" << filename << std::endl;
        std::cout << "sample_rate=" << read_file.sample_rate() << std::endl;
        std::cout << "bits_per_sample=" << read_file.bits_per_sample() << std::endl;
        std::cout << "channel_number=" << read_file.channel_number() << std::endl;
    }

    std::vector<std::vector<double>> parseWAVEFile(std::string inputFileName)
    {
        std::vector<std::vector<double>> data;
        uint32_t sample_rate;
        std::vector<float> V;

        readWave(inputFileName, V, sample_rate);
        double timeStep = 1.0 / sample_rate;
        double time(0);
        for (auto v : V)
        {
            data.push_back({time, v});
            time += timeStep;
        }

        // writeWave(V,"data/data_write.wav",sample_rate,16,1,1);

        return data;
    }

    void writeFFT(std::vector<float> V, double timeStep, std::string filename)
    {
        screen_display::write_string("Write FFT files: '" + filename + "'");
        size_t nb_observer_time = getNearestLowerPowerOf2(V.size());
        Eigen::MatrixXd pt(nb_observer_time, 2);

        double sum = 0.0;

        for (size_t i = 0; i < nb_observer_time; i++)
        {
            pt(i, 0) = i * timeStep;
            double p = V[i];
            pt(i, 1) = p;
            sum += pt(i, 1);
        }
        //! mean suppression /////////////

        sum /= pt.rows();
        pt.col(1) = pt.col(1).array() - sum;
        double I_p2 = 0.0;
        double dt;
        for (size_t i = 1; i < nb_observer_time; i++)
        {
            dt = timeStep;
            I_p2 += 0.5 * (pow(pt(i - 1, 1), 2) + pow(pt(i, 1), 2)) * dt;
        }
        //! mean suppression /////////////
        double T = timeStep * V.size();
        double p2_eff = (1.0 / T * I_p2) / nb_observer_time;
        double p0 = 2.0e-5;
        double SPL = 10.0 * log10(p2_eff / pow(p0, 2));

        std::string fft_filename = filename + "_fft.txt";
        std::string psd_filename = filename + "_psd.txt";
        WriteFFT(pt, fft_filename);
        WritePSD(pt, psd_filename);
    }
}

// Lapack with direct calling to fortran interface.
// => Migrate to Lapacke (conflict with Gmsh, segmentation fault)
namespace lapack
{
    // Compute the inverse of a square matrix A of size N*N.
    // Input can either be a array or a vector. Since c++11
    // vectors are also stored as a contiguous memory chunk.
    // Lapack assumes column major matrix, however as the
    // inverse of a transposed is the transposed of an inverse.
    // The matrix can either be column or row major.
    void inverse(double *A, int &N)
    {
        int IPIV[N];
        int LWORK = N * N;
        double WORK[LWORK];
        int INFO;
        dgetrf_(&N, &N, A, &N, IPIV, &INFO);
        dgetri_(&N, A, &N, IPIV, WORK, &LWORK, &INFO);
    }

    // Computes the solution to a real system of linear equations
    //         A * X = B,
    // where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
    // On exit, the solution X is stored in B.
    void solve(double *A, double *B, int &N)
    {
        int INFO;
        int IPIV[N];
        int NRHS = 1;
        dgesv_(&N, &NRHS, A, &N, IPIV, B, &N, &INFO);
    }

    // Normalize a vector/matrix with respect to Frobenius norm
    void normalize(double *A, int &N)
    {
        char NORM = 'F';
        int LDA = 1;
        double WORK[N];
        double norm;
        norm = dlange_(&NORM, &LDA, &N, A, &LDA, WORK);
        for (int i = 0; i < N; i++)
        {
            A[i] /= norm;
        };
    }

    // Matrix/vector product:  y := alpha*A*x + beta*y
    void linEq(double *A, double *X, double *Y, double &alpha, double beta, int &N)
    {
        char TRANS = 'T';
        int INC = 1;
        dgemv_(TRANS, N, N, alpha, A, N, X, INC, beta, Y, INC);
    }

    // Dot product between 2 vectors
    double dot(double *A, double *B, int N)
    {
        int INCX = 1;
        return ddot_(&N, A, &INCX, B, &INCX);
    }

    void minus(double *A, double *B, int N)
    {
        std::transform(A, A + N, B, A, std::minus<double>());
    }

    void plus(double *A, double *B, int N)
    {
        std::transform(A, A + N, B, A, std::plus<double>());
    }

    void plusTimes(double *A, double *B, double c, int N)
    {
        for (int i = 0; i < N; ++i)
            A[i] += B[i] * c;
    }
}

namespace eigen
{

    // Computes the solution to a real system of linear equations
    //         A * X = B,
    // where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
    // On exit, the solution X is stored in B.
    void solve(double *A, double *B, int &N)
    {
        Eigen::Map<Eigen::MatrixXd> A_eigen(A, N, N);
        Eigen::Map<Eigen::VectorXd> B_eigen(B, N);
        B_eigen = A_eigen.lu().solve(B_eigen);
    }

    void inverse(double *A, int &N)
    {
        Eigen::Map<Eigen::MatrixXd> A_eigen(A, N, N);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_eigen);
        A_eigen = A_eigen.inverse();
    }

    // Normalize a vector/matrix with respect to Frobenius norm
    void normalize(double *A, int &N)
    {
        Eigen::Map<Eigen::VectorXd> A_eigen(A, N);
        A_eigen.normalize();
    }

    // Matrix/vector product:  y := alpha*A*x + beta*y
    void linEq(double *A, double *X, double *Y, double &alpha, double beta, int &N)
    {
        Eigen::Map<Eigen::VectorXd> X_eigen(X, N);
        Eigen::Map<Eigen::VectorXd> Y_eigen(Y, N);
        Eigen::Map<Eigen::MatrixXd> A_eigen(A, N, N);
        Y_eigen = beta * Y_eigen + alpha * A_eigen * X_eigen;
    }

    // Dot product between 2 vectors
    double dot(double *A, double *B, int N)
    {
        Eigen::Map<Eigen::VectorXd> A_eigen(A, N);
        Eigen::Map<Eigen::VectorXd> B_eigen(B, N);
        return A_eigen.dot(B_eigen);
    }

    void minus(double *A, double *B, int N)
    {
        Eigen::Map<Eigen::VectorXd> A_eigen(A, N);
        Eigen::Map<Eigen::VectorXd> B_eigen(B, N);
        A_eigen -= B_eigen;
    }

    void plus(double *A, double *B, int N)
    {
        Eigen::Map<Eigen::VectorXd> A_eigen(A, N);
        Eigen::Map<Eigen::VectorXd> B_eigen(B, N);
        A_eigen += B_eigen;
    }

    void plusTimes(double *A, double *B, double c, int N)
    {
        Eigen::Map<Eigen::VectorXd> A_eigen(A, N);
        Eigen::Map<Eigen::VectorXd> B_eigen(B, N);
        A_eigen += c * B_eigen;
    }

    void cross(double *A, double *B, double *OUT)
    {
        Eigen::Map<Eigen::Vector3d> A_eigen(A);
        Eigen::Map<Eigen::Vector3d> B_eigen(B);
        Eigen::Map<Eigen::Vector3d> OUT_eigen(OUT);
        OUT_eigen = A_eigen.cross(B_eigen);
    }
}
