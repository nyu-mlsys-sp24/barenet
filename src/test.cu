#include <getopt.h>

#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"

unsigned long long randgen_seed = 0;

void test_matmul(int m, int n, int k, bool on_gpu)
{
    Tensor<float> A{m, k, on_gpu};
    op_uniform_init(A);

    Tensor<float> B{k, n, on_gpu};
    op_uniform_init(B);

    Tensor<float> C{m, n, on_gpu};
    op_mm(A, B, C);

    Tensor<float> C2{n, m, on_gpu};
    op_mm(B.transpose(), A.transpose(), C2);
    assert(op_allclose(C2.transpose(), C)); // test transpose

    std::cout << "matmul passed..." << std::endl;
}

void test_elemwise(int m, int n, bool on_gpu)
{
    
    Tensor<float> X{m, n, on_gpu};
    op_const_init(X, 2.0);

    Tensor<float> Y{m, n, on_gpu};
    op_const_init(Y, 3.0);

    Tensor<float> Z{m, n, on_gpu};
    op_const_init(Z, 10.0);
    op_add(X, Y, Z);

    Tensor<float> Zref{m, n, on_gpu};
    op_const_init(Zref, 5.0);
    assert(op_allclose(Z, Zref));

    Tensor<float> Y2{1, n, on_gpu};
    op_const_init(Y2, 3.0);
    op_add(X, Y2, Z); //test broadcasting
    assert(op_allclose(Z, Zref));

    op_add<float>(X, 3.0, Z);
    assert(op_allclose(Z, Zref));

    std::cout << "op_add passed..." << std::endl;

    op_multiply(X, Y, Z);

    op_const_init(Zref, 6.0);
    assert(op_allclose(Z, Zref));

    op_multiply(X, Y2, Z);
    assert(op_allclose(Z, Zref));

    op_multiply<float>(X, 3.0, Z);
    assert(op_allclose(Z, Zref));

    std::cout << "op_multiply passed..." << std::endl;

    float lr = 0.02;
    Tensor<float> A{m, n, on_gpu};
    op_uniform_init(A);
    Tensor<float> A_host = A.toHost();

    Tensor<float> dA{m, n, on_gpu};
    op_uniform_init(dA);
    Tensor<float> dA_host = dA.toHost();

    Tensor<float> Aref{m, n, false};
    for (int i = 0; i < Aref.h; i++)
    {
        for (int j = 0; j < Aref.w; j++)
        {
          Index(Aref, i, j) = Index(A_host, i, j) - lr * Index(dA_host, i, j);
        }
    }
    op_sgd(A, dA, A, lr);
    assert(op_allclose(A, Aref));

    std::cout << "op_sgd passed..." << std::endl;

}

void 
test_reduction(int m, int n, bool on_gpu)
{
    Tensor<int> X_host{m, n};
    op_const_init(X_host, 0);

    int reduce_sum = m>n?n:m;
    for (int i = 0; i < X_host.h; i++) 
    {
        if (i >= X_host.w) {
            break;
        }
        Index(X_host, i, i) = 1;
    }

    Tensor<int> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 
    Tensor<int> Y{1,n, on_gpu};
    op_sum(X,Y);

    Tensor<int> Yref{1,n};
    op_const_init(Yref,0);
    for (int j=0; j< reduce_sum; j++) {
        Index(Yref, 0, j) = 1;
    }
    Tensor<int> Y_host = Y.toHost();
    assert(op_allclose(Y, Yref));

    Tensor<int> Y1{1,1, on_gpu};
    op_sum(Y, Y1);
    Tensor<int> Y1_host = Y1.toHost();
    assert(Index(Y1_host,0,0) == reduce_sum);

    op_const_init(X, 1);
    op_sum(X, Y);
    op_const_init(Yref, X.h);
    assert(op_allclose(Y, Yref));
    
    Tensor<int> YY{m, 1, on_gpu};
    op_sum(X, YY);
    Tensor<int> YYref{m, 1};
    op_const_init(YYref, n);
    op_sum(YY, Y1);
    Y1_host = Y1.toHost();
    assert(Index(Y1_host, 0, 0) == m*n);

    std::cout << "op_sum passed..." << std::endl;

    //try to create an A matrix whose last column has the biggest value
    Tensor<float> A{m, n, on_gpu};
    op_uniform_init<float>(A, 0.0, 1.0);
    auto AA = A.slice(0, A.h, A.w-1, A.w);
    op_add<float>(AA, 10.0, AA);

    Tensor<int> ind{m, 1, on_gpu};
    op_argmax(A, ind);
    Tensor<int> indref{m, 1};
    op_const_init(indref, n-1);
    assert(op_allclose(ind, indref));
}

void 
test_views()
{
    Tensor<float> A{5, 5};
    for (int i = 0; i < A.h; i++) {
        for (int j = 0; j < A.w; j++) {
            Index(A, i, j) = i*A.w+j;
        }
    }
    auto B = A.slice(1,3,1,3);
    assert(Index(B, 0, 0) == 6);
    assert(Index(B, 0, 1) == 7);
    assert(Index(B, 1, 0) == 11);
    assert(Index(B, 1, 1) == 12);
    auto C = B.transpose();
    assert(Index(C, 0, 0) == 6);
    assert(Index(C, 0, 1) == 11);
    assert(Index(C, 1, 0) == 7);
    assert(Index(C, 1, 1) == 12);
    std::cout << "slice passed..." << std::endl;
}

int main(int argc, char *argv[])
{
    bool test_gpu = true;
    int test_m = 335, test_n = 587, test_k= 699;

    for (;;)
    {
        switch (getopt(argc, argv, "s:ch:l:b:e:"))
        {
        case 's':
            randgen_seed = atoll(optarg);
            continue;
        case 'c': //cpu testing only
            test_gpu = false;
            continue;
        case 'm':
            test_m = atoi(optarg);
            continue;
        case 'n':
            test_n = atoi(optarg);
            continue;
        case 'k':
            test_k = atoi(optarg);
            continue;
        case -1:
            break;
        }
        break;
    }
    test_views();
    test_elemwise(test_m, test_n, test_gpu);
    test_matmul(test_m, test_n, test_k, test_gpu);
    test_reduction(test_m, test_n, test_gpu);
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}
