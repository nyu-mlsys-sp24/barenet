#include <getopt.h>

#include "modules/mlp.cuh"
#include "modules/linear.cuh"
#include "modules/sgd.cuh"

#include "utils/dataset_mnist.hh"
#include "ops/op_elemwise.cuh"
#include "ops/op_cross_entropy.cuh"

unsigned long long randgen_seed = 0;

int correct(const Tensor<float> &logits, const Tensor<char> &targets) 
{
    assert(targets.w == 1);
    Tensor<int> predictions{targets.h, targets.w};
    op_argmax(logits, predictions);
    Tensor<int> correct_preds{targets.h, targets.w};
    op_equal(predictions, targets, correct_preds);
    Tensor<int> sum_correct{1,1};
    op_sum(correct_preds, sum_correct);
    return Index(sum_correct, 0, 0);
}

void train(int epochs, int batch_size, int hidden_dim, int n_layers, bool on_gpu)
{
    MNIST mnist_train{"../data/MNIST/raw", MNIST::Mode::kTrain};
    MNIST mnist_test{"../data/MNIST/raw", MNIST::Mode::kTest};
    
    std::cout << "# of data points=" << mnist_train.images.h << " feature size=" << mnist_train.images.w << std::endl;

    std::vector<int> layer_dims;
    for (int i = 0; i < n_layers - 1; i++)
    {
        layer_dims.push_back(hidden_dim);
    }
    layer_dims.push_back(10); // last layer's out dimension is always 10 (# of digits)

    MLP<float> mlp{batch_size, MNIST::kImageRows * MNIST::kImageColumns, layer_dims, false};
    mlp.init();
    SGD<float> sgd{mlp.parameters(), 0.01};

    Tensor<float> logits{batch_size, 10};
    Tensor<float> d_logits{batch_size, 10};
    Tensor<float> d_input_images{batch_size, mnist_train.images.w};
    float loss;
    int num_batches, total_correct;
    float total_loss;
    for (int i = 0; i < epochs; i++)
    {
        num_batches = 0;
        total_loss = 0.0;
        total_correct = 0;
        for (int b = 0; b < mnist_train.images.h / batch_size; b++)
        {
            if ((b + 1) * batch_size >= mnist_train.images.h)
            {
                break;
            }
            num_batches++;
            Tensor<float> input_images = mnist_train.images.slice(b * batch_size, (b + 1) * batch_size, 0, mnist_train.images.w);
            Tensor<float> debug_t = input_images.slice(0, 1, 0, mnist_train.images.w);
            Tensor<char> targets = mnist_train.targets.slice(b * batch_size, (b+1)*batch_size, 0, mnist_train.targets.w);
            mlp.forward(input_images, logits);

            loss = op_cross_entropy_loss(logits, targets, d_logits);
            total_loss += loss;
            total_correct += correct(logits, targets);
            
            mlp.backward(input_images, d_logits, d_input_images);
            sgd.step();
        }
        std::cout << "epoch=" << i << " loss=" << total_loss/num_batches 
            << " accuracy=" << total_correct/(float)(num_batches*batch_size)
            << " num_batches=" << num_batches 
            << std::endl;

    }
}

int main(int argc, char *argv[])
{
    bool on_gpu = false;
    int hidden_dim = 16;
    int n_layers = 2;
    int batch_size = 32;
    int num_epochs = 10;

    for (;;)
    {
        switch (getopt(argc, argv, "s:g:h:l:b:e:"))
        {
        case 's':
            randgen_seed = atoll(optarg);
            continue;
        case 'g':
            on_gpu = true;
            continue;
        case 'h':
            hidden_dim = atoi(optarg);
            continue;
        case 'l':
            n_layers = atoi(optarg);
            continue;
        case 'b':
            batch_size = atoi(optarg);
            continue;
        case 'e':
            num_epochs = atoi(optarg);
            continue;
        case -1:
            break;
        }
        break;
    }
    train(num_epochs, batch_size, hidden_dim, n_layers, on_gpu);
}
