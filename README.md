# Multi-GPU Framework for TensorFlow

Using multi-GPU framework you can run any normal tensorflow code on a multiple specified number of GPUs. The framework creates same copy of the tensorflow graph for each gpu and run parallely. The framework does not require any major change in the design of the tf graph. You can write your tensorflow code any way you want and use framework method call to enable it on multiple GPUs.

## Getting Started

This project implements LSTM and CNN using the Multi-GPU framework on MNIST data.

### Prerequisites

To run this project you will need following packages:
1. TensforFlow
2. NumPy

## Authors

* **Manish Vidyasagar**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
