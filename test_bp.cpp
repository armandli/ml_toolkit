#include <matrix.h>
#include <ml_util.h>
#include <ml_supervised.h>
#include <test_problem.h>
using namespace std;

int main(){
  string filename = "spiral_dataset.txt";
  Mtx<double> TrainX, TrainY, TestX, TestY;
  read_input(filename, TrainX, TrainY, TestX, TestY);

  Backpropagation<ReluFun> model(TrainX.cols(), TrainY.cols(), 32);
  double training_accuracy = model.train(TrainX, TrainY, 2000, 0.1, 0.001, 10);
  double test_accuracy = model.test(TestX, TestY);

  cout << "Training Accuracy: " << training_accuracy << " Test Accuracy: " << test_accuracy << endl;
}
