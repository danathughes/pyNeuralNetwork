#include <Eigen/Core>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

class Autoencoder
{
  private:
  public:
    unsigned int numInputs;
    unsigned int numHidden;
    Eigen::MatrixXd W_ih;
    Eigen::MatrixXd W_ho;
    Eigen::VectorXd bias_h;
    Eigen::VectorXd bias_o;

    Autoencoder(unsigned int, unsigned int);
    ~Autoencoder();

    Eigen::VectorXd* activate(Eigen::VectorXd);
    Eigen::VectorXd* backpropagation(Eigen::VectorXd*, Eigen::VectorXd);
    Eigen::VectorXd* backpropagation(Eigen::VectorXd*, Eigen::VectorXd, Eigen::VectorXd, double, double);

    Eigen::VectorXd predict(Eigen::VectorXd);
    Eigen::MatrixXd* getWeights();
    void updateWeights(Eigen::MatrixXd*);
    double cost(Eigen::VectorXd*, int);
    Eigen::MatrixXd* gradient(Eigen::VectorXd*, int);
    Eigen::MatrixXd* gradient(Eigen::VectorXd*, int, double, double);
    void save(const char*);
   // static Autoencoder* load(const char*);
    void print();
};


double sigmoid(double z)
{
   return 1.0 / (1.0 + exp(-z));
}

double costFunction(Eigen::VectorXd prediction, Eigen::VectorXd target)
{
   return (prediction - target).array().square().sum() / 2.0;
}

Eigen::VectorXd* load_dataset(const char* filename, int numItems)
{
   static Eigen::VectorXd dataset[7766015];

   ifstream data(filename);
   string line;

   int numFeatures = 36;

   for(int i=0; i<numItems; i++)
   {
      getline(data, line);
      stringstream lineStream(line);
      string cell;

      dataset[i] = Eigen::VectorXd(36);

      for(int j=0; j<numFeatures-1; j++)
      {
         getline(lineStream, cell, ',');
         dataset[i](j) = atof(cell.c_str());
      }
      getline(lineStream, cell);
      dataset[i](35) = atof(cell.c_str());
   }


   return dataset;
}

Autoencoder* load_autoencoder(const char*);

int main()
{
   fstream log_file;
   log_file.open("autoencoder_log.txt", fstream::out);

   int numItems = 7766016;
   int numFeatures = 36;
   int numHidden = 40;

   log_file << "Loading dataset...";
   log_file.flush();
   cout << "Loading dataset...";
   Eigen::VectorXd* dataset = load_dataset("patches.csv", numItems);
   cout << "Done!" << endl;
   log_file << "Done!" << endl;
   log_file.flush();

   Autoencoder* autoencoder = new Autoencoder(numFeatures, numHidden);

   double cost = autoencoder->cost(dataset, numItems);
   cout << "Cost (autoencoder): " << cost << endl;
   log_file << "Cost (0): " << cost << endl;
   log_file.flush();


   double lr = 0.9;

   for(int i=0; i<10000; i++)
   {
      Eigen::MatrixXd* grad = autoencoder->gradient(dataset, numItems, 0.15, 0.1);

      grad[0] = -lr*grad[0];
      grad[1] = -lr*grad[1];
      grad[2] = -lr*grad[2];
      grad[3] = -lr*grad[3];

      autoencoder->updateWeights(grad);
      if(i%100 == 0)
      {
        cost = autoencoder->cost(dataset, numItems);
        cout << "Cost (autoencoder) " << i << ": " << cost << endl;
        log_file << "Cost (" << i << "): " << cost << endl;
        log_file.flush();
      }

      if(i%500 == 0)
      {
         ostringstream sstream;
         sstream << "models/autoencoder_" << i << ".model";
         string filename = sstream.str();
         autoencoder->save(filename.c_str());
      }
   }


//   autoencoder->save("test.dat");

   log_file.close();

   delete autoencoder;
   return 0;
}

Autoencoder::Autoencoder(unsigned int numInputs, unsigned int numHidden)
{
   this->numInputs = numInputs;
   this->numHidden = numHidden;

   // Create the weight and bias matrices, and set so that the values are 
   // random, scaled by the fan-in
   this->W_ih = Eigen::MatrixXd::Random(numHidden, numInputs) / numInputs;
   this->W_ho = Eigen::MatrixXd::Random(numInputs, numHidden) / numHidden;
   this->bias_h = Eigen::VectorXd::Random(numHidden) / numHidden;
   this->bias_o = Eigen::VectorXd::Random(numInputs) / numInputs;
}

Autoencoder::~Autoencoder()
{

}

Eigen::VectorXd* Autoencoder::activate(Eigen::VectorXd data)
{
   static Eigen::VectorXd activations[3];

   activations[0] = Eigen::VectorXd(data);

   Eigen::VectorXd hidden_net = this->W_ih * activations[0] + this->bias_h;
   activations[1] = hidden_net.unaryExpr(ptr_fun(sigmoid));

   Eigen::VectorXd output_net = this->W_ho * activations[1] + this->bias_o;
   activations[2] = output_net.unaryExpr(ptr_fun(sigmoid));

   return activations;
}

Eigen::VectorXd Autoencoder::predict(Eigen::VectorXd data)
{
   return this->activate(data)[2];
}


Eigen::MatrixXd* Autoencoder::getWeights()
{
   static Eigen::MatrixXd weights[4];
   
   weights[0] = this->W_ih;
   weights[1] = this->W_ho;
   weights[2] = this->bias_h;
   weights[3] = this->bias_o;

   return weights;
}


void Autoencoder::updateWeights(Eigen::MatrixXd* weights)
{
   this->W_ih += weights[0];
   this->W_ho += weights[1];
   this->bias_h += weights[2];
   this->bias_o += weights[3];
}


double Autoencoder::cost(Eigen::VectorXd* dataset, int numItems)
{
   double cost = 0.0;
   Eigen::VectorXd prediction;

   for(int i=0; i<numItems; i++)
   {
      prediction = this->predict(dataset[i]);
      cost += costFunction(prediction, dataset[i]);
   }

   return cost / numItems;
}

Eigen::VectorXd* Autoencoder::backpropagation(Eigen::VectorXd* activations, Eigen::VectorXd output)
{
   static Eigen::VectorXd deltas[3];

   Eigen::ArrayXd df2 = activations[2].array() * (1.0 - activations[2].array());
   Eigen::ArrayXd df1 = activations[1].array() * (1.0 - activations[1].array());

   deltas[2] = -df2*(output - activations[2]).array();
   deltas[2] = deltas[2].matrix();

   deltas[1] = (this->W_ih*deltas[2]).array()*df1;
   deltas[1] = deltas[1].matrix();

   return deltas;
}


Eigen::VectorXd* Autoencoder::backpropagation(Eigen::VectorXd* activations, Eigen::VectorXd output, Eigen::VectorXd avgActivation, double rho, double beta)
{
   static Eigen::VectorXd deltas[3];

   Eigen::ArrayXd df2 = activations[2].array() * (1.0 - activations[2].array());
   Eigen::ArrayXd df1 = activations[1].array() * (1.0 - activations[1].array());
   
   Eigen::ArrayXd KL = (1.0-rho)/(1.0 - avgActivation.array());
   KL -= rho/avgActivation.array();

   deltas[2] = -df2*(output - activations[2]).array();
   deltas[2] = deltas[2].matrix();

   deltas[1] = (this->W_ih*deltas[2] + beta*KL.matrix()).array()*df1;
   deltas[1] = deltas[1].matrix();

   return deltas;
}


Eigen::MatrixXd* Autoencoder::gradient(Eigen::VectorXd* dataset, int numItems)
{
   return this->gradient(dataset, numItems, 0, 0);
}

Eigen::MatrixXd* Autoencoder::gradient(Eigen::VectorXd* dataset, int numItems, double rho, double beta)
{
   static Eigen::MatrixXd gradient[4];
   Eigen::VectorXd data;
   Eigen::VectorXd* activations;
   Eigen::VectorXd* deltas;

   // Initialize the gradients to zero
   Eigen::MatrixXd dW_ih = Eigen::MatrixXd::Zero(this->numHidden, this->numInputs);
   Eigen::MatrixXd dW_ho = Eigen::MatrixXd::Zero(this->numInputs, this->numHidden);
   Eigen::VectorXd dbias_h = Eigen::VectorXd::Zero(this->numHidden);
   Eigen::VectorXd dbias_o = Eigen::VectorXd::Zero(this->numInputs);

   // Calculate the average activation of the hidden layer
   Eigen::VectorXd avgActivations = Eigen::VectorXd::Zero(this->numHidden);

   if(beta >= 0)
   {   for(int i=0; i<numItems; i++)
      {
         data = dataset[i];
         activations = this->activate(data);
         avgActivations += activations[1];
      }
      avgActivations /= numItems;
   }

   // Update the gradient for each data point
   for(int i=0; i<numItems; i++)
   {
      data = dataset[i];

      // Forward pass - compute the activations
      activations = this->activate(data);

      // Backward pass - compute the deltas
      if(beta>0)
      {
         deltas = this->backpropagation(activations, data, avgActivations, rho, beta);
      }
      else
      {
         deltas = this->backpropagation(activations, data);
      }

      dW_ih = dW_ih + deltas[1]*activations[0].transpose();
      dW_ho = dW_ho + deltas[2]*activations[1].transpose();
      dbias_h = dbias_h + deltas[1];
      dbias_o = dbias_o + deltas[2];
   }

   // Return the gradients
   gradient[0] = dW_ih / numItems;
   gradient[1] = dW_ho / numItems;
   gradient[2] = dbias_h / numItems;
   gradient[3] = dbias_o / numItems;

   return gradient;
}

void Autoencoder::save(const char* filename)
{
   fstream outfile;
   outfile.open(filename, fstream::out | fstream::binary);

   if(outfile.is_open())
   {
      int input_size = this->numInputs;
      int hidden_size = this->numHidden;
      outfile.write((char*) (&input_size), sizeof(int));      
      outfile.write((char*) (&hidden_size), sizeof(int));
      outfile.write((char*) this->W_ih.data(), input_size*hidden_size*sizeof(double));
      outfile.write((char*) this->W_ho.data(), input_size*hidden_size*sizeof(double));
      outfile.write((char*) this->bias_h.data(), hidden_size*sizeof(double));
      outfile.write((char*) this->bias_o.data(), input_size*sizeof(double));

      outfile.close(); 
   }
}

Autoencoder* load_autoencoder(const char* filename)
{
   Autoencoder* autoencoder = NULL;

   fstream infile;
   infile.open(filename, fstream::in | fstream::binary);

   if(infile.is_open())
   {
      int input_size = 0;
      int hidden_size = 0;

      infile.read((char*) (&input_size), sizeof(int));
      infile.read((char*) (&hidden_size), sizeof(int));

      autoencoder = new Autoencoder(input_size, hidden_size);

      infile.read((char*) autoencoder->W_ih.data(), input_size*hidden_size*sizeof(double));
      infile.read((char*) autoencoder->W_ho.data(), input_size*hidden_size*sizeof(double));
      infile.read((char*) autoencoder->bias_h.data(), hidden_size*sizeof(double));
      infile.read((char*) autoencoder->bias_o.data(), input_size*sizeof(double));

      infile.close();
   }

   return autoencoder;
}

void Autoencoder::print()
{
   cout << "Num Input: " << this->numInputs << "\tNum Hidden: " << this->numHidden << endl;
   cout << "W_ih:" << endl << this->W_ih << endl;
   cout << "W_ho:" << endl << this->W_ho << endl;
   cout << "b_h:" << endl << this->bias_h << endl;
   cout << "b_o:" << endl << this->bias_o << endl;

}
