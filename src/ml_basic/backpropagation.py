# %%
import random
import math
from typing import List


# %%
class Neuron:

    def __init__(self, bias: float) -> None:
        self.bias = bias
        self.weights = []
        self.inputs = []

    def _sigmoid(self, total_net_input: float) -> float:
        return 1 / (1 + math.exp(-total_net_input))

    def _calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] + self.weights[i]
        return total + self.bias

    def caculate_output(self, inputs: List[float]) -> float:
        self.inputs = inputs.copy()
        self.output = self._sigmoid(self._calculate_total_net_input())
        return self.output

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(
            target_output) * self.calculate_pd_total_net_input_wrt_input()

    # 每一个神经元的误差是由平方差公式计算的
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output)**2

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


class Denselayer:

    def __init__(self, units, bias) -> None:
        self.bias = bias if bias else random.random()
        self.neurons: List[Neuron] = []
        for i in range(units):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs: List[float]) -> List[float]:
        outputs = []
        for n in self.neurons:
            outputs.append(n.caculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for n in self.neurons:
            outputs.append(n.output)
        return outputs


class NeuralNetwork:

    def __init__(self,
                 num_inputs: int,
                 num_hidden: int,
                 num_outputs: int,
                 hidden_layer_weights: List[float] = None,
                 hidden_layer_bias: float = None,
                 output_layer_weights: List[float] = None,
                 output_layer_bias: float = None,
                 learning_rate: float = 0.5) -> None:
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate

        self.hidden_layer = Denselayer(num_hidden, hidden_layer_bias)
        self.output_layer = Denselayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(
            hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(
            output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(
            self, hidden_layer_weights: List[float]):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                weight = random.random(
                ) if not hidden_layer_weights else hidden_layer_weights[
                    weight_num]
                self.hidden_layer.neurons[h].weights.append(weight)
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(
            self, output_layer_weights: List[float]):
        weight_num = 0
        for h in range(len(self.output_layer.neurons)):
            for i in range(self.num_inputs):
                weight = random.random(
                ) if not output_layer_weights else output_layer_weights[
                    weight_num]
                self.output_layer.neurons[h].weights.append(weight)
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # output neurons' values
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(
            self.output_layer.neurons)
        for i in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron_total_net_input[
                i] = self.output_layer.neurons[
                    i].calculate_pd_error_wrt_total_net_input(
                        training_outputs[i])

        # hidden layers'
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(
            self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[
                    o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[
                h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[
                    h].calculate_pd_total_net_input_wrt_input()

        # 3. 更新输出层权重系数
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[
                    o] * self.output_layer.neurons[
                        o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[
                    w_ho] -= self.learning_rate * pd_error_wrt_weight

        # 4. 更新隐含层的权重系数
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[
                    h] * self.hidden_layer.neurons[
                        h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[
                    w_ih] -= self.learning_rate * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(
                    training_outputs[o])
        return total_error


nn = NeuralNetwork(2,
                   2,
                   2,
                   hidden_layer_weights=[0.15, 0.2, 0.25, 0.3],
                   hidden_layer_bias=0.35,
                   output_layer_weights=[0.4, 0.45, 0.5, 0.55],
                   output_layer_bias=0.6)
for i in range(10000):
    nn.train([0.05, 0.1], [0.01, 0.09])
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.09]]]), 9))

# %%
nn.inspect()
# %%
