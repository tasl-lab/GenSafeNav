import numpy as np
import matplotlib.pyplot as plt

# Define vectorized functions for zero max and zero min
vec_zero_max = np.vectorize(lambda x: max(x, 0)) # unused
vec_zero_min = np.vectorize(lambda x: min(x, 0))

# Definition of the pinball loss function
def pinball(u, alpha):
    return alpha * u - vec_zero_min(u)

class DtACI:
    def __init__(self, alpha=0.1, gammas=np.array([0.05, 0.1, 0.2]), 
                 sigma=1/1000, eta=2.72, initial_pred=0.5):
        self.alpha = alpha
        self.gammas = gammas
        self.sigma = sigma
        self.eta = eta
        self.true_values = []
        self.predictions = []
        self.errors = []
        self.gamma_sequence = []
        self.weighted_average_prediction_sequence = []
        self.coverage = []
        self.num_experts = len(gammas)
        self.initial_pred = initial_pred
        self.expert_predictions = np.full(self.num_experts, initial_pred)
        self.expert_weights = np.ones(self.num_experts)
        self.current_expert = np.random.choice(np.arange(self.num_experts))        
        self.expert_cumulative_losses = np.zeros(self.num_experts)
        self.expert_probs = np.full(self.num_experts, 1 / self.num_experts)
        self.last_prediction = initial_pred

    def make_prediction(self):
        if len(self.true_values) == 0:
            self.last_prediction = self.initial_pred
            return self.initial_pred
        else:
            prediction = self.expert_predictions[self.current_expert]
            self.last_prediction = prediction
            return prediction

    def update_true_value(self, new_true_value):
        self.true_values.append(new_true_value)
        truth = new_true_value
        prediction = self.last_prediction

        self.predictions.append(prediction)
        self.gamma_sequence.append(self.gammas[self.current_expert])
        self.weighted_average_prediction_sequence.append(np.sum(self.expert_probs * self.expert_predictions))

        expert_losses = pinball(truth - self.expert_predictions, self.alpha)
        self.expert_predictions -= self.gammas * (self.alpha - (self.expert_predictions < truth).astype(float))
        
        if self.eta < float('inf'):
            expert_bar_weights = self.expert_weights * np.exp(-self.eta * expert_losses)
            expert_next_weights = (1 - self.sigma) * expert_bar_weights / np.sum(expert_bar_weights) + self.sigma / self.num_experts
            self.expert_probs = expert_next_weights / np.sum(expert_next_weights)
            self.current_expert = np.random.choice(np.arange(self.num_experts), p=self.expert_probs)
            self.expert_weights = expert_next_weights
        else:
            self.expert_cumulative_losses += expert_losses
            self.current_expert = np.argmin(self.expert_cumulative_losses)
            
        self.errors.append(prediction - truth)
        self.coverage.append(float(prediction >= truth))

def compute_local_coverage(coverage, window_size):
    local_coverage = []
    for i in range(len(coverage)):
        start = max(0, i - window_size + 1)
        local_coverage.append(np.mean(coverage[start:i+1]))
    return local_coverage

# Plotting the results
def plot_results(true_values, predictions, gamma_sequence, weighted_average_prediction_sequence, errors, coverage, filename='test_dtaci.png', window_size=10):
    fig, axs = plt.subplots(6, 1, figsize=(10, 30))

    axs[0].plot(true_values, label='True Values', color='green')
    axs[0].plot(predictions, label='Predictions', color='blue', linestyle='dotted')
    axs[0].set_title('True Values and Predictions Over Time')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].legend()

    axs[1].plot(gamma_sequence, label='Gamma Sequence', color='purple')
    axs[1].set_title('Gamma Sequence Over Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Gamma Value')
    axs[1].legend()

    axs[2].plot(weighted_average_prediction_sequence, label='Weighted Average Prediction Sequence', color='orange')
    axs[2].set_title('Weighted Average Prediction Sequence Over Time')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Prediction Value')
    axs[2].legend()

    axs[3].plot(errors, label='Errors', color='red')
    axs[3].set_title('Errors Over Time')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Error Value')
    axs[3].legend()

    coverage_adapt = np.cumsum(coverage) / np.arange(1, len(coverage) + 1)
    axs[4].plot(coverage_adapt, label='Coverage Probability', color='blue')
    axs[4].set_title('Coverage Probability Over Time')
    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('Coverage Probability')
    axs[4].legend()
    
    local_coverage = compute_local_coverage(coverage, window_size)
    axs[5].plot(local_coverage, label=f'Local Coverage Probability (window size={window_size})', color='magenta')
    axs[5].set_title('Local Coverage Probability Over Time')
    axs[5].set_xlabel('Time')
    axs[5].set_ylabel('Local Coverage Probability')
    axs[5].legend()

    plt.tight_layout()
    plt.savefig(filename)

# Example usage
if __name__ == "__main__":
    model = DtACI(alpha=0.1, gammas=np.array([0.1, 0.05, 0.2]))
    betas = np.array([np.sin(i / 10) + np.random.normal() * 0.1 for i in range(1, 1000)])

    for beta in betas:
        prediction = model.make_prediction()
        model.update_true_value(beta)

    plot_results(model.true_values, model.predictions, model.gamma_sequence, model.weighted_average_prediction_sequence, model.errors, model.coverage)