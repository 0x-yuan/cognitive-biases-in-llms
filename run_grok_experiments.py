"""
Script to run cognitive bias experiments with the Grok-3 model.
"""

import os
import argparse
import pandas as pd
from datetime import datetime
from core.utils import get_generator, get_metric, get_model, get_supported_models

os.environ["XAI_API_KEY"] = "xai-g2O6AziRqlNWabEi3vHy7FcnlNLy7aOWMPpYqOpX99QPWoHyJ4n0OvxO1Bd1TY7wcrmsgMg9eqfEQEJv"

DEFAULT_TEMPERATURE_GENERATION = 0.7
DEFAULT_TEMPERATURE_DECISION = 0.0
DEFAULT_RANDOMLY_FLIP_OPTIONS = True
DEFAULT_SHUFFLE_ANSWER_OPTIONS = False
DEFAULT_NUM_INSTANCES = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_SEED = 42

def run_experiment(bias, model_name="Grok-3", temperature_generation=DEFAULT_TEMPERATURE_GENERATION, 
                  temperature_decision=DEFAULT_TEMPERATURE_DECISION, randomly_flip_options=DEFAULT_RANDOMLY_FLIP_OPTIONS,
                  shuffle_answer_options=DEFAULT_SHUFFLE_ANSWER_OPTIONS, num_instances=DEFAULT_NUM_INSTANCES, 
                  max_retries=DEFAULT_MAX_RETRIES, seed=DEFAULT_SEED):
    """
    Run a cognitive bias experiment with the specified parameters.
    
    Args:
        bias (str): The cognitive bias to test.
        model_name (str): The name of the model to use.
        temperature_generation (float): The temperature for generation.
        temperature_decision (float): The temperature for decision.
        randomly_flip_options (bool): Whether to randomly flip options.
        shuffle_answer_options (bool): Whether to shuffle answer options.
        num_instances (int): The number of instances to generate.
        max_retries (int): The maximum number of retries.
        seed (int): The seed for randomization.
        
    Returns:
        tuple: A tuple containing the test cases, decision results, and metrics.
    """
    print(f"Running experiment for bias: {bias} with model: {model_name}")
    
    with open('data/scenarios.txt') as f:
        scenarios = f.readlines()
    
    generator = get_generator(bias)
    metric_class = get_metric(bias)
    
    generation_model = get_model(model_name)
    decision_model = get_model(model_name, randomly_flip_options, shuffle_answer_options)
    
    print(f"Generating {num_instances} test cases per scenario...")
    test_cases = generator.generate_all(
        generation_model, 
        scenarios, 
        temperature_generation, 
        seed, 
        num_instances=num_instances, 
        max_retries=max_retries
    )
    
    print("Making decisions on test cases...")
    decision_results = decision_model.decide_all(
        test_cases, 
        temperature_decision, 
        seed, 
        max_retries=max_retries
    )
    
    print("Calculating metrics...")
    # metric_class is already an instance
    metric_class.test_results = list(zip(test_cases, decision_results))
    computed_metric = metric_class.compute()
    aggregated_metric = metric_class.aggregate(computed_metric)
    
    return test_cases, decision_results, computed_metric, aggregated_metric

def save_results(bias, model_name, test_cases, decision_results, computed_metric, aggregated_metric):
    """
    Save the results of an experiment to a CSV file.
    
    Args:
        bias (str): The cognitive bias tested.
        model_name (str): The name of the model used.
        test_cases (list): The test cases.
        decision_results (list): The decision results.
        computed_metric (numpy.ndarray): The computed metrics.
        aggregated_metric (float): The aggregated metric.
    """
    os.makedirs('data/grok_results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    for i, (tc, dr, cm) in enumerate(zip(test_cases, decision_results, computed_metric)):
        if tc is not None and dr is not None:
            result = {
                'bias': bias,
                'model': model_name,
                'test_case_id': i,
                'control_decision': dr.CONTROL_DECISION if dr.CONTROL_DECISION is not None else None,
                'treatment_decision': dr.TREATMENT_DECISION if dr.TREATMENT_DECISION is not None else None,
                'metric_value': float(cm) if cm is not None else None,
                'timestamp': timestamp
            }
            results.append(result)
    
    df = pd.DataFrame(results)
    filename = f'data/grok_results/{bias}_{model_name}_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    summary_filename = f'data/grok_results/summary_{timestamp}.csv'
    summary_df = pd.DataFrame([{
        'bias': bias,
        'model': model_name,
        'aggregated_metric': float(aggregated_metric) if aggregated_metric is not None else None,
        'timestamp': timestamp
    }])
    
    if os.path.exists(summary_filename):
        existing_df = pd.read_csv(summary_filename)
        summary_df = pd.concat([existing_df, summary_df])
    
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary saved to {summary_filename}")

def main():
    """
    Main function to parse arguments and run experiments.
    """
    parser = argparse.ArgumentParser(description='Run cognitive bias experiments with the Grok-3 model.')
    parser.add_argument('--bias', type=str, default='Anchoring', help='The cognitive bias to test.')
    parser.add_argument('--model', type=str, default='Grok-3', help='The model to use.')
    parser.add_argument('--temperature-generation', type=float, default=DEFAULT_TEMPERATURE_GENERATION, 
                        help='The temperature for generation.')
    parser.add_argument('--temperature-decision', type=float, default=DEFAULT_TEMPERATURE_DECISION, 
                        help='The temperature for decision.')
    parser.add_argument('--randomly-flip-options', type=bool, default=DEFAULT_RANDOMLY_FLIP_OPTIONS, 
                        help='Whether to randomly flip options.')
    parser.add_argument('--shuffle-answer-options', type=bool, default=DEFAULT_SHUFFLE_ANSWER_OPTIONS, 
                        help='Whether to shuffle answer options.')
    parser.add_argument('--num-instances', type=int, default=DEFAULT_NUM_INSTANCES, 
                        help='The number of instances to generate per scenario.')
    parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, 
                        help='The maximum number of retries.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, 
                        help='The seed for randomization.')
    parser.add_argument('--all-biases', action='store_true', 
                        help='Run experiments for all available biases.')
    
    args = parser.parse_args()
    
    if args.all_biases:
        biases = [d for d in os.listdir('tests') if os.path.isdir(os.path.join('tests', d))]
    else:
        biases = [args.bias]
    
    for bias in biases:
        try:
            test_cases, decision_results, computed_metric, aggregated_metric = run_experiment(
                bias,
                args.model,
                args.temperature_generation,
                args.temperature_decision,
                args.randomly_flip_options,
                args.shuffle_answer_options,
                args.num_instances,
                args.max_retries,
                args.seed
            )
            
            save_results(bias, args.model, test_cases, decision_results, computed_metric, aggregated_metric)
            
            print(f"Experiment for bias {bias} completed successfully.")
            print(f"Aggregated metric: {aggregated_metric}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error running experiment for bias {bias}: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()
