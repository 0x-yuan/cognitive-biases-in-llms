"""
Script to run cognitive bias experiments with the Grok-3 model.
"""

import os
import argparse
import pandas as pd
import json
from datetime import datetime
from core.utils import get_generator, get_metric, get_model, get_supported_models

if "XAI_API_KEY" not in os.environ:
    print("Warning: XAI_API_KEY environment variable not set. Please set it before running this script.")
    print("Example: export XAI_API_KEY=your-api-key")
    exit(1)

AGENTS_FILE = 'data/generated_agents.json'

def load_agents():
    """
    Load agent descriptions from the JSON file.
    
    Returns:
        dict: A dictionary mapping agent IDs to their descriptions.
    """
    if not os.path.exists(AGENTS_FILE):
        print(f"Warning: Agents file {AGENTS_FILE} not found.")
        return {}
    
    try:
        with open(AGENTS_FILE, 'r') as f:
            data = json.load(f)
        
        agents = {}
        for i, agent in enumerate(data):
            if 'name' in agent and 'description' in agent:
                agent_id = agent.get('name', f'agent_{i}').replace(' ', '_').lower()
                agents[agent_id] = agent['description']
        
        if not agents:
            agents = {
                'mathematician': 'You are a PhD-level mathematician with expertise in probability, statistics, and decision theory.',
                'economist': 'You are an experienced economist specializing in behavioral economics and decision-making.',
                'psychologist': 'You are a cognitive psychologist with expertise in human decision-making and cognitive biases.',
                'rationalist': 'You are a rationalist trained to identify and avoid cognitive biases.',
                'intuitive': 'You rely heavily on intuition and gut feelings when making decisions.'
            }
        
        return agents
    except Exception as e:
        print(f"Error loading agents file: {e}")
        return {}

DEFAULT_TEMPERATURE_GENERATION = 0.7
DEFAULT_TEMPERATURE_DECISION = 0.0
DEFAULT_RANDOMLY_FLIP_OPTIONS = True
DEFAULT_SHUFFLE_ANSWER_OPTIONS = False
DEFAULT_NUM_INSTANCES = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_SEED = 42

def run_experiment(bias, model_name="Grok-3", agent_id=None, temperature_generation=DEFAULT_TEMPERATURE_GENERATION, 
                  temperature_decision=DEFAULT_TEMPERATURE_DECISION, randomly_flip_options=DEFAULT_RANDOMLY_FLIP_OPTIONS,
                  shuffle_answer_options=DEFAULT_SHUFFLE_ANSWER_OPTIONS, num_instances=DEFAULT_NUM_INSTANCES, 
                  max_retries=DEFAULT_MAX_RETRIES, seed=DEFAULT_SEED):
    """
    Run a cognitive bias experiment with the specified parameters.
    
    Args:
        bias (str): The cognitive bias to test.
        model_name (str): The name of the model to use.
        agent_id (str): The ID of the agent description to use.
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
    # Load agent descriptions
    agents = load_agents()
    agent_description = None
    
    if agent_id:
        if agent_id in agents:
            agent_description = agents[agent_id]
            print(f"Using agent description: {agent_id}")
        else:
            print(f"Warning: Agent ID '{agent_id}' not found in agents file. Using default.")
    
    print(f"Running experiment for bias: {bias} with model: {model_name}")
    
    with open('data/scenarios.txt') as f:
        scenarios = f.readlines()
    
    generator = get_generator(bias)
    metric_class = get_metric(bias)
    
    # Create models with agent description
    if model_name == "Grok-3" and agent_description:
        from models.XAI.model import GrokThree
        generation_model = GrokThree(agent_description=agent_description)
        decision_model = GrokThree(randomly_flip_options=randomly_flip_options, 
                                  shuffle_answer_options=shuffle_answer_options,
                                  agent_description=agent_description)
    else:
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
    parser.add_argument('--agent-id', type=str, help='The ID of the agent description to use.')
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
    parser.add_argument('--all-agents', action='store_true',
                        help='Run experiments for all available agent descriptions.')
    
    args = parser.parse_args()
    
    if args.all_biases:
        biases = [d for d in os.listdir('tests') if os.path.isdir(os.path.join('tests', d))]
    else:
        biases = [args.bias]
    
    # Load agent descriptions
    agents = load_agents()
    
    agent_ids = []
    if args.all_agents:
        agent_ids = list(agents.keys())
        print(f"Running experiments with all {len(agent_ids)} agents")
    elif args.agent_id:
        if args.agent_id in agents:
            agent_ids = [args.agent_id]
            print(f"Running experiments with agent: {args.agent_id}")
        else:
            print(f"Warning: Agent ID '{args.agent_id}' not found. Running without agent description.")
            agent_ids = [None]
    else:
        agent_ids = [None]  # Run without agent description
    
    for bias in biases:
        for agent_id in agent_ids:
            try:
                agent_suffix = f"_{agent_id}" if agent_id else ""
                print(f"Running experiment for bias: {bias} with model: {args.model}{agent_suffix}")
                
                test_cases, decision_results, computed_metric, aggregated_metric = run_experiment(
                    bias,
                    args.model,
                    agent_id,
                    args.temperature_generation,
                    args.temperature_decision,
                    args.randomly_flip_options,
                    args.shuffle_answer_options,
                    args.num_instances,
                    args.max_retries,
                    args.seed
                )
                
                model_name = f"{args.model}_{agent_id}" if agent_id else args.model
                save_results(bias, model_name, test_cases, decision_results, computed_metric, aggregated_metric)
                
                print(f"Experiment for bias {bias} with agent {agent_id if agent_id else 'default'} completed successfully.")
                print(f"Aggregated metric: {aggregated_metric}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error running experiment for bias {bias} with agent {agent_id if agent_id else 'default'}: {e}")
                print("-" * 50)

if __name__ == "__main__":
    main()
