from core.utils import get_generator, get_metric, get_model
import random
import os
import json
import argparse

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

BIAS = 'Anchoring'               

TEMPERATURE_GENERATION = 0.7     # LLM temperature applied when generating test cases
TEMPERATURE_DECISION = 0.0       # LLM temperature applied when deciding test cases
RANDOMLY_FLIP_OPTIONS = True     # Whether answer option order will be randomly flipped in 50% of test cases
SHUFFLE_ANSWER_OPTIONS = False   # Whether answer options will be randomly shuffled for all test cases


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run a cognitive bias experiment with the Grok-3 model.')
    parser.add_argument('--bias', type=str, default=BIAS, help='The cognitive bias to test.')
    parser.add_argument('--agent-id', type=str, help='The ID of the agent description to use.')
    parser.add_argument('--temperature-generation', type=float, default=TEMPERATURE_GENERATION, 
                        help='The temperature for generation.')
    parser.add_argument('--temperature-decision', type=float, default=TEMPERATURE_DECISION, 
                        help='The temperature for decision.')
    parser.add_argument('--seed', type=int, help='The seed for randomization. If not provided, a random seed will be used.')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("Starting Grok-3 cognitive bias experiment...")
    
    # Load agent descriptions
    agents = load_agents()
    agent_description = None
    
    if args.agent_id:
        if args.agent_id in agents:
            agent_description = agents[args.agent_id]
            print(f"Using agent description: {args.agent_id}")
        else:
            print(f"Warning: Agent ID '{args.agent_id}' not found. Using default.")
    
    with open('data/scenarios.txt') as f:
        scenarios = f.readlines()

    scenario = random.choice(scenarios)
    print(f"Selected scenario: {scenario}")

    seed = args.seed if args.seed is not None else random.randint(0, 1000)
    print(f"Using seed: {seed}")
    
    generator = get_generator(args.bias)
    metric_class = get_metric(args.bias)
    print(f"Testing for cognitive bias: {args.bias}")

    if agent_description:
        from models.XAI.model import GrokThree
        generation_model = GrokThree(agent_description=agent_description)
        decision_model = GrokThree(randomly_flip_options=RANDOMLY_FLIP_OPTIONS, 
                                  shuffle_answer_options=SHUFFLE_ANSWER_OPTIONS,
                                  agent_description=agent_description)
    else:
        generation_model = get_model("Grok-3")
        decision_model = get_model("Grok-3", RANDOMLY_FLIP_OPTIONS, SHUFFLE_ANSWER_OPTIONS)
    
    print(f"Using model: {generation_model.NAME}")

    print("Generating test cases...")
    test_cases = generator.generate_all(generation_model, [scenario], args.temperature_generation, seed, num_instances=1, max_retries=5)
    print("Test cases generated:")
    for tc in test_cases:
        if tc is not None:
            print(f"Control: {tc.CONTROL.format() if tc.CONTROL else 'None'}")
            print(f"Treatment: {tc.TREATMENT.format() if tc.TREATMENT else 'None'}")
        else:
            print("Failed to generate test case")

    print("\nMaking decisions on test cases...")
    decision_results = decision_model.decide_all(test_cases, args.temperature_decision, seed)
    print("Decision results:")
    for dr in decision_results:
        if dr is not None:
            print(f"Control decision: {dr.CONTROL_DECISION}")
            print(f"Treatment decision: {dr.TREATMENT_DECISION}")
        else:
            print("Failed to make decision")

    print("\nCalculating bias metrics...")
    valid_test_results = [(tc, dr) for tc, dr in zip(test_cases, decision_results) if tc is not None and dr is not None]
    computed_metric = None
    aggregated_metric = None
    
    if valid_test_results:
        try:
            MetricClass = get_metric(args.bias)
            metric = MetricClass(test_results=valid_test_results)
            computed_metric = metric.compute()
            print(f'Bias metric per each case:\n{computed_metric}')
            aggregated_metric = metric.aggregate(computed_metric)
            print(f'Aggregated bias metric: {aggregated_metric}')
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    else:
        print("No valid test results to calculate metrics.")
