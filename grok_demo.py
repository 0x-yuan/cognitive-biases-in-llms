from core.utils import get_generator, get_metric, get_model
import random
import os

os.environ["XAI_API_KEY"] = "xai-g2O6AziRqlNWabEi3vHy7FcnlNLy7aOWMPpYqOpX99QPWoHyJ4n0OvxO1Bd1TY7wcrmsgMg9eqfEQEJv"

BIAS = 'Anchoring'               

TEMPERATURE_GENERATION = 0.7     # LLM temperature applied when generating test cases
TEMPERATURE_DECISION = 0.0       # LLM temperature applied when deciding test cases
RANDOMLY_FLIP_OPTIONS = True     # Whether answer option order will be randomly flipped in 50% of test cases
SHUFFLE_ANSWER_OPTIONS = False   # Whether answer options will be randomly shuffled for all test cases


if __name__ == "__main__":
    print("Starting Grok-3 cognitive bias experiment...")

    with open('data/scenarios.txt') as f:
        scenarios = f.readlines()

    scenario = random.choice(scenarios)
    print(f"Selected scenario: {scenario}")

    seed = random.randint(0, 1000)
    print(f"Using seed: {seed}")
    
    generator = get_generator(BIAS)
    metric_class = get_metric(BIAS)
    print(f"Testing for cognitive bias: {BIAS}")

    generation_model = get_model("Grok-3")
    decision_model = get_model("Grok-3", RANDOMLY_FLIP_OPTIONS, SHUFFLE_ANSWER_OPTIONS)
    print(f"Using model: {generation_model.NAME}")

    print("Generating test cases...")
    test_cases = generator.generate_all(generation_model, [scenario], TEMPERATURE_GENERATION, seed, num_instances=1, max_retries=5)
    print("Test cases generated:")
    for tc in test_cases:
        if tc is not None:
            print(f"Control: {tc.CONTROL.format() if tc.CONTROL else 'None'}")
            print(f"Treatment: {tc.TREATMENT.format() if tc.TREATMENT else 'None'}")
        else:
            print("Failed to generate test case")

    print("\nMaking decisions on test cases...")
    decision_results = decision_model.decide_all(test_cases, TEMPERATURE_DECISION, seed)
    print("Decision results:")
    for dr in decision_results:
        if dr is not None:
            print(f"Control decision: {dr.CONTROL_DECISION}")
            print(f"Treatment decision: {dr.TREATMENT_DECISION}")
        else:
            print("Failed to make decision")

    print("\nCalculating bias metrics...")
    # metric_class is already an instance
    metric_class.test_results = list(zip(test_cases, decision_results))
    computed_metric = metric_class.compute()
    print(f'Bias metric per each case:\n{computed_metric}')
    aggregated_metric = metric_class.aggregate(computed_metric)
    print(f'Aggregated bias metric: {aggregated_metric}')
