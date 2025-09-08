import requests
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants that you'll need to define
START_SEED = 12345  # Replace with your actual start seed
END_SEED = 67890    # Replace with your actual end seed
STEPS = 50          # Replace with your actual steps
SEED_IMAGE_PATH = ""  # Replace with actual path
MASK_IMAGE_PATH = ""  # Replace with actual path
ALPHA = 0.5         # Replace with your actual alpha value
OUTPUT_PATH = "path/to/output/"  # Replace with actual output path

def load_experiments(json_file_path):
    """Load experiments from JSON file"""
    try:
        with open(json_file_path, 'r') as file:
            experiments = json.load(file)
        return experiments
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return None

def run_experiment(experiment):
    """Run a single experiment"""
    experiment_id = experiment.get('experiment_id')
    parameters = experiment.get('parameters', {})
    
    # Extract denoising and guidance from the experiment parameters
    denoising = parameters.get('denoising', 0.2)  # Default fallback
    guidance = parameters.get('guidance', 0.2)    # Default fallback
    
    # Prepare the data payload
    data = {
        "start": {
            "prompt": "", 
            "seed": START_SEED, 
            "denoising": denoising, 
            "guidance": guidance
        },
        "num_inference_steps": STEPS,
        "seed_image_path": SEED_IMAGE_PATH,
        "mask_image_path": MASK_IMAGE_PATH,
        "alpha": ALPHA,
        "end": {
            "prompt": "", 
            "seed": END_SEED, 
            "denoising": denoising, 
            "guidance": guidance
        },
        "output_path": f"{OUTPUT_PATH}/experiment_{experiment_id}",  # Unique output path per experiment
    }
    
    logger.info(f"Running experiment {experiment_id} with denoising={denoising}, guidance={guidance}")
    
    try:
        response = requests.post("http://127.0.0.1:5000/run_inference/", json=data)
        logger.info(f"Experiment {experiment_id} - Response status code: {response.status_code}")
        logger.info(f"Experiment {experiment_id} - Response text: {response.text[:500]}")
        
        # Return experiment results
        return {
            "experiment_id": experiment_id,
            "parameters": parameters,
            "status_code": response.status_code,
            "response": response.text[:500],
            "success": response.status_code == 200
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Experiment {experiment_id} - Request failed: {e}")
        return {
            "experiment_id": experiment_id,
            "parameters": parameters,
            "error": str(e),
            "success": False
        }

def main():
    """Main function to run all experiments"""
    # Load experiments from JSON file
    json_file_path = "experiments.json"  # Replace with your actual file path
    experiments = load_experiments(json_file_path)
    
    if not experiments:
        logger.error("No experiments loaded. Exiting.")
        return
    
    logger.info(f"Loaded {len(experiments)} experiments")
    
    # Store results
    results = []
    
    # Loop through each experiment
    for experiment in experiments:
        result = run_experiment(experiment)
        results.append(result)
        
        # Optional: Add a small delay between requests to avoid overwhelming the server
        import time
        time.sleep(1)
    
    # Log summary
    successful_experiments = sum(1 for r in results if r.get('success', False))
    logger.info(f"Completed {len(results)} experiments. {successful_experiments} successful.")
    
    # Optional: Save results to file
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to experiment_results.json")

if __name__ == "__main__":
    main()