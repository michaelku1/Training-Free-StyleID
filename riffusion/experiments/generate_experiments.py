import itertools
import requests
import json
from copy import deepcopy

# Your parameter lists
DENOISING = [0.2, 0.3, 0.4, 0.5]
GUIDANCE = [0.2, 0.3, 0.4, 0.5]

# Your other constants (replace with actual values)
START_SEED = 12345
END_SEED = 67890
STEPS = 50
SEED_IMAGE_PATH = "path/to/seed_image.jpg"
MASK_IMAGE_PATH = "path/to/mask_image.jpg"
ALPHA = 0.5
OUTPUT_PATH = "path/to/output"
API_ENDPOINT = "https://your-api-endpoint.com/generate"  # Replace with your actual endpoint

# Base data template
base_data = {
    "start": {"prompt": "", "seed": START_SEED},
    "num_inference_steps": STEPS,
    "seed_image_path": SEED_IMAGE_PATH,
    "mask_image_path": MASK_IMAGE_PATH,
    "alpha": ALPHA,
    "end": {"prompt": "", "seed": END_SEED},
    "output_path": OUTPUT_PATH,
}

def safe_json_parse(response):
    """Safely parse JSON response with error handling."""
    try:
        return response.json()
    except json.JSONDecodeError:
        # Return the raw text if JSON parsing fails
        return {"raw_response": response.text, "content_type": response.headers.get('content-type', 'unknown')}

def generate_parameter_combinations():
    """Generate all combinations of denoising and guidance parameters."""
    combinations = []
    for start_denoising, start_guidance, end_denoising, end_guidance in itertools.product(
        DENOISING, GUIDANCE, DENOISING, GUIDANCE
    ):
        combo = {
            'start_denoising': start_denoising,
            'start_guidance': start_guidance,
            'end_denoising': end_denoising,
            'end_guidance': end_guidance
        }
        combinations.append(combo)
    return combinations

def create_request_data(combo, experiment_id):
    """Create request data for a specific parameter combination."""
    data = deepcopy(base_data)
    
    # Add the parameter combinations to start and end
    data['start']['denoising'] = combo['start_denoising']
    data['start']['guidance'] = combo['start_guidance']
    data['end']['denoising'] = combo['end_denoising']
    data['end']['guidance'] = combo['end_guidance']
    
    # Optionally modify output path to include experiment details
    data['output_path'] = f"{OUTPUT_PATH}/exp_{experiment_id}_sd{combo['start_denoising']}_sg{combo['start_guidance']}_ed{combo['end_denoising']}_eg{combo['end_guidance']}"
    
    return data

def run_experiments():
    """Run experiments for all parameter combinations."""
    combinations = generate_parameter_combinations()
    results = []
    
    print(f"Running {len(combinations)} experiments...")
    
    for i, combo in enumerate(combinations):
        print(f"Experiment {i+1}/{len(combinations)}: {combo}")
        
        # Create request data
        request_data = create_request_data(combo, i+1)
        
        try:
            # Send POST request
            response = requests.post(
                API_ENDPOINT, 
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=300  # 5 minute timeout
            )
            
            # Print response details for debugging
            print(f"  Response status: {response.status_code}")
            print(f"  Response content-type: {response.headers.get('content-type', 'unknown')}")
            print(f"  Response length: {len(response.text)} characters")
            
            if response.status_code == 200:
                parsed_response = safe_json_parse(response)
                result = {
                    'experiment_id': i+1,
                    'parameters': combo,
                    'status': 'success',
                    'response': parsed_response
                }
                print(f"✓ Experiment {i+1} completed successfully")
            else:
                result = {
                    'experiment_id': i+1,
                    'parameters': combo,
                    'status': 'error',
                    'error': f"HTTP {response.status_code}: {response.text[:200]}..."  # Limit error text
                }
                print(f"✗ Experiment {i+1} failed: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            result = {
                'experiment_id': i+1,
                'parameters': combo,
                'status': 'error',
                'error': str(e)
            }
            print(f"✗ Experiment {i+1} failed: {e}")
        
        results.append(result)
    
    return results

def save_results(results, filename="./experiment_results.json"):
    """Save experiment results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

# Alternative: If you want the same denoising/guidance for both start and end
def generate_simple_combinations():
    """Generate combinations where start and end use the same parameters."""
    combinations = []
    for denoising, guidance in itertools.product(DENOISING, GUIDANCE):
        combo = {
            'denoising': denoising,
            'guidance': guidance
        }
        combinations.append(combo)
    return combinations

def create_simple_request_data(combo, experiment_id):
    """Create request data with same parameters for start and end."""
    data = deepcopy(base_data)
    
    # Use same parameters for both start and end
    data['start']['denoising'] = combo['denoising']
    data['start']['guidance'] = combo['guidance']
    data['end']['denoising'] = combo['denoising']
    data['end']['guidance'] = combo['guidance']
    
    data['output_path'] = f"{OUTPUT_PATH}/exp_{experiment_id}_d{combo['denoising']}_g{combo['guidance']}"
    
    return data

def run_simple_experiments():
    """Run experiments with same parameters for start and end."""
    combinations = generate_simple_combinations()
    results = []
    
    print(f"Running {len(combinations)} simple experiments...")
    
    for i, combo in enumerate(combinations):
        print(f"Experiment {i+1}/{len(combinations)}: denoising={combo['denoising']}, guidance={combo['guidance']}")
        
        request_data = create_simple_request_data(combo, i+1)
        
        try:
            response = requests.post(
                API_ENDPOINT, 
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=300
            )
            
            # Print response details for debugging
            print(f"  Response status: {response.status_code}")
            print(f"  Response content-type: {response.headers.get('content-type', 'unknown')}")
            
            if response.status_code == 200:
                parsed_response = safe_json_parse(response)
                result = {
                    'experiment_id': i+1,
                    'parameters': combo,
                    'status': 'success',
                    'response': parsed_response
                }
                print(f"✓ Experiment {i+1} completed successfully")
            else:
                result = {
                    'experiment_id': i+1,
                    'parameters': combo,
                    'status': 'error',
                    'error': f"HTTP {response.status_code}: {response.text[:200]}..."
                }
                print(f"✗ Experiment {i+1} failed: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            result = {
                'experiment_id': i+1,
                'parameters': combo,
                'status': 'error',
                'error': str(e)
            }
            print(f"✗ Experiment {i+1} failed: {e}")
        
        results.append(result)
    
    return results

def test_api_endpoint():
    """Test the API endpoint before running experiments."""
    print(f"Testing API endpoint: {API_ENDPOINT}")
    
    try:
        # Try a simple GET request first
        response = requests.get(API_ENDPOINT, timeout=10)
        print(f"GET response: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Response preview: {response.text[:200]}...")
        
        # Try a POST with minimal data
        test_data = {"test": "connection"}
        response = requests.post(
            API_ENDPOINT,
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"POST response: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Response preview: {response.text[:200]}...")
        
    except requests.exceptions.RequestException as e:
        print(f"API endpoint test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Test API endpoint first
    print("=" * 50)
    if not test_api_endpoint():
        print("\n⚠️  API endpoint test failed. Please check your API_ENDPOINT variable.")
        print("Current endpoint:", API_ENDPOINT)
        exit(1)
    
    print("\n" + "=" * 50)
    
    # Choose one of the following approaches:
    
    # Option 1: Full combinations (start and end can have different parameters)
    # This creates 4x4x4x4 = 256 combinations
    # results = run_experiments()
    
    # Option 2: Simple combinations (start and end use same parameters)
    # This creates 4x4 = 16 combinations
    results = run_simple_experiments()
    
    # Save results
    save_results(results)
    
    # Print summary
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'error'])