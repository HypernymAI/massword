#!/usr/bin/env python3
"""
Facility Support Demolition Test - SCALED VERSION
Tests ALL yellies categories individually with 4x samples and parallelization
"""
import os
import sys
import json
import yaml
import sqlite3
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_prompt_ops.core.metrics import FacilityMetric
from llama_prompt_ops.core.datasets import ConfigurableJSONAdapter
from facility_support_yellies import FacilityYellies
import dspy
from dspy.evaluate import Evaluate

# Thread-safe counter for rate limiting
class RateLimiter:
    def __init__(self, max_per_minute=60):
        self.lock = threading.Lock()
        self.calls = []
        self.max_per_minute = max_per_minute
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]
            
            if len(self.calls) >= self.max_per_minute:
                # Wait until the oldest call is > 1 minute old
                sleep_time = 60 - (now - self.calls[0]) + 1
                print(f"Rate limit reached ({len(self.calls)} calls in last minute), sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            
            self.calls.append(now)

rate_limiter = RateLimiter(max_per_minute=60)

def setup_database(samples, runs):
    """Create database for storing detailed results."""
    # Create timestamped database name with test parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_name = f'facility_scaled_results_{timestamp}_s{samples}_r{runs}.db'
    print(f"Creating database: {db_name}")
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS test_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        samples_per_test INTEGER,
        total_samples INTEGER
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_run_id INTEGER,
        yellie_type TEXT,
        example_id INTEGER,
        metric_score REAL,
        urgency_predicted TEXT,
        urgency_actual TEXT,
        sentiment_predicted TEXT,
        sentiment_actual TEXT,
        categories_score REAL,
        valid_json BOOLEAN,
        response_length INTEGER,
        error_message TEXT,
        FOREIGN KEY(test_run_id) REFERENCES test_runs(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS aggregate_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_run_id INTEGER,
        yellie_type TEXT,
        avg_score REAL,
        valid_json_rate REAL,
        urgency_accuracy REAL,
        sentiment_accuracy REAL,
        avg_categories_score REAL,
        total_examples INTEGER,
        improvement_over_baseline REAL,
        FOREIGN KEY(test_run_id) REFERENCES test_runs(id)
    )''')
    
    conn.commit()
    return conn, db_name

def evaluate_single_example(program, example, yellie_type, example_id, metric, max_retries=3):
    """Evaluate a single example with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            # Rate limit
            rate_limiter.wait_if_needed()
            
            # Run prediction
            pred = program(example['question'])
            response = pred.answer
            
            # Evaluate
            try:
                # Parse JSON to check validity
                parsed = json.loads(response)
                valid_json = True
                
                # Get metric results
                metric_result = metric.evaluate(example['answer'], response)
                
                # Extract components
                ground_truth = json.loads(example['answer'])
                
                result = {
                    'yellie_type': yellie_type,
                    'example_id': example_id,
                    'metric_score': metric_result['total'],
                    'urgency_predicted': parsed.get('urgency', ''),
                    'urgency_actual': ground_truth.get('urgency', ''),
                    'sentiment_predicted': parsed.get('sentiment', ''),
                    'sentiment_actual': ground_truth.get('sentiment', ''),
                    'categories_score': metric_result.get('correct_categories', 0),
                    'valid_json': valid_json,
                    'response_length': len(response),
                    'error_message': None
                }
                
                return result
                
            except json.JSONDecodeError as e:
                # Invalid JSON
                return {
                    'yellie_type': yellie_type,
                    'example_id': example_id,
                    'metric_score': 0.0,
                    'urgency_predicted': '',
                    'urgency_actual': json.loads(example['answer']).get('urgency', ''),
                    'sentiment_predicted': '',
                    'sentiment_actual': json.loads(example['answer']).get('sentiment', ''),
                    'categories_score': 0.0,
                    'valid_json': False,
                    'response_length': len(response),
                    'error_message': f"Invalid JSON: {str(e)}"
                }
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Error on attempt {attempt + 1}, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                return {
                    'yellie_type': yellie_type,
                    'example_id': example_id,
                    'metric_score': 0.0,
                    'urgency_predicted': '',
                    'urgency_actual': '',
                    'sentiment_predicted': '',
                    'sentiment_actual': '',
                    'categories_score': 0.0,
                    'valid_json': False,
                    'response_length': 0,
                    'error_message': f"Failed after {max_retries} attempts: {str(e)}"
                }

def evaluate_yellie_variant(yellie_type, instruction, valset, metric, max_workers=5, runs_per_example=1):
    """Evaluate a single yellie variant with parallelization."""
    print(f"\n{'='*60}")
    print(f"Testing: {yellie_type.upper()}")
    print(f"Instruction length: {len(instruction)} chars")
    print(f"Examples: {len(valset)}, Runs per example: {runs_per_example}")
    print(f"Total evaluations: {len(valset) * runs_per_example}")
    print(f"{'='*60}")
    
    # Create program
    class TestProgram(dspy.Module):
        def __init__(self, instruction):
            super().__init__()
            self.predictor = dspy.Predict("question -> answer")
            self.predictor.signature.instructions = instruction
        
        def forward(self, question):
            return self.predictor(question=question)
    
    program = TestProgram(instruction)
    
    # Run evaluations in parallel
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks (multiple runs per example)
        future_to_example = {}
        for i, example in enumerate(valset):
            for run in range(runs_per_example):
                # Create unique ID for each run
                eval_id = f"{i}_run{run}"
                future = executor.submit(evaluate_single_example, program, example, yellie_type, eval_id, metric)
                future_to_example[future] = (i, example, run)
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_example):
            completed += 1
            example_id, example, run_num = future_to_example[future]
            
            try:
                result = future.result()
                # Store original example_id for aggregation
                result['original_example_id'] = example_id
                result['run_number'] = run_num
                results.append(result)
                
                total_tasks = len(valset) * runs_per_example
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_tasks - completed) / rate
                    print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) - "
                          f"Rate: {rate:.1f} ex/s - ETA: {eta:.0f}s")
                    
            except Exception as e:
                print(f"Error processing example {example_id}: {str(e)}")
                results.append({
                    'yellie_type': yellie_type,
                    'example_id': example_id,
                    'metric_score': 0.0,
                    'valid_json': False,
                    'error_message': str(e)
                })
    
    total_time = time.time() - start_time
    print(f"\nCompleted {yellie_type} in {total_time:.1f}s ({len(valset)/total_time:.1f} ex/s)")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Facility Support Scaled Demolition Test')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples per variant (default: 50)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per example (default: 1)')
    args = parser.parse_args()
    
    print("="*80)
    print("FACILITY SUPPORT SCALED DEMOLITION TEST")
    print("="*80)
    print(f"Testing ALL yellie categories")
    print(f"Workers: {args.workers}")
    print(f"Samples per variant: {args.samples}")
    print(f"Runs per example: {args.runs}")
    print(f"Total evaluations per variant: {args.samples * args.runs}")
    print(f"Test started: {datetime.now()}")
    
    # Setup database
    conn, db_name = setup_database(args.samples, args.runs)
    c = conn.cursor()
    
    # Load dataset
    dataset_path = "use-cases/facility-support-analyzer/dataset.json"
    adapter = ConfigurableJSONAdapter(
        dataset_path=dataset_path,
        input_field=["fields", "input"],
        golden_output_field="answer"
    )
    raw_data = adapter.load_raw_data()
    
    # Convert to DSPy Example format
    examples = []
    for item in raw_data:
        example = dspy.Example(
            question=item['fields']['input'],
            answer=item['answer']
        ).with_inputs('question')
        examples.append(example)
    
    # Create validation set based on requested samples
    # Use first N examples
    start_idx = 0
    end_idx = min(args.samples, len(examples))
    valset = examples[start_idx:end_idx]
    
    print(f"\nDataset: {len(examples)} total examples")
    print(f"Validation set: {len(valset)} examples")
    
    # Configure DSPy
    model_name = "openrouter/meta-llama/llama-3.3-70b-instruct"
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("Please run: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Set API key in environment for litellm
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    lm = dspy.LM(
        model=model_name,
        max_tokens=1000,
        temperature=0,
        api_key=api_key
    )
    dspy.settings.configure(lm=lm)
    
    # Initialize metric
    metric = FacilityMetric(strict_json=False)
    
    # Load optimized prompt
    with open("use-cases/facility-support-analyzer/example_result.yaml", 'r') as f:
        result_data = yaml.safe_load(f)
    
    full_prompt = result_data['system']
    
    # Extract just the instruction (remove few-shot examples)
    lines = full_prompt.strip().split('\n')
    instruction_lines = []
    for line in lines:
        if line.strip().startswith('---') or 'Example' in line:
            break
        instruction_lines.append(line)
    instruction_only = '\n'.join(instruction_lines).strip()
    
    print(f"\nBaseline instruction: {len(instruction_only)} chars")
    
    # Create test run
    c.execute("INSERT INTO test_runs (timestamp, samples_per_test, total_samples) VALUES (?, ?, ?)",
              (datetime.now().isoformat(), len(valset), len(valset) * 7))
    test_run_id = c.lastrowid
    conn.commit()
    
    # Define all yellie variants to test
    yellie_variants = [
        ("baseline", instruction_only),
        ("financial_penalties", FacilityYellies.financial_penalties(instruction_only)),
        ("domain_failures", FacilityYellies.domain_failure_examples(instruction_only)),
        ("category_requirements", FacilityYellies.category_requirements(instruction_only)),
        ("urgency_keywords", FacilityYellies.urgency_keywords(instruction_only)),
        ("sentiment_analysis", FacilityYellies.sentiment_analysis(instruction_only)),
        ("kitchen_sink", FacilityYellies.combine_all(instruction_only))
    ]
    
    # Store all results
    all_results = {}
    
    # Test each variant
    for yellie_type, instruction in yellie_variants:
        results = evaluate_yellie_variant(yellie_type, instruction, valset, metric, 
                                        max_workers=args.workers, runs_per_example=args.runs)
        all_results[yellie_type] = results
        
        # Save to database
        for r in results:
            c.execute("""INSERT INTO results 
                (test_run_id, yellie_type, example_id, metric_score, urgency_predicted, 
                 urgency_actual, sentiment_predicted, sentiment_actual, categories_score, 
                 valid_json, response_length, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (test_run_id, r['yellie_type'], r['example_id'], r['metric_score'],
                 r.get('urgency_predicted', ''), r.get('urgency_actual', ''),
                 r.get('sentiment_predicted', ''), r.get('sentiment_actual', ''),
                 r.get('categories_score', 0), r['valid_json'], 
                 r.get('response_length', 0), r.get('error_message')))
        
        # Calculate aggregates
        valid_results = [r for r in results if r.get('error_message') is None]
        if valid_results:
            avg_score = sum(r['metric_score'] for r in valid_results) / len(valid_results)
            valid_json_rate = sum(1 for r in valid_results if r['valid_json']) / len(valid_results)
            
            # Component accuracies
            urgency_correct = sum(1 for r in valid_results 
                                if r.get('urgency_predicted') == r.get('urgency_actual'))
            sentiment_correct = sum(1 for r in valid_results 
                                  if r.get('sentiment_predicted') == r.get('sentiment_actual'))
            
            urgency_acc = urgency_correct / len(valid_results) if valid_results else 0
            sentiment_acc = sentiment_correct / len(valid_results) if valid_results else 0
            avg_cat_score = sum(r.get('categories_score', 0) for r in valid_results) / len(valid_results)
            
            # Save aggregate
            c.execute("""INSERT INTO aggregate_results 
                (test_run_id, yellie_type, avg_score, valid_json_rate, urgency_accuracy,
                 sentiment_accuracy, avg_categories_score, total_examples, improvement_over_baseline)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (test_run_id, yellie_type, avg_score, valid_json_rate, urgency_acc,
                 sentiment_acc, avg_cat_score, len(valid_results), 0))  # Will update improvement later
        
        conn.commit()
    
    # Update improvement over baseline
    c.execute("SELECT avg_score FROM aggregate_results WHERE test_run_id = ? AND yellie_type = 'baseline'",
              (test_run_id,))
    baseline_score = c.fetchone()[0]
    
    for yellie_type in [v[0] for v in yellie_variants if v[0] != 'baseline']:
        c.execute("SELECT avg_score FROM aggregate_results WHERE test_run_id = ? AND yellie_type = ?",
                  (test_run_id, yellie_type))
        yellie_score = c.fetchone()[0]
        improvement = (yellie_score - baseline_score) * 100
        
        c.execute("UPDATE aggregate_results SET improvement_over_baseline = ? WHERE test_run_id = ? AND yellie_type = ?",
                  (improvement, test_run_id, yellie_type))
    
    conn.commit()
    
    # Display final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    c.execute("""SELECT yellie_type, avg_score, improvement_over_baseline, valid_json_rate,
                 urgency_accuracy, sentiment_accuracy, avg_categories_score
                 FROM aggregate_results WHERE test_run_id = ? ORDER BY avg_score DESC""",
              (test_run_id,))
    
    results = c.fetchall()
    
    print(f"\n{'Yellie Type':<20} {'Avg Score':<12} {'vs Baseline':<12} {'Valid JSON':<12} {'Urgency Acc':<12} {'Sentiment Acc':<12} {'Cat Score':<12}")
    print("-" * 104)
    
    for r in results:
        yellie_type, avg_score, improvement, valid_json, urgency_acc, sentiment_acc, cat_score = r
        improvement_str = f"{improvement:+.2f}%" if yellie_type != 'baseline' else "---"
        print(f"{yellie_type:<20} {avg_score:.4f}       {improvement_str:<12} {valid_json:.2%}       {urgency_acc:.2%}        {sentiment_acc:.2%}          {cat_score:.4f}")
    
    print("\n" + "="*80)
    print(f"Test completed. Results saved to {db_name}")
    print(f"Test run ID: {test_run_id}")
    print(f"\nTo query results:")
    print(f"sqlite3 {db_name} \"SELECT * FROM aggregate_results WHERE test_run_id = {test_run_id}\"")
    print("="*80)
    
    # Close database
    conn.close()

if __name__ == "__main__":
    main()