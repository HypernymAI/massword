#!/usr/bin/env python3
"""
Facility Support Temperature Sweep Test WITH CACHING
Tests urgency keyword intervention across different temperature settings
Caches all evaluation results to avoid repeated API calls
"""
import os
import sys
import json
import yaml
import sqlite3
import time
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple, Any, Optional
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

# Global cache database path
CACHE_DB_PATH = "temperature_sweep_cache.db"

def init_cache_db():
    """Initialize the cache database"""
    conn = sqlite3.connect(CACHE_DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS evaluation_cache (
        cache_key TEXT PRIMARY KEY,
        model_name TEXT,
        temperature REAL,
        system_prompt TEXT,
        user_prompt TEXT,
        trial_number INTEGER,
        result_json TEXT,
        timestamp TEXT,
        example_id TEXT,
        yellie_type TEXT
    )''')
    
    # Create indexes for common queries
    c.execute('CREATE INDEX IF NOT EXISTS idx_model_temp ON evaluation_cache(model_name, temperature)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trial ON evaluation_cache(trial_number)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluation_cache(timestamp)')
    
    conn.commit()
    conn.close()

def get_cache_key(model_name: str, temperature: float, system_prompt: str, 
                  user_prompt: str, trial_number: int) -> str:
    """Generate a cache key from the input parameters"""
    # Create a tuple and hash it for the key
    cache_tuple = (model_name, temperature, system_prompt, user_prompt, trial_number)
    cache_str = json.dumps(cache_tuple, sort_keys=True)
    return hashlib.sha256(cache_str.encode()).hexdigest()

def get_from_cache(model_name: str, temperature: float, system_prompt: str,
                   user_prompt: str, trial_number: int) -> Optional[Dict]:
    """Retrieve a cached result if it exists"""
    cache_key = get_cache_key(model_name, temperature, system_prompt, user_prompt, trial_number)
    
    conn = sqlite3.connect(CACHE_DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT result_json FROM evaluation_cache WHERE cache_key = ?''', (cache_key,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return json.loads(row[0])
    return None

def save_to_cache(model_name: str, temperature: float, system_prompt: str,
                  user_prompt: str, trial_number: int, result: Dict,
                  example_id: str, yellie_type: str):
    """Save a result to the cache"""
    cache_key = get_cache_key(model_name, temperature, system_prompt, user_prompt, trial_number)
    
    conn = sqlite3.connect(CACHE_DB_PATH)
    c = conn.cursor()
    
    c.execute('''INSERT OR REPLACE INTO evaluation_cache 
                 (cache_key, model_name, temperature, system_prompt, user_prompt, 
                  trial_number, result_json, timestamp, example_id, yellie_type)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (cache_key, model_name, temperature, system_prompt, user_prompt,
               trial_number, json.dumps(result), datetime.now().isoformat(),
               example_id, yellie_type))
    
    conn.commit()
    conn.close()

def setup_database(samples, runs, temperature):
    """Create database for storing detailed results."""
    # Create timestamped database name with test parameters INCLUDING TEMPERATURE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Format temperature for filename (0.3 becomes 03, 1.0 becomes 10)
    temp_str = f"{int(temperature * 10):02d}"
    db_name = f'facility_temperature_sweep_{timestamp}_s{samples}_r{runs}_t{temp_str}.db'
    print(f"Creating database: {db_name}")
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS test_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        samples_per_test INTEGER,
        total_samples INTEGER,
        temperature REAL,
        cache_hits INTEGER DEFAULT 0,
        cache_misses INTEGER DEFAULT 0
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_run_id INTEGER,
        yellie_type TEXT,
        example_id INTEGER,
        trial_number INTEGER,
        metric_score REAL,
        urgency_predicted TEXT,
        urgency_actual TEXT,
        sentiment_predicted TEXT,
        sentiment_actual TEXT,
        categories_score REAL,
        valid_json BOOLEAN,
        response_length INTEGER,
        error_message TEXT,
        from_cache BOOLEAN DEFAULT 0,
        FOREIGN KEY(test_run_id) REFERENCES test_runs(id)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS aggregate_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_run_id INTEGER,
        yellie_type TEXT,
        avg_score REAL,
        std_score REAL,
        valid_json_rate REAL,
        urgency_accuracy REAL,
        urgency_accuracy_std REAL,
        sentiment_accuracy REAL,
        avg_categories_score REAL,
        total_examples INTEGER,
        improvement_over_baseline REAL,
        FOREIGN KEY(test_run_id) REFERENCES test_runs(id)
    )''')
    
    conn.commit()
    return conn, db_name

def evaluate_single_example(program, example, yellie_type, example_id, metric, 
                          model_name, temperature, trial_number, test_run_id, db_name):
    """Evaluate a single example with caching support."""
    # Extract the actual instruction from the program
    instruction = program.predictor.signature.instructions
    
    # Build the actual prompts that will be sent
    system_prompt = instruction
    user_prompt = f"Question: {example['question']}\n\nAnswer:"
    
    # Check cache first
    cached_result = get_from_cache(model_name, temperature, system_prompt, user_prompt, trial_number)
    
    if cached_result:
        # Update cache hit counter
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute("UPDATE test_runs SET cache_hits = cache_hits + 1 WHERE id = ?", (test_run_id,))
        conn.commit()
        conn.close()
        
        # Mark result as from cache
        cached_result['from_cache'] = True
        return cached_result
    
    # Update cache miss counter
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("UPDATE test_runs SET cache_misses = cache_misses + 1 WHERE id = ?", (test_run_id,))
    conn.commit()
    conn.close()
    
    # Not in cache, need to actually evaluate
    max_retries = 3
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
                    'trial_number': trial_number,
                    'metric_score': metric_result['total'],
                    'urgency_predicted': parsed.get('urgency', ''),
                    'urgency_actual': ground_truth.get('urgency', ''),
                    'sentiment_predicted': parsed.get('sentiment', ''),
                    'sentiment_actual': ground_truth.get('sentiment', ''),
                    'categories_score': metric_result.get('correct_categories', 0),
                    'valid_json': valid_json,
                    'response_length': len(response),
                    'error_message': None,
                    'from_cache': False
                }
                
                # Save to cache
                save_to_cache(model_name, temperature, system_prompt, user_prompt, 
                            trial_number, result, example_id, yellie_type)
                
                return result
                
            except json.JSONDecodeError as e:
                # Invalid JSON result
                result = {
                    'yellie_type': yellie_type,
                    'example_id': example_id,
                    'trial_number': trial_number,
                    'metric_score': 0.0,
                    'urgency_predicted': '',
                    'urgency_actual': json.loads(example['answer']).get('urgency', ''),
                    'sentiment_predicted': '',
                    'sentiment_actual': json.loads(example['answer']).get('sentiment', ''),
                    'categories_score': 0.0,
                    'valid_json': False,
                    'response_length': len(response),
                    'error_message': f"Invalid JSON: {str(e)}",
                    'from_cache': False
                }
                
                # Save even failed results to cache
                save_to_cache(model_name, temperature, system_prompt, user_prompt,
                            trial_number, result, example_id, yellie_type)
                
                return result
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Error on attempt {attempt + 1}, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                result = {
                    'yellie_type': yellie_type,
                    'example_id': example_id,
                    'trial_number': trial_number,
                    'metric_score': 0.0,
                    'urgency_predicted': '',
                    'urgency_actual': '',
                    'sentiment_predicted': '',
                    'sentiment_actual': '',
                    'categories_score': 0.0,
                    'valid_json': False,
                    'response_length': 0,
                    'error_message': f"Failed after {max_retries} attempts: {str(e)}",
                    'from_cache': False
                }
                
                # Save error to cache too
                save_to_cache(model_name, temperature, system_prompt, user_prompt,
                            trial_number, result, example_id, yellie_type)
                
                return result

def evaluate_yellie_variant(yellie_type, instruction, valset, metric, model_name, 
                          temperature, test_run_id, db_name, max_workers=5, 
                          num_trials=50):
    """Evaluate a single yellie variant with multiple trials for statistics."""
    print(f"\n{'='*60}")
    print(f"Testing: {yellie_type.upper()}")
    print(f"Instruction length: {len(instruction)} chars")
    print(f"Examples: {len(valset)}, Trials per example: {num_trials}")
    print(f"Total evaluations: {len(valset) * num_trials}")
    print(f"Temperature: {temperature}")
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
    cache_hits = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks (multiple trials per example)
        future_to_example = {}
        for i, example in enumerate(valset):
            for trial in range(num_trials):
                # Create unique ID for this evaluation
                eval_id = f"{i}_trial{trial}"
                future = executor.submit(evaluate_single_example, program, example, 
                                       yellie_type, eval_id, metric, model_name,
                                       temperature, trial, test_run_id, db_name)
                future_to_example[future] = (i, example, trial)
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_example):
            completed += 1
            example_id, example, trial_num = future_to_example[future]
            
            try:
                result = future.result()
                # Store original example_id for aggregation
                result['original_example_id'] = example_id
                results.append(result)
                
                if result.get('from_cache', False):
                    cache_hits += 1
                
                total_tasks = len(valset) * num_trials
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_tasks - completed) / rate if rate > 0 else 0
                    cache_rate = cache_hits / completed * 100
                    print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) - "
                          f"Rate: {rate:.1f} ex/s - ETA: {eta:.0f}s - Cache: {cache_rate:.1f}%")
                    
            except Exception as e:
                print(f"Error processing example {example_id}: {str(e)}")
                results.append({
                    'yellie_type': yellie_type,
                    'example_id': example_id,
                    'trial_number': trial_num,
                    'metric_score': 0.0,
                    'valid_json': False,
                    'error_message': str(e),
                    'from_cache': False
                })
    
    total_time = time.time() - start_time
    print(f"\nCompleted {yellie_type} in {total_time:.1f}s ({len(results)/total_time:.1f} ex/s)")
    print(f"Cache hits: {cache_hits}/{len(results)} ({cache_hits/len(results)*100:.1f}%)")
    
    return results

def main():
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Facility Support Temperature Sweep Test with Caching')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples per variant (default: 50)')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials per example for statistics (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for model sampling (default: 0.0)')
    args = parser.parse_args()
    
    print("="*80)
    print("FACILITY SUPPORT TEMPERATURE SWEEP TEST (WITH CACHING)")
    print("="*80)
    print(f"Temperature: {args.temperature}")
    print(f"Workers: {args.workers}")
    print(f"Samples per variant: {args.samples}")
    print(f"Trials per example: {args.trials}")
    print(f"Total evaluations per variant: {args.samples * args.trials}")
    print(f"Cache database: {CACHE_DB_PATH}")
    print(f"Test started: {datetime.now()}")
    
    # Initialize cache database
    init_cache_db()
    
    # Setup results database WITH TEMPERATURE
    conn, db_name = setup_database(args.samples, args.trials, args.temperature)
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
    
    # Configure DSPy WITH TEMPERATURE PARAMETER
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
        temperature=args.temperature,  # USE COMMAND-LINE TEMPERATURE
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
    print(f"Temperature setting: {args.temperature}")
    
    # Create test run WITH TEMPERATURE
    c.execute("INSERT INTO test_runs (timestamp, samples_per_test, total_samples, temperature) VALUES (?, ?, ?, ?)",
              (datetime.now().isoformat(), len(valset), len(valset) * 2 * args.trials, args.temperature))
    test_run_id = c.lastrowid
    conn.commit()
    
    # Test only baseline and urgency_keywords for temperature sweep
    yellie_variants = [
        ("baseline", instruction_only),
        ("urgency_keywords", FacilityYellies.urgency_keywords(instruction_only)),
    ]
    
    # Store all results
    all_results = {}
    
    # Test each variant
    for yellie_type, instruction in yellie_variants:
        results = evaluate_yellie_variant(yellie_type, instruction, valset, metric,
                                        model_name, args.temperature, test_run_id, db_name,
                                        max_workers=args.workers, num_trials=args.trials)
        all_results[yellie_type] = results
        
        # Save to database
        for r in results:
            c.execute("""INSERT INTO results 
                (test_run_id, yellie_type, example_id, trial_number, metric_score, 
                 urgency_predicted, urgency_actual, sentiment_predicted, sentiment_actual, 
                 categories_score, valid_json, response_length, error_message, from_cache)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (test_run_id, r['yellie_type'], r['example_id'], r.get('trial_number', 0),
                 r['metric_score'], r.get('urgency_predicted', ''), r.get('urgency_actual', ''),
                 r.get('sentiment_predicted', ''), r.get('sentiment_actual', ''),
                 r.get('categories_score', 0), r['valid_json'], 
                 r.get('response_length', 0), r.get('error_message'),
                 r.get('from_cache', 0)))
        
        # Calculate aggregates WITH STATISTICS
        valid_results = [r for r in results if r.get('error_message') is None]
        if valid_results:
            # Group by original example to calculate per-example statistics
            from collections import defaultdict
            example_scores = defaultdict(list)
            example_urgency = defaultdict(list)
            
            for r in valid_results:
                orig_id = r.get('original_example_id', 0)
                example_scores[orig_id].append(r['metric_score'])
                
                # Track urgency accuracy per example
                urgency_correct = r.get('urgency_predicted') == r.get('urgency_actual')
                example_urgency[orig_id].append(urgency_correct)
            
            # Calculate overall statistics
            all_scores = [r['metric_score'] for r in valid_results]
            avg_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            
            valid_json_rate = sum(1 for r in valid_results if r['valid_json']) / len(valid_results)
            
            # Component accuracies with variance
            urgency_accuracies = []
            for ex_id, urgency_results in example_urgency.items():
                urgency_accuracies.append(np.mean(urgency_results))
            
            urgency_acc = np.mean(urgency_accuracies)
            urgency_acc_std = np.std(urgency_accuracies)
            
            sentiment_correct = sum(1 for r in valid_results 
                                  if r.get('sentiment_predicted') == r.get('sentiment_actual'))
            sentiment_acc = sentiment_correct / len(valid_results) if valid_results else 0
            
            avg_cat_score = sum(r.get('categories_score', 0) for r in valid_results) / len(valid_results)
            
            # Save aggregate with statistics
            c.execute("""INSERT INTO aggregate_results 
                (test_run_id, yellie_type, avg_score, std_score, valid_json_rate, 
                 urgency_accuracy, urgency_accuracy_std, sentiment_accuracy, 
                 avg_categories_score, total_examples, improvement_over_baseline)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (test_run_id, yellie_type, avg_score, std_score, valid_json_rate,
                 urgency_acc, urgency_acc_std, sentiment_acc, avg_cat_score, 
                 len(valid_results), 0))
        
        conn.commit()
    
    # Update improvement over baseline
    c.execute("SELECT avg_score, urgency_accuracy FROM aggregate_results WHERE test_run_id = ? AND yellie_type = 'baseline'",
              (test_run_id,))
    baseline_result = c.fetchone()
    
    if baseline_result:
        baseline_score, baseline_urgency = baseline_result
        
        c.execute("SELECT avg_score, urgency_accuracy FROM aggregate_results WHERE test_run_id = ? AND yellie_type = 'urgency_keywords'",
                  (test_run_id,))
        urgency_result = c.fetchone()
        
        if urgency_result:
            urgency_score, urgency_acc = urgency_result
            improvement = (urgency_score - baseline_score) * 100
            urgency_improvement = (urgency_acc - baseline_urgency) * 100
            
            c.execute("UPDATE aggregate_results SET improvement_over_baseline = ? WHERE test_run_id = ? AND yellie_type = 'urgency_keywords'",
                      (improvement, test_run_id))
    
    conn.commit()
    
    # Display final results WITH ERROR BARS
    print("\n" + "="*80)
    print(f"TEMPERATURE {args.temperature} RESULTS (WITH STATISTICS)")
    print("="*80)
    
    c.execute("""SELECT yellie_type, avg_score, std_score, improvement_over_baseline, 
                        urgency_accuracy, urgency_accuracy_std
                 FROM aggregate_results WHERE test_run_id = ? ORDER BY yellie_type""",
              (test_run_id,))
    
    results = c.fetchall()
    
    print(f"\n{'Variant':<20} {'Avg Score':<20} {'Urgency Acc':<20} {'vs Baseline':<15}")
    print("-" * 75)
    
    for r in results:
        yellie_type, avg_score, std_score, improvement, urgency_acc, urgency_std = r
        score_str = f"{avg_score:.4f} ± {std_score:.4f}"
        urgency_str = f"{urgency_acc:.2%} ± {urgency_std:.2%}"
        improvement_str = f"{improvement:+.2f}%" if yellie_type != 'baseline' else "---"
        print(f"{yellie_type:<20} {score_str:<20} {urgency_str:<20} {improvement_str:<15}")
    
    # Find which examples flipped (with counts across trials)
    c.execute("""
        SELECT 
            CAST(SUBSTR(b.example_id, 1, INSTR(b.example_id, '_') - 1) AS INTEGER) as ex_num,
            COUNT(DISTINCT b.trial_number) as baseline_wrong_count,
            COUNT(DISTINCT u.trial_number) as urgency_correct_count
        FROM results b
        JOIN results u ON SUBSTR(b.example_id, 1, INSTR(b.example_id, '_') - 1) = 
                          SUBSTR(u.example_id, 1, INSTR(u.example_id, '_') - 1)
                      AND b.trial_number = u.trial_number
        WHERE b.test_run_id = ? AND u.test_run_id = ?
          AND b.yellie_type = 'baseline' 
          AND u.yellie_type = 'urgency_keywords'
          AND b.urgency_predicted != b.urgency_actual
          AND u.urgency_predicted = u.urgency_actual
        GROUP BY ex_num
        ORDER BY ex_num
    """, (test_run_id, test_run_id))
    
    flipped_examples = [(row[0], row[1], row[2]) for row in c.fetchall()]
    
    print(f"\nExamples that improved with urgency keywords:")
    print(f"{'Example':<10} {'Baseline Wrong':<15} {'Keywords Correct':<17}")
    print("-" * 42)
    for ex_num, baseline_wrong, keywords_correct in flipped_examples[:10]:  # Show first 10
        print(f"{ex_num:<10} {baseline_wrong}/{args.trials:<15} {keywords_correct}/{args.trials:<17}")
    
    if len(flipped_examples) > 10:
        print(f"... and {len(flipped_examples) - 10} more examples")
    
    print(f"\nTotal examples that improved: {len(flipped_examples)}")
    
    # Get cache statistics
    c.execute("SELECT cache_hits, cache_misses FROM test_runs WHERE id = ?", (test_run_id,))
    cache_hits, cache_misses = c.fetchone()
    total_calls = cache_hits + cache_misses
    cache_rate = cache_hits / total_calls * 100 if total_calls > 0 else 0
    
    print("\n" + "="*80)
    print(f"Test completed. Results saved to {db_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Test run ID: {test_run_id}")
    print(f"Cache performance: {cache_hits}/{total_calls} hits ({cache_rate:.1f}%)")
    print("="*80)
    
    # Close database
    conn.close()

if __name__ == "__main__":
    main()