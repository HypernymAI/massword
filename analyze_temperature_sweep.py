#!/usr/bin/env python3
"""
Analyze temperature sweep results to show consistent improvement
across all stochastic sampling paths WITH ERROR BARS
"""

import sqlite3
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

def analyze_sweep():
    # Collect all result databases
    db_files = glob.glob('temperature_sweep_results/*.db')
    
    results = defaultdict(dict)
    flipped_examples = defaultdict(set)
    
    for db_file in sorted(db_files):
        # Extract temperature from filename
        temp_str = db_file.split('_t')[1].split('.db')[0]
        temperature = float(temp_str) / 10.0
        
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        
        # Get baseline and urgency_keywords results WITH STATISTICS
        c.execute("""
            SELECT yellie_type, avg_score, std_score, urgency_accuracy, urgency_accuracy_std 
            FROM aggregate_results 
            WHERE yellie_type IN ('baseline', 'urgency_keywords')
        """)
        
        data = {}
        for row in c.fetchall():
            yellie_type, avg_score, std_score, urgency_acc, urgency_std = row
            data[yellie_type] = {
                'avg_score': avg_score,
                'std_score': std_score or 0,
                'urgency_acc': urgency_acc,
                'urgency_std': urgency_std or 0
            }
        
        if 'baseline' in data and 'urgency_keywords' in data:
            # Calculate improvements
            score_improvement = data['urgency_keywords']['avg_score'] - data['baseline']['avg_score']
            urgency_improvement = (data['urgency_keywords']['urgency_acc'] - 
                                 data['baseline']['urgency_acc']) * 100
            
            # Store results
            results[temperature] = {
                'baseline': data['baseline'],
                'urgency_keywords': data['urgency_keywords'],
                'score_improvement': score_improvement,
                'urgency_improvement': urgency_improvement
            }
            
            # Find which examples flipped (across all trials)
            c.execute("""
                SELECT DISTINCT CAST(SUBSTR(b.example_id, 1, INSTR(b.example_id, '_') - 1) AS INTEGER)
                FROM results b
                JOIN results u ON SUBSTR(b.example_id, 1, INSTR(b.example_id, '_') - 1) = 
                                  SUBSTR(u.example_id, 1, INSTR(u.example_id, '_') - 1)
                              AND b.trial_number = u.trial_number
                WHERE b.yellie_type = 'baseline' 
                  AND u.yellie_type = 'urgency_keywords'
                  AND b.urgency_predicted != b.urgency_actual
                  AND u.urgency_predicted = u.urgency_actual
            """)
            
            flipped = {row[0] for row in c.fetchall()}
            flipped_examples[temperature].update(flipped)
        
        conn.close()
    
    # Calculate statistics
    print("TEMPERATURE SWEEP ANALYSIS")
    print("=" * 80)
    print(f"{'Temp':>5} {'Baseline Score':>20} {'Keywords Score':>20} {'Urgency Improve':>18} {'# Examples':>12}")
    print("-" * 80)
    
    for temp in sorted(results.keys()):
        r = results[temp]
        baseline = r['baseline']
        keywords = r['urgency_keywords']
        
        baseline_str = f"{baseline['avg_score']:.3f} ± {baseline['std_score']:.3f}"
        keywords_str = f"{keywords['avg_score']:.3f} ± {keywords['std_score']:.3f}"
        urgency_imp = r['urgency_improvement']
        unique_examples = len(flipped_examples[temp])
        
        print(f"{temp:5.1f} {baseline_str:>20} {keywords_str:>20} {urgency_imp:>16.2f}% {unique_examples:>12d}")
    
    # Show example overlap analysis
    print("\nEXAMPLE OVERLAP ANALYSIS")
    print("=" * 80)
    
    if 0.0 in flipped_examples:
        deterministic_examples = flipped_examples[0.0]
        print(f"Deterministic examples (temp=0.0): {sorted(list(deterministic_examples))[:20]}")
        if len(deterministic_examples) > 20:
            print(f"... and {len(deterministic_examples) - 20} more")
        print(f"Total at temp=0.0: {len(deterministic_examples)}")
        
        for temp in sorted(results.keys()):
            if temp > 0:
                overlap = deterministic_examples & flipped_examples[temp]
                unique = flipped_examples[temp] - deterministic_examples
                print(f"\nTemp={temp}:")
                print(f"  Overlap with temp=0: {len(overlap)} examples ({len(overlap)/len(deterministic_examples)*100:.1f}%)")
                print(f"  Unique to this temp: {len(unique)} examples")
                print(f"  Total examples: {len(flipped_examples[temp])}")
                if len(unique) > 0:
                    print(f"  New examples: {sorted(list(unique))[:10]}")
                    if len(unique) > 10:
                        print(f"  ... and {len(unique) - 10} more")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Score Improvement vs Temperature (with error bars)
    temps = sorted(results.keys())
    baseline_scores = [results[t]['baseline']['avg_score'] for t in temps]
    baseline_stds = [results[t]['baseline']['std_score'] for t in temps]
    keywords_scores = [results[t]['urgency_keywords']['avg_score'] for t in temps]
    keywords_stds = [results[t]['urgency_keywords']['std_score'] for t in temps]
    
    ax1.errorbar(temps, baseline_scores, yerr=baseline_stds, marker='o', 
                label='Baseline', linewidth=2, capsize=5)
    ax1.errorbar(temps, keywords_scores, yerr=keywords_stds, marker='s', 
                label='With Keywords', linewidth=2, capsize=5)
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Overall Score')
    ax1.set_title('Overall Score vs Temperature (with error bars)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Urgency Improvement vs Temperature
    urgency_improvements = [results[t]['urgency_improvement'] for t in temps]
    urgency_baseline = [results[t]['baseline']['urgency_acc'] * 100 for t in temps]
    urgency_keywords = [results[t]['urgency_keywords']['urgency_acc'] * 100 for t in temps]
    
    ax2.plot(temps, urgency_baseline, marker='o', label='Baseline', linewidth=2)
    ax2.plot(temps, urgency_keywords, marker='s', label='With Keywords', linewidth=2)
    ax2.axhline(y=92.46, color='r', linestyle='--', alpha=0.5, label='Deterministic (92.46%)')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Urgency Accuracy (%)')
    ax2.set_title('Urgency Detection Accuracy vs Temperature')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Number of unique examples flipped
    unique_counts = [len(flipped_examples[t]) for t in temps]
    ax3.plot(temps, unique_counts, marker='D', linewidth=2, color='green')
    ax3.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Deterministic count')
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('Number of Unique Examples Improved')
    ax3.set_title('Diversity of Improved Examples vs Temperature')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Improvement consistency
    improvements = [results[t]['urgency_improvement'] for t in temps]
    ax4.bar(temps, improvements, width=0.08, alpha=0.7, color='orange')
    ax4.axhline(y=2.01, color='r', linestyle='--', label='Deterministic improvement')
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Urgency Improvement (%)')
    ax4.set_title('Urgency Improvement Consistency Across Temperatures')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('temperature_sweep_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as temperature_sweep_analysis.png")
    
    # Statistical summary
    print("\nSTATISTICAL SUMMARY")
    print("=" * 80)
    improvements = [results[t]['urgency_improvement'] for t in temps]
    print(f"Mean urgency improvement across temperatures: {np.mean(improvements):.2f}%")
    print(f"Std dev of improvements: {np.std(improvements):.2f}%")
    print(f"Range: {np.min(improvements):.2f}% to {np.max(improvements):.2f}%")
    
    # Check if improvement is statistically significant at each temperature
    print("\nStatistical Significance (approximate z-test):")
    for temp in sorted(results.keys()):
        r = results[temp]
        baseline_acc = r['baseline']['urgency_acc']
        keywords_acc = r['urgency_keywords']['urgency_acc']
        baseline_std = r['baseline']['urgency_std']
        keywords_std = r['urgency_keywords']['urgency_std']
        
        # Approximate z-score (assuming independent samples)
        if baseline_std > 0 or keywords_std > 0:
            se = np.sqrt(baseline_std**2 + keywords_std**2)
            if se > 0:
                z = (keywords_acc - baseline_acc) / se
                print(f"  Temp {temp}: z = {z:.2f} ({'significant' if abs(z) > 1.96 else 'not significant'} at p<0.05)")

if __name__ == "__main__":
    analyze_sweep()