#!/bin/bash
# Temperature Sweep Test Runner WITH CACHING
# Tests urgency keyword intervention across temperature range
# Uses caching to avoid repeated API calls

echo "=========================================="
echo "TEMPERATURE SWEEP STOCHASTIC INTERVENTION"
echo "WITH EVALUATION CACHING"
echo "=========================================="
echo "Date: $(date)"
echo "Testing samples: 200"
echo "Trials per example: 50"
echo ""

# Create results directory
mkdir -p temperature_sweep_results

# Initialize cache database
echo "Cache database: temperature_sweep_cache.db"
echo ""

# Test each temperature
for temp in 0.0 0.3 0.5 0.7 1.0; do
    echo ""
    echo "Testing temperature: $temp"
    echo "-----------------------------------------"
    
    python test_facility_temperature_sweep_cached.py \
        --samples 200 \
        --trials 50 \
        --workers 10 \
        --temperature $temp
    
    # Move database to results directory
    mv facility_temperature_sweep_*_t*.db temperature_sweep_results/ 2>/dev/null
    
    # Brief pause between runs
    sleep 5
done

echo ""
echo "Temperature sweep complete!"
echo "Results saved in temperature_sweep_results/"
echo "Cache saved in temperature_sweep_cache.db"
echo ""
echo "Run analysis with: python analyze_temperature_sweep.py"