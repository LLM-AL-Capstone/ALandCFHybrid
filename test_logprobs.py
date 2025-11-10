#!/usr/bin/env python3
"""
Test script to demonstrate OpenAI logprobs for better uncertainty estimation.

This shows the difference between:
1. Old approach: Heuristic (0.9 for predicted, 0.1 distributed)
2. New approach: Actual probabilities from OpenAI logprobs
"""

from utils import load_config, get_llm_provider, SimpleICLClassifier
import numpy as np

def test_logprobs():
    """Test logprobs functionality with a simple example."""
    
    print("\n" + "="*70)
    print("Testing OpenAI Logprobs for Uncertainty Estimation")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Check if using OpenAI
    if config['llm']['provider'] != 'openai':
        print("\nWARNING: This test requires OpenAI provider.")
        print("Current provider:", config['llm']['provider'])
        print("Change provider to 'openai' in config.yaml to use logprobs.")
        return
    
    print(f"\nProvider: {config['llm']['provider']}")
    print(f"Model: {config['llm']['openai']['model']}")
    
    # Initialize LLM and classifier
    llm_provider = get_llm_provider(config)
    classifier = SimpleICLClassifier(config, llm_provider)
    
    # Create simple training examples
    print("\n=== Creating Training Set ===")
    training_examples = [
        {'id': 1, 'text': 'I love this!', 'label': 'joy'},
        {'id': 2, 'text': 'This is amazing!', 'label': 'joy'},
        {'id': 3, 'text': 'I am so sad', 'label': 'sadness'},
        {'id': 4, 'text': 'This makes me cry', 'label': 'sadness'},
        {'id': 5, 'text': 'I am furious!', 'label': 'anger'},
        {'id': 6, 'text': 'This is infuriating', 'label': 'anger'},
    ]
    
    print(f"Training with {len(training_examples)} examples")
    for ex in training_examples:
        print(f"  - '{ex['text']}' → {ex['label']}")
    
    # Train classifier
    classifier.train(training_examples)
    
    # Test examples with varying levels of clarity
    print("\n=== Testing Uncertainty Estimation ===")
    test_examples = [
        "This is wonderful!",  # Clear joy
        "I'm somewhat happy, I guess",  # Uncertain joy
        "not sure how I feel about this",  # Very uncertain
        "I hate this so much!",  # Clear anger
    ]
    
    print("\nComputing probabilities with logprobs...")
    print("(This will make API calls, please wait...)\n")
    
    # Get probabilities
    probs = classifier.predict_proba(test_examples)
    
    # Display results
    labels = classifier.get_labels()
    
    print("\n" + "="*70)
    print("Results: Probability Distributions")
    print("="*70)
    
    for i, text in enumerate(test_examples):
        print(f"\nExample {i+1}: '{text}'")
        print("-" * 70)
        
        # Show probabilities for each label
        for j, label in enumerate(labels):
            prob = probs[i][j]
            bar = "█" * int(prob * 50)  # Visual bar
            print(f"  {label:12s}: {prob:.3f} {bar}")
        
        # Calculate and show entropy (uncertainty measure)
        entropy = -np.sum(probs[i] * np.log(probs[i] + 1e-10))
        max_entropy = np.log(len(labels))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy
        
        print(f"\n  Entropy: {entropy:.3f} (normalized: {normalized_entropy:.3f})")
        
        if normalized_entropy > 0.8:
            print("  → VERY UNCERTAIN (good AL candidate!)")
        elif normalized_entropy > 0.5:
            print("  → MODERATELY UNCERTAIN")
        else:
            print("  → CONFIDENT prediction")
    
    print("\n" + "="*70)
    print("Interpretation")
    print("="*70)
    print("""
High entropy (close to 1.0):
  - Model is very uncertain
  - Probabilities spread across multiple labels
  - IDEAL for Active Learning selection!
  - Example: "not sure how I feel" → might be 0.4, 0.35, 0.25

Low entropy (close to 0.0):
  - Model is very confident
  - One label has high probability
  - NOT good for Active Learning (already knows the answer)
  - Example: "I love this!" → might be 0.95, 0.03, 0.02

With OpenAI logprobs, we get REAL probability distributions instead of
the old 0.9/0.1 heuristic, leading to much better uncertainty estimates
and more effective Active Learning!
    """)
    
    print("="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_logprobs()

