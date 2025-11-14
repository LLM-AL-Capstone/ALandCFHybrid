#!/usr/bin/env python3
"""
Test script for retrieval-based ICL classifier.

Validates:
1. Configuration loading
2. Static classifier (backward compatibility)
3. Retrieval classifier initialization
4. Both classifier types can train and predict
"""

import sys
from utils import load_config, get_llm_provider
from utils.classifier import SimpleICLClassifier
from utils.retrieval_classifier import get_retrieval_classifier

def test_config_loading():
    """Test that config loads with retrieval settings."""
    print("\n" + "="*80)
    print("TEST 1: Configuration Loading")
    print("="*80)
    
    config = load_config()
    
    # Check evaluation config
    eval_config = config['evaluation']
    print(f"✓ Config loaded successfully")
    print(f"  Classifier type: {eval_config.get('classifier_type', 'static')}")
    
    if 'retrieval' in eval_config:
        retrieval_config = eval_config['retrieval']
        print(f"  Retrieval backend: {retrieval_config.get('embedding_backend', 'N/A')}")
        print(f"  k_per_class: {retrieval_config.get('k_per_class', 'N/A')}")
        print(f"  total_k_max: {retrieval_config.get('total_k_max', 'N/A')}")
        print("✓ Retrieval configuration found")
    else:
        print("⚠ No retrieval configuration (will use defaults)")
    
    return config


def test_static_classifier(config):
    """Test static ICL classifier (backward compatibility)."""
    print("\n" + "="*80)
    print("TEST 2: Static ICL Classifier")
    print("="*80)
    
    try:
        llm_provider = get_llm_provider(config)
        classifier = SimpleICLClassifier(config, llm_provider)
        
        # Create mock labeled pool
        labeled_pool = [
            {'text': 'I am happy', 'label': 'joy'},
            {'text': 'I am sad', 'label': 'sadness'},
            {'text': 'I am angry', 'label': 'anger'},
        ]
        
        classifier.train(labeled_pool)
        print("✓ Static classifier trained successfully")
        
        # Test prediction
        prediction = classifier.predict("I feel great today")
        print(f"✓ Static classifier prediction: '{prediction}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Static classifier test failed: {e}")
        return False


def test_retrieval_classifier(config, backend='tfidf'):
    """Test retrieval-based ICL classifier."""
    print("\n" + "="*80)
    print(f"TEST 3: Retrieval ICL Classifier ({backend})")
    print("="*80)
    
    try:
        # Temporarily modify config for this test
        original_backend = config['evaluation']['retrieval']['embedding_backend']
        config['evaluation']['retrieval']['embedding_backend'] = backend
        
        llm_provider = get_llm_provider(config)
        classifier = get_retrieval_classifier(config, llm_provider)
        
        print(f"✓ Retrieval classifier created with {backend} backend")
        
        # Create mock labeled pool (larger for retrieval)
        labeled_pool = [
            {'text': 'I am very happy today', 'label': 'joy'},
            {'text': 'This makes me feel great', 'label': 'joy'},
            {'text': 'I am so sad', 'label': 'sadness'},
            {'text': 'This is depressing', 'label': 'sadness'},
            {'text': 'I am furious', 'label': 'anger'},
            {'text': 'This makes me angry', 'label': 'anger'},
            {'text': 'I love this', 'label': 'love'},
            {'text': 'This is wonderful', 'label': 'love'},
        ]
        
        classifier.train(labeled_pool)
        print("✓ Retrieval classifier trained and encoded examples")
        
        # Test prediction
        prediction = classifier.predict("I feel great today")
        print(f"✓ Retrieval classifier prediction: '{prediction}'")
        
        # Restore original backend
        config['evaluation']['retrieval']['embedding_backend'] = original_backend
        
        return True
        
    except Exception as e:
        print(f"✗ Retrieval classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_classifier_switching(config):
    """Test switching between classifier types via config."""
    print("\n" + "="*80)
    print("TEST 4: Classifier Type Switching")
    print("="*80)
    
    llm_provider = get_llm_provider(config)
    
    # Test with static
    original_type = config['evaluation'].get('classifier_type', 'static')
    config['evaluation']['classifier_type'] = 'static'
    
    if config['evaluation']['classifier_type'] == 'static':
        classifier = SimpleICLClassifier(config, llm_provider)
        print("✓ Successfully switched to static classifier")
    
    # Test with retrieval (using TF-IDF for speed)
    config['evaluation']['classifier_type'] = 'retrieval'
    config['evaluation']['retrieval']['embedding_backend'] = 'tfidf'
    
    if config['evaluation']['classifier_type'] == 'retrieval':
        classifier = get_retrieval_classifier(config, llm_provider)
        print("✓ Successfully switched to retrieval classifier")
    
    # Restore original
    config['evaluation']['classifier_type'] = original_type
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RETRIEVAL-BASED ICL CLASSIFIER TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Test 1: Config loading
    try:
        config = test_config_loading()
        results['config'] = True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        results['config'] = False
        return
    
    # Test 2: Static classifier (backward compatibility)
    results['static'] = test_static_classifier(config)
    
    # Test 3: Retrieval classifier (TF-IDF - no extra dependencies)
    results['retrieval_tfidf'] = test_retrieval_classifier(config, backend='tfidf')
    
    # Test 4: Classifier switching
    results['switching'] = test_classifier_switching(config)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Retrieval-based ICL is ready to use.")
        print("\nTo use retrieval-based ICL, set in config.yaml:")
        print("  evaluation:")
        print("    classifier_type: 'retrieval'")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

