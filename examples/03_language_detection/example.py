#!/usr/bin/env python3
"""
Language Detection Example
=========================

This example demonstrates how to use DFBI for automatic language detection
using character-level patterns and different alphabets.
"""

import numpy as np
from pathlib import Path
from dfbi import fingerprint
from dfbi.alphabet import RU41, EN34
from dfbi.metrics import dist_cosine, dist_l2
import yaml

def load_config():
    """Load configuration for language detection."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert decay lists to tuples for compatibility
    def convert_config_section(section):
        if isinstance(section, dict):
            for key, value in section.items():
                if key == 'decay' and isinstance(value, list):
                    section[key] = tuple(value)
                elif key == 'phase' and isinstance(value, list):
                    section[key] = tuple(value)
                elif isinstance(value, dict):
                    convert_config_section(value)
    
    for section in config.values():
        convert_config_section(section)
    
    return config

def create_language_samples():
    """Create sample texts in different languages."""
    samples = {
        'english': [
            "The quick brown fox jumps over the lazy dog. This is a sample English text for language detection.",
            "Machine learning and artificial intelligence are transforming the world of technology.",
            "Natural language processing enables computers to understand and generate human language."
        ],
        'russian': [
            "Быстрая коричневая лиса прыгает через ленивую собаку. Это образец русского текста для определения языка.",
            "Машинное обучение и искусственный интеллект преобразуют мир технологий.",
            "Обработка естественного языка позволяет компьютерам понимать и генерировать человеческий язык."
        ]
    }
    return samples

def build_language_profiles():
    """Build language profiles using training data."""
    print("=== Building Language Profiles ===")
    
    config = load_config()
    samples = create_language_samples()
    
    profiles = {}
    
    for language, texts in samples.items():
        print(f"Building profile for {language}...")
        
        # Determine alphabet
        alphabet = RU41 if language == 'russian' else EN34
        
        # Create fingerprints for all training texts
        fingerprints = []
        for text in texts[:2]:  # Use first 2 texts for training
            fp = fingerprint(
                text,
                alphabet=alphabet,
                **config['language_detection']
            )
            fingerprints.append(fp)
        
        # Average fingerprints to create language profile
        profile = np.mean(fingerprints, axis=0)
        profiles[language] = {
            'profile': profile,
            'alphabet': alphabet
        }
        
        print(f"  Profile shape: {profile.shape}")
        print(f"  Profile density: {np.count_nonzero(profile) / profile.size:.3f}")
    
    return profiles

def detect_language(text, profiles, config):
    """Detect language of given text using profiles."""
    best_language = None
    best_score = float('inf')
    
    for language, lang_data in profiles.items():
        # Create fingerprint using language-specific alphabet
        fp = fingerprint(
            text,
            alphabet=lang_data['alphabet'],
            **config['language_detection']
        )
        
        # Calculate distance to language profile
        distance = dist_cosine(fp, lang_data['profile'])
        
        if distance < best_score:
            best_score = distance
            best_language = language
    
    return best_language, best_score

def language_detection_demo():
    """Demonstrate language detection functionality."""
    print("\n=== Language Detection Demo ===")
    
    config = load_config()
    profiles = build_language_profiles()
    samples = create_language_samples()
    
    # Test on remaining samples
    correct_predictions = 0
    total_predictions = 0
    
    for true_language, texts in samples.items():
        test_text = texts[2]  # Use third text for testing
        
        predicted_language, confidence = detect_language(test_text, profiles, config)
        
        print(f"\nText: '{test_text[:60]}...'")
        print(f"True language: {true_language}")
        print(f"Predicted language: {predicted_language}")
        print(f"Confidence score: {1 - confidence:.4f}")
        
        if predicted_language == true_language:
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    print(f"\nAccuracy: {accuracy:.2%}")

def cross_alphabet_analysis():
    """Analyze how different alphabets affect language detection."""
    print("\n=== Cross-Alphabet Analysis ===")
    
    config = load_config()
    samples = create_language_samples()
    
    # Test English text with both alphabets
    english_text = samples['english'][0]
    russian_text = samples['russian'][0]
    
    print("English text analysis:")
    print(f"Text: '{english_text[:50]}...'")
    
    # Analyze with English alphabet
    fp_en_en = fingerprint(english_text, alphabet=EN34, **config['cross_alphabet'])
    print(f"With EN34 alphabet: density = {np.count_nonzero(fp_en_en) / fp_en_en.size:.3f}")
    
    # Analyze with Russian alphabet
    fp_en_ru = fingerprint(english_text, alphabet=RU41, **config['cross_alphabet'])
    print(f"With RU41 alphabet: density = {np.count_nonzero(fp_en_ru) / fp_en_ru.size:.3f}")
    
    print("\nRussian text analysis:")
    print(f"Text: '{russian_text[:50]}...'")
    
    # Analyze with Russian alphabet
    fp_ru_ru = fingerprint(russian_text, alphabet=RU41, **config['cross_alphabet'])
    print(f"With RU41 alphabet: density = {np.count_nonzero(fp_ru_ru) / fp_ru_ru.size:.3f}")
    
    # Analyze with English alphabet
    fp_ru_en = fingerprint(russian_text, alphabet=EN34, **config['cross_alphabet'])
    print(f"With EN34 alphabet: density = {np.count_nonzero(fp_ru_en) / fp_ru_en.size:.3f}")

def multilingual_comparison():
    """Compare texts across languages using unified approach."""
    print("\n=== Multilingual Comparison ===")
    
    config = load_config()
    samples = create_language_samples()
    
    # Use a common alphabet approach (EN34 for simplicity)
    print("Using unified EN34 alphabet for all languages:")
    
    all_fingerprints = []
    labels = []
    
    for language, texts in samples.items():
        for i, text in enumerate(texts):
            fp = fingerprint(
                text,
                alphabet=EN34,  # Use same alphabet for all
                **config['multilingual']
            )
            all_fingerprints.append(fp)
            labels.append(f"{language}_{i+1}")
    
    # Calculate pairwise distances
    print("\nPairwise cosine distances:")
    n = len(all_fingerprints)
    
    for i in range(n):
        for j in range(i+1, n):
            distance = dist_cosine(all_fingerprints[i], all_fingerprints[j])
            lang1 = labels[i].split('_')[0]
            lang2 = labels[j].split('_')[0]
            same_lang = lang1 == lang2
            
            print(f"{labels[i]:12s} vs {labels[j]:12s}: {distance:.4f} {'(same)' if same_lang else '(diff)'}")

def character_frequency_analysis():
    """Analyze character frequency patterns in different languages."""
    print("\n=== Character Frequency Analysis ===")
    
    samples = create_language_samples()
    
    for language, texts in samples.items():
        print(f"\n{language.upper()} character patterns:")
        
        # Combine all texts for this language
        combined_text = ' '.join(texts)
        
        # Count character frequencies
        char_counts = {}
        total_chars = 0
        
        for char in combined_text.lower():
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1
                total_chars += 1
        
        # Show top 10 most frequent characters
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 10 characters:")
        for char, count in sorted_chars[:10]:
            frequency = count / total_chars
            print(f"  {char}: {frequency:.3f}")

if __name__ == "__main__":
    profiles = build_language_profiles()
    language_detection_demo()
    cross_alphabet_analysis()
    multilingual_comparison()
    character_frequency_analysis()
    print("\n✓ Language detection example completed!")