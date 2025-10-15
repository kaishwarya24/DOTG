"""
Advanced Evaluation Metrics for DOTG Framework
Simplified version without external ML dependencies
"""

import re
import math
from typing import Dict, List, Any
from collections import Counter


# ============ CLARITY SCORE ============

def calculate_clarity(question_text: str) -> Dict[str, float]:
    """
    Measure question clarity and comprehensibility
    """
    # Remove extra whitespace
    question_text = ' '.join(question_text.split())
    
    # Component 1: Length appropriateness (optimal 10-25 words)
    words = question_text.split()
    word_count = len(words)
    
    if 10 <= word_count <= 25:
        length_score = 1.0
    elif 5 <= word_count < 10 or 25 < word_count <= 35:
        length_score = 0.7
    else:
        length_score = 0.4
    
    # Component 2: Question mark presence
    has_question_mark = '?' in question_text
    question_mark_score = 1.0 if has_question_mark else 0.3
    
    # Component 3: Multiple questions (reduces clarity)
    question_count = question_text.count('?')
    if question_count == 1:
        single_question_score = 1.0
    elif question_count == 0:
        single_question_score = 0.5  # Might be statement form
    else:
        single_question_score = 0.3  # Multiple questions confusing
    
    # Component 4: Ambiguous words (reduce clarity)
    ambiguous_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 
                       'some', 'somewhat', 'kind of', 'sort of', 'various']
    ambiguous_count = sum(1 for word in ambiguous_words if word in question_text.lower())
    ambiguity_penalty = max(0, 1.0 - (ambiguous_count * 0.2))
    
    # Component 5: Clear interrogatives (what, which, how, why, when, who)
    interrogatives = ['what', 'which', 'how', 'why', 'when', 'who', 'where']
    has_interrogative = any(word in question_text.lower().split()[:5] for word in interrogatives)
    interrogative_score = 1.0 if has_interrogative else 0.6
    
    # Component 6: Negation clarity (double negatives reduce clarity)
    negations = ['not', "n't", 'never', 'no ', 'none', 'neither']
    negation_count = sum(question_text.lower().count(neg) for neg in negations)
    if negation_count == 0:
        negation_score = 1.0
    elif negation_count == 1:
        negation_score = 0.9
    else:
        negation_score = 0.5  # Multiple negations confusing
    
    # Weighted combination
    clarity_score = (
        0.20 * length_score +
        0.15 * question_mark_score +
        0.15 * single_question_score +
        0.20 * ambiguity_penalty +
        0.15 * interrogative_score +
        0.15 * negation_score
    )
    
    return {
        "clarity": clarity_score,
        "length_score": length_score,
        "question_mark_score": question_mark_score,
        "single_question_score": single_question_score,
        "ambiguity_score": ambiguity_penalty,
        "interrogative_score": interrogative_score,
        "negation_clarity": negation_score,
        "word_count": word_count
    }


# ============ RELEVANCE SCORE ============

def calculate_relevance(question_text: str, options: List[str], 
                       knowledge_base: str = None) -> Dict[str, float]:
    """
    Measure relevance of question and options to the knowledge base
    """
    # Extract keywords from question
    question_words = set(question_text.lower().split())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'is', 'are', 'was', 'were', 'what', 'which', 'how'}
    question_keywords = question_words - stop_words
    
    # Component 1: Options relate to question
    option_relevance_scores = []
    for option in options:
        option_words = set(option.lower().split()) - stop_words
        overlap = len(question_keywords & option_words)
        total = len(question_keywords | option_words)
        relevance = overlap / total if total > 0 else 0
        option_relevance_scores.append(relevance)
    
    avg_option_relevance = sum(option_relevance_scores) / len(option_relevance_scores) if option_relevance_scores else 0
    
    # Component 2: Knowledge base relevance (if provided)
    if knowledge_base:
        kb_words = set(knowledge_base.lower().split()) - stop_words
        kb_overlap = len(question_keywords & kb_words)
        kb_relevance = min(kb_overlap / len(question_keywords), 1.0) if question_keywords else 0
    else:
        kb_relevance = 0.5  # Neutral if no KB provided
    
    # Component 3: Technical terminology (longer words indicate domain specificity)
    avg_word_length = sum(len(word) for word in question_keywords) / len(question_keywords) if question_keywords else 0
    # Words >8 characters often domain-specific
    technical_score = min(avg_word_length / 8, 1.0)
    
    # Component 4: All options on-topic (low variance in relevance)
    if option_relevance_scores:
        relevance_variance = sum((score - avg_option_relevance) ** 2 for score in option_relevance_scores) / len(option_relevance_scores)
        consistency_score = 1.0 - min(relevance_variance, 1.0)
    else:
        consistency_score = 0.0
    
    # Weighted combination
    relevance_score = (
        0.35 * avg_option_relevance +
        0.25 * kb_relevance +
        0.20 * technical_score +
        0.20 * consistency_score
    )
    
    return {
        "relevance": relevance_score,
        "option_relevance": avg_option_relevance,
        "knowledge_base_relevance": kb_relevance,
        "technical_terminology": technical_score,
        "option_consistency": consistency_score
    }


# ============ READABILITY LEVEL ============

def calculate_readability_level(text: str) -> Dict[str, Any]:
    """
    Calculate readability metrics (Flesch-Kincaid Grade Level, Flesch Reading Ease)
    """
    # Clean text
    text = re.sub(r'[^\w\s.]', ' ', text)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    words = text.split()
    
    if not sentences or not words:
        return {
            "flesch_kincaid_grade": 0,
            "flesch_reading_ease": 0,
            "readability_level": "Unknown",
            "avg_sentence_length": 0,
            "avg_syllables_per_word": 0
        }
    
    # Count syllables (approximation)
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        if syllable_count == 0:
            syllable_count = 1
        
        return syllable_count
    
    total_syllables = sum(count_syllables(word) for word in words)
    num_sentences = len(sentences)
    num_words = len(words)
    
    avg_sentence_length = num_words / num_sentences
    avg_syllables_per_word = total_syllables / num_words
    
    # Flesch-Kincaid Grade Level
    if num_sentences > 0 and num_words > 0:
        fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59
        fk_grade = max(0, fk_grade)
    else:
        fk_grade = 0
    
    # Flesch Reading Ease
    if num_sentences > 0 and num_words > 0:
        flesch_ease = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (total_syllables / num_words)
        flesch_ease = max(0, min(100, flesch_ease))
    else:
        flesch_ease = 0
    
    # Determine readability level
    if fk_grade < 6:
        level = "Elementary (Grades 1-5)"
    elif fk_grade < 9:
        level = "Middle School (Grades 6-8)"
    elif fk_grade < 13:
        level = "High School (Grades 9-12)"
    elif fk_grade < 16:
        level = "College (Undergraduate)"
    else:
        level = "Graduate/Professional"
    
    return {
        "flesch_kincaid_grade": round(fk_grade, 2),
        "flesch_reading_ease": round(flesch_ease, 2),
        "readability_level": level,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2)
    }


# ============ QUALITY OF DISTRACTORS (Enhanced) ============

def calculate_distractor_quality_detailed(question: str, correct_answer: str, 
                                         distractors: List[str]) -> Dict[str, Any]:
    """
    Detailed analysis of distractor quality beyond basic DPS
    """
    if not distractors:
        return {"distractor_quality_detailed": 0.0}
    
    all_options = [correct_answer] + distractors
    
    # 1. Homogeneity Score (similar length and structure)
    lengths = [len(opt.split()) for opt in all_options]
    avg_length = sum(lengths) / len(lengths)
    length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    homogeneity_score = 1.0 / (1.0 + length_variance / 10)  # Normalize
    
    # 2. Grammatical Parallelism (similar structure)
    # Check if options start with similar parts of speech
    first_words = [opt.split()[0].lower() if opt.split() else '' for opt in all_options]
    first_word_similarity = len(set(first_words)) / len(first_words) if first_words else 0
    parallelism_score = 1.0 - first_word_similarity  # Lower variety = better parallelism
    
    # 3. No obvious patterns
    obvious_patterns = [
        'all of the above', 'none of the above', 'both a and b',
        'neither', 'cannot be determined', 'not enough information',
        'a and b', 'b and c', 'a and c'
    ]
    
    pattern_penalties = []
    for distractor in distractors:
        has_pattern = any(pattern in distractor.lower() for pattern in obvious_patterns)
        pattern_penalties.append(0.0 if has_pattern else 1.0)
    
    no_pattern_score = sum(pattern_penalties) / len(pattern_penalties) if pattern_penalties else 1.0
    
    # 4. Keyword relevance to question
    question_keywords = set(question.lower().split())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    question_keywords -= stop_words
    
    relevance_scores = []
    for distractor in distractors:
        dist_keywords = set(distractor.lower().split()) - stop_words
        overlap = len(question_keywords & dist_keywords)
        relevance = overlap / max(len(question_keywords), 1)
        relevance_scores.append(min(relevance, 1.0))
    
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    
    # 5. Plausibility (not too short, substantive)
    substantive_scores = []
    for distractor in distractors:
        word_count = len(distractor.split())
        if word_count >= 10:
            substantive_scores.append(1.0)
        elif word_count >= 5:
            substantive_scores.append(0.7)
        else:
            substantive_scores.append(0.3)
    
    avg_substantive = sum(substantive_scores) / len(substantive_scores) if substantive_scores else 0
    
    # Overall distractor quality
    overall_quality = (
        0.25 * homogeneity_score +
        0.20 * parallelism_score +
        0.25 * no_pattern_score +
        0.15 * avg_relevance +
        0.15 * avg_substantive
    )
    
    return {
        "distractor_quality_detailed": overall_quality,
        "homogeneity": homogeneity_score,
        "grammatical_parallelism": parallelism_score,
        "no_obvious_patterns": no_pattern_score,
        "keyword_relevance": avg_relevance,
        "substantiveness": avg_substantive,
        "avg_length": round(avg_length, 1),
        "length_variance": round(length_variance, 2)
    }


# ============ QUALITY OF RATIONALE ============

def calculate_rationale_quality(rationale: str, question: str, 
                               correct_answer: str) -> Dict[str, float]:
    """
    Evaluate the quality of the explanation/rationale
    """
    if not rationale or len(rationale.strip()) < 10:
        return {
            "rationale_quality": 0.0,
            "length_score": 0.0,
            "question_reference": 0.0,
            "answer_justification": 0.0,
            "educational_value": 0.0
        }
    
    # Component 1: Length appropriateness (20-100 words ideal)
    word_count = len(rationale.split())
    if 20 <= word_count <= 100:
        length_score = 1.0
    elif 10 <= word_count < 20 or 100 < word_count <= 150:
        length_score = 0.7
    else:
        length_score = 0.4
    
    # Component 2: References question concepts
    question_keywords = set(question.lower().split())
    rationale_keywords = set(rationale.lower().split())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'is', 'are', 'was', 'were', 'this', 'that', 'it'}
    
    question_keywords -= stop_words
    overlap = len(question_keywords & rationale_keywords)
    question_reference_score = min(overlap / max(len(question_keywords), 1), 1.0)
    
    # Component 3: Explains WHY (justification words)
    justification_words = ['because', 'since', 'therefore', 'thus', 'hence', 
                          'as a result', 'due to', 'reason', 'explains', 'indicates']
    has_justification = any(word in rationale.lower() for word in justification_words)
    justification_score = 1.0 if has_justification else 0.5
    
    # Component 4: References correct answer
    correct_keywords = set(correct_answer.lower().split()) - stop_words
    answer_overlap = len(correct_keywords & rationale_keywords)
    answer_reference_score = min(answer_overlap / max(len(correct_keywords), 1), 1.0)
    
    # Component 5: Educational indicators (explains concepts)
    educational_words = ['process', 'function', 'mechanism', 'principle', 'concept',
                        'means', 'refers to', 'defined as', 'characterized by']
    educational_count = sum(1 for word in educational_words if word in rationale.lower())
    educational_score = min(educational_count / 2, 1.0)
    
    # Component 6: Addresses alternatives (mentions why others wrong)
    alternative_words = ['however', 'whereas', 'while', 'unlike', 'incorrect', 
                        'wrong', 'not', 'other options', 'alternatives']
    addresses_alternatives = any(word in rationale.lower() for word in alternative_words)
    alternative_score = 1.0 if addresses_alternatives else 0.7
    
    # Overall rationale quality
    rationale_quality = (
        0.20 * length_score +
        0.20 * question_reference_score +
        0.20 * justification_score +
        0.15 * answer_reference_score +
        0.15 * educational_score +
        0.10 * alternative_score
    )
    
    return {
        "rationale_quality": rationale_quality,
        "length_score": length_score,
        "question_reference": question_reference_score,
        "answer_justification": justification_score,
        "answer_reference": answer_reference_score,
        "educational_value": educational_score,
        "addresses_alternatives": alternative_score,
        "word_count": word_count
    }


# ============ DPS AND SOS FUNCTIONS ============

def calculate_dps(question: str, correct_answer: str, distractors: List[str], knowledge_base: str = None) -> Dict[str, Any]:
    """
    Calculate Distractor Plausibility Score (DPS)
    """
    if not distractors:
        return {"DPS": 0.0}
    
    # Simple DPS calculation based on semantic similarity
    question_words = set(question.lower().split())
    correct_words = set(correct_answer.lower().split())
    
    dps_scores = []
    
    for distractor in distractors:
        distractor_words = set(distractor.lower().split())
        
        # Calculate similarity to question (should be high)
        q_similarity = len(question_words & distractor_words) / len(question_words | distractor_words) if question_words | distractor_words else 0
        
        # Calculate similarity to correct answer (should be moderate)
        c_similarity = len(correct_words & distractor_words) / len(correct_words | distractor_words) if correct_words | distractor_words else 0
        
        # DPS score: high question relevance, moderate correct answer similarity
        dps_score = (0.7 * q_similarity) + (0.3 * c_similarity)
        dps_scores.append(dps_score)
    
    avg_dps = sum(dps_scores) / len(dps_scores) if dps_scores else 0
    
    return {
        "DPS": avg_dps,
        "DPS_per_distractor": dps_scores,
        "num_distractors": len(distractors)
    }


def calculate_sos(options: List[str]) -> Dict[str, Any]:
    """
    Calculate Semantic Option Similarity (SOS)
    """
    if len(options) < 2:
        return {"SOS": 0.0, "option_diversity": 0.0}
    
    # Calculate pairwise similarities
    similarities = []
    
    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            words_i = set(options[i].lower().split())
            words_j = set(options[j].lower().split())
            
            # Jaccard similarity
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    # Option diversity is inverse of similarity
    option_diversity = 1.0 - avg_similarity
    
    return {
        "SOS": avg_similarity,
        "option_diversity": option_diversity,
        "pairwise_similarities": similarities,
        "num_pairs": len(similarities)
    }


# ============ QUESTION PARSING ============

def parse_question_block(question_block: str) -> Dict[str, Any]:
    """
    Parse a question block to extract components
    """
    import re
    
    # Initialize result
    result = {
        "question": "",
        "options": [],
        "correct_answer": None,
        "explanation": ""
    }
    
    # Extract question text (everything before options)
    question_match = re.search(r'Question \d+:\s*(.+?)(?=[ABCD]\)|$)', question_block, re.DOTALL)
    if question_match:
        result["question"] = question_match.group(1).strip()
    
    # Extract options
    option_pattern = r'([ABCD])\)\s*(.+?)(?=[ABCD]\)|Correct|Answer|Explanation|$)'
    options = re.findall(option_pattern, question_block, re.DOTALL)
    
    result["options"] = [opt[1].strip() for opt in options]
    
    # Extract correct answer
    answer_patterns = [
        r'correct\s+answer[:\s]*([ABCD])',
        r'answer[:\s]*([ABCD])',
        r'correct[:\s]*([ABCD])',
        r'\*\*([ABCD])\*\*'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, question_block, re.IGNORECASE)
        if match:
            result["correct_answer"] = match.group(1).upper()
            break
    
    # Extract explanation
    explanation_match = re.search(r'explanation[:\s]*(.+?)(?=\n\s*question|$)', question_block, re.IGNORECASE | re.DOTALL)
    if explanation_match:
        result["explanation"] = explanation_match.group(1).strip()
    
    return result


# ============ COMPREHENSIVE EVALUATION ============

def evaluate_question_comprehensive(question_data: Dict[str, Any],
                                   knowledge_base: str = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation including all metrics
    """
    results = {}
    
    # 1. Clarity
    if question_data.get("question"):
        clarity_metrics = calculate_clarity(question_data["question"])
        results.update(clarity_metrics)
    
    # 2. Relevance
    if question_data.get("question") and question_data.get("options"):
        relevance_metrics = calculate_relevance(
            question_data["question"],
            question_data["options"],
            knowledge_base
        )
        results.update(relevance_metrics)
    
    # 3. Readability Level (combine question + options)
    full_text = question_data.get("question", "")
    if question_data.get("options"):
        full_text += " " + " ".join(question_data["options"])
    
    if full_text:
        readability_metrics = calculate_readability_level(full_text)
        results.update(readability_metrics)
    
    # 4. Quality of Distractors (detailed)
    if question_data.get("options") and question_data.get("correct_answer"):
        correct_idx = ord(question_data["correct_answer"]) - ord('A')
        if 0 <= correct_idx < len(question_data["options"]):
            correct_option = question_data["options"][correct_idx]
            distractors = [opt for i, opt in enumerate(question_data["options"]) if i != correct_idx]
            
            # Basic DPS
            dps_metrics = calculate_dps(
                question_data["question"],
                correct_option,
                distractors,
                knowledge_base
            )
            results.update(dps_metrics)
            
            # Detailed distractor quality
            detailed_dist_metrics = calculate_distractor_quality_detailed(
                question_data["question"],
                correct_option,
                distractors
            )
            results.update(detailed_dist_metrics)
    
    # 5. SOS (Semantic Option Similarity)
    if question_data.get("options"):
        sos_metrics = calculate_sos(question_data["options"])
        results.update(sos_metrics)
    
    # 6. Quality of Rationale
    if question_data.get("explanation"):
        correct_idx = ord(question_data["correct_answer"]) - ord('A') if question_data.get("correct_answer") else 0
        correct_answer = question_data["options"][correct_idx] if question_data.get("options") and 0 <= correct_idx < len(question_data["options"]) else ""
        
        rationale_metrics = calculate_rationale_quality(
            question_data["explanation"],
            question_data.get("question", ""),
            correct_answer
        )
        results.update(rationale_metrics)
    
    return results


# ============ BATCH EVALUATION ============

def evaluate_question_set_comprehensive(questions_file: str,
                                       knowledge_base_file: str = None,
                                       reference_file: str = None) -> Dict[str, Any]:
    """
    Evaluate a set of questions with ALL metrics including new ones
    """
    # Load questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_text = f.read()
    
    # Load KB if provided
    knowledge_base = None
    if knowledge_base_file:
        try:
            with open(knowledge_base_file, 'r', encoding='utf-8') as f:
                knowledge_base = f.read()
        except:
            pass
    
    # Parse questions
    question_blocks = re.findall(r'Question \d+:.*?(?=Question \d+:|$)', questions_text, re.DOTALL)
    parsed_questions = [parse_question_block(block) for block in question_blocks]
    
    # Evaluate each question
    all_results = []
    for question_data in parsed_questions:
        results = evaluate_question_comprehensive(question_data, knowledge_base)
        all_results.append(results)
    
    # Aggregate results
    aggregate = {
        "num_questions": len(parsed_questions),
        "metrics": {}
    }
    
    # Calculate averages for each metric
    if all_results:
        all_metrics = set()
        for result in all_results:
            all_metrics.update(result.keys())
        
        # Exclude list/complex types from averaging
        exclude_from_avg = ['DPS_per_distractor', 'pairwise_similarities', 
                           'readability_level', 'ease_interpretation']
        
        for metric in all_metrics:
            if metric not in exclude_from_avg:
                values = []
                for r in all_results:
                    if metric in r:
                        val = r[metric]
                        # Only average numeric values
                        if isinstance(val, (int, float)):
                            values.append(val)
                
                if values:
                    aggregate["metrics"][f"avg_{metric}"] = sum(values) / len(values)
                    aggregate["metrics"][f"min_{metric}"] = min(values)
                    aggregate["metrics"][f"max_{metric}"] = max(values)
    
    # Add per-question results
    aggregate["per_question_results"] = all_results
    
    return aggregate


# ============ ENHANCED REPORTING ============

def print_comprehensive_report(evaluation_results: Dict[str, Any], output_file: str = None):
    """
    Print comprehensive evaluation report with all metrics
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("COMPREHENSIVE DOTG EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\nTotal Questions Evaluated: {evaluation_results['num_questions']}\n")
    
    metrics = evaluation_results["metrics"]
    
    # Group metrics by category
    clarity_metrics = {k: v for k, v in metrics.items() if "clarity" in k or "length_score" in k or "ambiguity" in k}
    relevance_metrics = {k: v for k, v in metrics.items() if "relevance" in k}
    readability_metrics = {k: v for k, v in metrics.items() if "flesch" in k or "readability" in k or "syllable" in k}
    distractor_metrics = {k: v for k, v in metrics.items() if "DPS" in k or "distractor" in k or "homogeneity" in k or "parallelism" in k}
    sos_metrics = {k: v for k, v in metrics.items() if "SOS" in k or "diversity" in k or "similarity" in k}
    rationale_metrics = {k: v for k, v in metrics.items() if "rationale" in k or "justification" in k}
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("COMPREHENSIVE METRICS")
    report_lines.append("=" * 70)
    
    if clarity_metrics:
        report_lines.append("\n📝 CLARITY (Question Comprehensibility)")
        report_lines.append("-" * 70)
        for metric, value in sorted(clarity_metrics.items()):
            if isinstance(value, (int, float)):
                report_lines.append(f"{metric:40s}: {value:.4f}")
    
    if relevance_metrics:
        report_lines.append("\n🎯 RELEVANCE (Topic Alignment)")
        report_lines.append("-" * 70)
        for metric, value in sorted(relevance_metrics.items()):
            if isinstance(value, (int, float)):
                report_lines.append(f"{metric:40s}: {value:.4f}")
    
    if readability_metrics:
        report_lines.append("\n📚 READABILITY LEVEL")
        report_lines.append("-" * 70)
        for metric, value in sorted(readability_metrics.items()):
            if isinstance(value, (int, float)):
                report_lines.append(f"{metric:40s}: {value:.4f}")
    
    if distractor_metrics:
        report_lines.append("\n🎭 DISTRACTOR QUALITY (Detailed)")
        report_lines.append("-" * 70)
        for metric, value in sorted(distractor_metrics.items()):
            if isinstance(value, (int, float)):
                report_lines.append(f"{metric:40s}: {value:.4f}")
    
    if sos_metrics:
        report_lines.append("\n🔀 SEMANTIC DIVERSITY (Option Variety)")
        report_lines.append("-" * 70)
        for metric, value in sorted(sos_metrics.items()):
            if isinstance(value, (int, float)):
                report_lines.append(f"{metric:40s}: {value:.4f}")
    
    if rationale_metrics:
        report_lines.append("\n💡 RATIONALE QUALITY (Explanation Quality)")
        report_lines.append("-" * 70)
        for metric, value in sorted(rationale_metrics.items()):
             if isinstance(value, (int, float)):
                report_lines.append(f"{metric:40s}: {value:.4f}")
    
    report_lines.append("\n" + "=" * 60)
    
    report = "\n".join(report_lines)
    print(report)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    # Example usage
    print("DOTG Evaluation Metrics Module")
    print("Import this module to use evaluation functions")