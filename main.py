"""
Dynamic Objective Test Generation (DOTG) Framework
"""

import os
import time
import re
from dotenv import load_dotenv
from crewai import Crew, Process
from crewai import LLM
from userProfile import UserProfile
from Agents import (
    document_parser, topic_researcher, knowledge_structurer,
    question_generator, distractor_specialist, user_analyzer, difficulty_adapter)
from utils import (
    extract_correct_answer, extract_reasoning, clean_question_for_display
)
from tasks import (
    create_knowledge_tasks, create_question_tasks, create_adaptive_tasks)
from evaluation_metrics import (
    evaluate_question_set_comprehensive as evaluate_question_set, 
    print_comprehensive_report as print_evaluation_report,
    parse_question_block,
    calculate_dps,
    calculate_sos
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = LLM(
    model="openrouter/google/gemma-2-9b-it",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

def phase1_knowledge_prep(input_type, input_data):
    """Phase 1: Knowledge Preparation"""
    print("\nPreparing knowledge base...")
    
    if input_type == "document":
        from utils import parse_document
        
        if not os.path.exists(input_data):
            print(f"Error: Document not found: {input_data}")
            return None
        
        doc_result = parse_document(input_data)
        
        if "error" in doc_result:
            print(f"Error: {doc_result['error']}")
            return None
        
        document_content = doc_result["content"]
        if not document_content or len(document_content.strip()) < 50:
            print("Error: No content extracted")
            return None
        
        input_data = document_content
    
    try:
        tasks = create_knowledge_tasks(input_type, input_data)
        agents = [document_parser if input_type == "document" else topic_researcher, 
                  knowledge_structurer]
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        with open("knowledge_base.txt", "w", encoding="utf-8") as f:
            f.write(str(result))
        
        print("Knowledge base created")
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def phase2_question_generation(knowledge_base, difficulty="easy", count=3, save_questions=True):
    """Phase 2: Question Generation with optional saving for evaluation"""
    print(f"\nGenerating {count} {difficulty} questions...")
    
    kb_str = str(knowledge_base)
    if len(kb_str.strip()) < 50:
        print("Error: Knowledge base too small")
        return None
    
    try:
        tasks = create_question_tasks(knowledge_base, difficulty, count)
        
        crew = Crew(
            agents=[question_generator, distractor_specialist],
            tasks=tasks,
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        # Save questions to file for evaluation
        if save_questions:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"questions_{difficulty}_{timestamp}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(str(result))
            
            print(f"Questions saved to: {filename}")
        
        print("Questions generated")
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def phase3_adaptive_analysis(user_profile: UserProfile):
    """Phase 3: Adaptive User Analysis"""
    tasks = create_adaptive_tasks(user_profile)
    
    crew = Crew(
        agents=[user_analyzer, difficulty_adapter],
        tasks=tasks,
        process=Process.sequential,
        verbose=False
    )
    
    return crew.kickoff()

def evaluate_generated_questions(questions_file, knowledge_base_file=None, reference_file=None):
    """
    Evaluate generated questions using all metrics
    
    Args:
        questions_file: Path to generated questions file
        knowledge_base_file: Optional path to knowledge base
        reference_file: Optional path to reference questions
    """
    print(f"\n{'='*60}")
    print("EVALUATING QUESTIONS")
    print(f"{'='*60}")
    print(f"Questions file: {questions_file}")
    
    # Use knowledge_base.txt if it exists and no file specified
    if knowledge_base_file is None and os.path.exists("knowledge_base.txt"):
        knowledge_base_file = "knowledge_base.txt"
        print(f"Knowledge base: {knowledge_base_file}")
    
    try:
        # Run evaluation
        results = evaluate_question_set(
            questions_file=questions_file,
            knowledge_base_file=knowledge_base_file,
            reference_file=reference_file
        )
        
        # Print detailed report
        print_evaluation_report(results)
        
        # Save report to file
        report_filename = questions_file.replace(".txt", "_evaluation.txt")
        print_evaluation_report(results, output_file=report_filename)
        
        # Save JSON results
        json_filename = questions_file.replace(".txt", "_evaluation.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            import json
            # Convert to JSON-serializable format
            json_results = {
                "num_questions": results["num_questions"],
                "metrics": results["metrics"]
            }
            json.dump(json_results, f, indent=2)
        print(f"JSON results saved to: {json_filename}")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_session_questions(questions_text, round_num):
    """
    Evaluate questions in real-time during a learning session
    Provides quick feedback on question quality
    """
    try:
        # Parse questions
        question_blocks = re.findall(r'Question \d+:.*?(?=Question \d+:|$)', questions_text, re.DOTALL)
        
        if not question_blocks:
            return
        
        print(f"\n--- Round {round_num} Question Quality ---")
        
        total_dps = 0
        total_sos = 0
        valid_questions = 0
        
        for block in question_blocks:
            question_data = parse_question_block(block)
            
            if question_data["options"] and len(question_data["options"]) >= 4:
                # Calculate DPS
                if question_data["correct_answer"]:
                    correct_idx = ord(question_data["correct_answer"]) - ord('A')
                    if 0 <= correct_idx < len(question_data["options"]):
                        correct_option = question_data["options"][correct_idx]
                        distractors = [opt for i, opt in enumerate(question_data["options"]) if i != correct_idx]
                        
                        dps_result = calculate_dps(question_data["question"], correct_option, distractors)
                        total_dps += dps_result.get("DPS", 0)
                
                # Calculate SOS
                sos_result = calculate_sos(question_data["options"])
                total_sos += sos_result.get("option_diversity", 0)
                
                valid_questions += 1
        
        if valid_questions > 0:
            avg_dps = total_dps / valid_questions
            avg_diversity = total_sos / valid_questions
            
            print(f"Distractor Quality (DPS): {avg_dps:.3f} {'✓ Good' if avg_dps > 0.6 else '⚠ Could be better'}")
            print(f"Option Diversity: {avg_diversity:.3f} {'✓ Good' if avg_diversity > 0.5 else '⚠ Options too similar'}")
        
    except Exception as e:
        # Silent fail - don't interrupt the session
        pass


def interactive_user_session(user_id, input_type, input_data, questions_per_round=5):
    """Interactive learning session with real-time question quality feedback"""
    print(f"\nLearning session for: {user_id}")
    
    user_profile = UserProfile.load(user_id)
    knowledge_base = phase1_knowledge_prep(input_type, input_data)
    
    if not knowledge_base:
        print("Failed to prepare knowledge base")
        return
    
    difficulty = user_profile.current_difficulty
    
    for round_num in range(1, 4):
        print(f"\n{'='*60}")
        print(f"Round {round_num} - Difficulty: {difficulty.upper()}")
        print(f"{'='*60}")
        
        questions = phase2_question_generation(knowledge_base, difficulty, count=questions_per_round)
        
        if not questions:
            print("Failed to generate questions")
            return
        
        # Real-time quality evaluation disabled
        # evaluate_session_questions(str(questions), round_num)
        
        questions_text = str(questions)
        question_blocks = re.findall(r'Question \d+:.*?(?=Question \d+:|$)', questions_text, re.DOTALL)
        question_details = []
        
        actual_questions = min(questions_per_round, len(question_blocks))
        correct_count = 0
        
        for i in range(actual_questions):
            print(f"\n{'─'*60}")
            print(f"Question {i+1}/{actual_questions}")
            print(f"{'─'*60}")
            
            question_block = question_blocks[i]
            correct_answer = extract_correct_answer(question_block)
            
            if correct_answer is None:
                print(f"\nSkipping question {i+1} - could not extract correct answer")
                continue
            
            clean_question = clean_question_for_display(question_block)
            print(clean_question)
            
            start_time = time.time()
            user_answer = input("\nYour Answer (A/B/C/D): ").strip().upper()
            time_taken = time.time() - start_time
            
            confidence = int(input("Confidence (1-5): ").strip() or "3")
            
            correct = user_answer == correct_answer
            if correct:
                correct_count += 1
            
            user_profile.add_response(
                question_id=f"q_{round_num}_{i}",
                correct=correct,
                time_taken=time_taken,
                difficulty=difficulty,
                topic=input_data,
                confidence=confidence
            )
            
            result = "✓ Correct" if correct else "✗ Incorrect"
            print(f"\n{result} (Time: {time_taken:.1f}s)")
            
            question_details.append({
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'correct': correct,
                'full_question': question_block
            })
        
        # Round summary
        print(f"\n{'='*60}")
        print(f"Round {round_num} Summary")
        print(f"{'='*60}")
        print(f"Score: {correct_count}/{actual_questions} ({correct_count/actual_questions*100:.1f}%)")
        
        print(f"\n{'='*60}")
        print(f"Detailed Explanations")
        print(f"{'='*60}")
        
        for i, detail in enumerate(question_details, 1):
            status = "✓ CORRECT" if detail['correct'] else "✗ INCORRECT"
            print(f"\nQuestion {i}: {status}")
            print(f"Your answer: {detail['user_answer']}")
            print(f"Correct answer: {detail['correct_answer']}")
            
            reasoning = extract_reasoning(detail['full_question'])
            print(f"Explanation: {reasoning}")
        
        # Adaptive difficulty adjustment
        analysis = phase3_adaptive_analysis(user_profile)
        
        if "hard" in str(analysis).lower() and "increase" in str(analysis).lower():
            difficulty = "hard"
        elif "easy" in str(analysis).lower() and "decrease" in str(analysis).lower():
            difficulty = "easy"
        else:
            difficulty = "medium"
        
        user_profile.current_difficulty = difficulty
        
        if round_num < 3:
            input("\nPress Enter to continue to next round...")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("FINAL SESSION STATISTICS")
    print(f"{'='*60}")
    stats = user_profile.get_stats()
    print(f"Total Questions Answered: {stats['total_questions']}")
    print(f"Overall Accuracy: {stats['accuracy_rate']:.1%}")
    print(f"Average Response Time: {stats['avg_response_time']:.1f}s")
    print(f"Current Skill Level: {stats['skill_level']:.0f}")
    print(f"Current Difficulty: {stats['current_difficulty'].title()}")
    
    user_profile.save()
    print("\n✓ Profile saved!")

def main():
    while True:
        print("\n" + "="*40)
        print("DOTG Learning System")
        print("="*40)
        print("1. Start Learning Session")
        print("2. View Performance")
        print("3. Evaluate Questions (Comprehensive)")
        print("4. Quick Generate & Evaluate")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == "1":
            user_id = input("Name/ID: ").strip()
            input_type = input("Study from (document/topic): ").strip().lower()
            if input_type == "document":
                input_data = input("Document path: ").strip()
            else:
                input_data = input("Topic: ").strip()
            questions_per_round = int(input("Questions per round (default 3): ").strip() or "3")
            interactive_user_session(user_id, input_type, input_data, questions_per_round)
            
        elif choice == "2":
            user_id = input("Name/ID: ").strip()
            profile = UserProfile.load(user_id)
            
            if len(profile.performance_history) == 0:
                print(f"\nNo data for '{user_id}'. Complete a session first.")
            else:
                print(f"\nPerformance Report: {user_id}")
                stats = profile.get_stats()
                print(f"Total Questions: {stats['total_questions']}")
                print(f"Accuracy: {stats['accuracy_rate']:.1%}")
                print(f"Avg Time: {stats['avg_response_time']:.1f}s")
                print(f"Skill Level: {stats['skill_level']:.0f}")
                print(f"Difficulty: {stats['current_difficulty'].title()}")
                
                if stats['topics_mastery']:
                    print("\nTopic Mastery:")
                    for topic, mastery in stats['topics_mastery'].items():
                        accuracy = mastery['correct'] / mastery['total'] * 100
                        print(f"  {topic}: {accuracy:.1f}% ({mastery['correct']}/{mastery['total']})")
            
        elif choice == "3":
            # Comprehensive evaluation
            print("\n--- Question Evaluation ---")
            
            # List available question files
            question_files = [f for f in os.listdir('.') if f.startswith('questions_') and f.endswith('.txt')]
            
            if not question_files:
                print("No question files found. Generate questions first (option 1 or 4).")
                continue
            
            print("\nAvailable question files:")
            for i, file in enumerate(question_files, 1):
                print(f"{i}. {file}")
            
            file_choice = input(f"\nSelect file (1-{len(question_files)}): ").strip()
            
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(question_files):
                    selected_file = question_files[file_idx]
                    
                    # Optional: reference questions
                    use_reference = input("Compare with reference questions? (y/n): ").strip().lower()
                    reference_file = None
                    if use_reference == 'y':
                        reference_file = input("Reference file path: ").strip()
                        if not os.path.exists(reference_file):
                            print("Reference file not found. Continuing without reference.")
                            reference_file = None
                    
                    # Run evaluation
                    evaluate_generated_questions(selected_file, reference_file=reference_file)
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
        
        elif choice == "4":
            # Quick generate and evaluate
            print("\n--- Quick Generate & Evaluate ---")
            
            input_type = input("Study from (document/topic): ").strip().lower()
            if input_type == "document":
                input_data = input("Document path: ").strip()
            else:
                input_data = input("Topic: ").strip()
            
            difficulty = input("Difficulty (easy/medium/hard): ").strip().lower() or "medium"
            count = int(input("Number of questions (default 5): ").strip() or "5")
            
            # Phase 1: Prepare knowledge base
            knowledge_base = phase1_knowledge_prep(input_type, input_data)
            
            if not knowledge_base:
                print("Failed to prepare knowledge base")
                continue
            
            # Phase 2: Generate questions
            questions = phase2_question_generation(knowledge_base, difficulty, count, save_questions=True)
            
            if not questions:
                print("Failed to generate questions")
                continue
            
            # Find the most recent questions file
            question_files = [f for f in os.listdir('.') if f.startswith('questions_') and f.endswith('.txt')]
            if question_files:
                latest_file = max(question_files, key=os.path.getctime)
                
                # Automatically evaluate
                print("\nAutomatically evaluating generated questions...")
                evaluate_generated_questions(latest_file)
            
        elif choice == "5":
            print("\nLearning Never stops, This is just a break comeback sooner and stronger!")
            print("\nThank you for using DOTG!")
            return
        
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()