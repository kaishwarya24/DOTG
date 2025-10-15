# ============ TASK CREATION FUNCTIONS ============
from Agents import (
    document_parser, topic_researcher, knowledge_structurer,
    question_generator, distractor_specialist,user_analyzer, difficulty_adapter)
from crewai import Agent, Task, Crew, Process
import json
from userProfile import UserProfile

def create_knowledge_tasks(input_type, input_data):
    """Create Phase 1 tasks"""
    if input_type == "document":
        parse_task = Task(
            description=f"""COMPREHENSIVELY analyze the following document content:
            
            DOCUMENT CONTENT:
            {input_data[:4000]}{'...' if len(input_data) > 4000 else ''}
            
            SYSTEMATIC ANALYSIS PROCESS:
            1. Analyze ALL sections and content provided above
            2. Extract content from ALL parts of the document text
            3. Process every concept and detail thoroughly
            4. Capture ALL educational content, no matter how detailed
            
            EXTRACT EVERYTHING INCLUDING:
            - ALL key concepts and definitions throughout the document
            - ALL examples, case studies, and illustrations
            - ALL facts, principles, and theoretical content
            - ALL procedural knowledge and step-by-step processes
            - ALL formulas, equations, and technical details
            - Section headings and document structure
            - Prerequisites and learning objectives
            - Summary points and conclusions
            
            ORGANIZE BY DOCUMENT STRUCTURE:
            - Maintain original section/chapter organization
            - Preserve context and relationships between topics
            - Tag content with source location (chapter, section, page)
            - Identify difficulty progression throughout document
            
            IMPORTANT: Extract ONLY from the provided document content above. Do not add external knowledge.
            Your output should be comprehensive enough to generate questions from any part of the document.""",
            agent=document_parser,
            expected_output="Complete, comprehensive extraction of ALL content from the entire document, organized by structure"
        )
        context = [parse_task]
    else:  # topic
        parse_task = Task(
            description=f"""Research topic: {input_data}
            Gather: core concepts, principles, examples, applications.
            Create learning progression from basics to advanced.""",
            agent=topic_researcher,
            expected_output="Comprehensive topic research"
        )
        context = [parse_task]
    
    structure_task = Task(
        description="""Structure ALL extracted document content into a comprehensive, well-organized knowledge base:
        
        CRITICAL: Process and organize ALL content extracted from the document. Do not omit any sections.
        
        COMPREHENSIVE ORGANIZATION:
        - Create knowledge chunks from ALL document sections and chapters
        - Ensure broad topic coverage across the entire document
        - Tag content by difficulty (easy/medium/hard) based on document progression
        - Assign Bloom's taxonomy levels to different content types
        - Preserve document structure and section relationships
        - Create topic categories for every major subject area covered
        
        CONTENT DISTRIBUTION:
        - Include foundational concepts from early sections
        - Cover intermediate topics from middle sections  
        - Include advanced concepts from later sections
        - Preserve examples and case studies from throughout
        - Maintain procedural knowledge and technical details
        
        QUALITY ASSURANCE:
        - Ensure knowledge base supports question generation from any document part
        - Verify comprehensive coverage of all major topics
        - Maintain content relationships and dependencies
        - Tag content with source location for traceability
        
        Use ONLY the content extracted from the document. Do not add external information.""",
        agent=knowledge_structurer,
        expected_output="Comprehensive knowledge base covering ALL document content, organized for complete topic coverage",
        context=context
    )
    
    return [parse_task, structure_task]

def create_question_tasks(knowledge_base, difficulty, count=5):
    """Create Phase 2 tasks"""
    
    # Convert knowledge_base to string - use more content for comprehensive coverage
    kb_text = str(knowledge_base)[:3000] if hasattr(knowledge_base, '__str__') else str(knowledge_base)[:3000]
    
    generate_task = Task(
        description=f"""Generate EXACTLY {count} DIVERSE, CHALLENGING MCQs at {difficulty} difficulty from the knowledge base.
        
        MANDATORY: Generate exactly {count} questions with NO REPETITION in question types or content focus.
        
        ADVANCED QUESTION REQUIREMENTS:
        - Create questions at different Bloom's taxonomy levels:
          * {difficulty.upper()} EASY: Knowledge/Comprehension (definitions, basic facts)
          * {difficulty.upper()} MEDIUM: Application/Analysis (scenarios, comparisons, problem-solving)
          * {difficulty.upper()} HARD: Synthesis/Evaluation (critical thinking, complex scenarios)
        
        QUESTION VARIETY (use different types):
        1. Scenario-based questions ("Given situation X, what would happen if...")
        2. Comparison questions ("What is the main difference between X and Y?")
        3. Application questions ("In which scenario would you use X?")
        4. Analysis questions ("Why does X lead to Y?")
        5. Evaluation questions ("Which approach is most effective for...")
        
        AVOID:
        - Simple definition questions only
        - Repetitive question stems
        - Questions that can be answered without understanding
        - Obvious or trivial questions
        
        CONTENT REQUIREMENTS:
        - Draw questions from DIFFERENT sections/topics of the knowledge base
        - Use specific examples and details from various document parts
        - Test understanding of relationships between concepts
        - Challenge students to apply knowledge in new contexts
        - Ensure questions cover BROAD range of document content, not just one section
        
        TOPIC DISTRIBUTION:
        - Generate questions from early, middle, and later document sections
        - Cover foundational, intermediate, and advanced concepts
        - Include questions on different subject areas within the document
        - Avoid clustering all questions on the same topic
        
        FORMAT:
        Question 1: [challenging question from topic area A]
        A) [option A]
        B) [option B] 
        C) [option C]
        D) [option D]
        Correct answer: [A/B/C/D]
        
        Question 2: [challenging question from different topic area B]
        [continue with diverse topics]
        
        Knowledge base: {kb_text}...""",
        agent=question_generator,
        expected_output=f"Exactly {count} diverse, challenging MCQ questions at different cognitive levels with correct answers marked"
    )
    
    distractor_task = Task(
        description=f"""Take the {count} questions from the previous task and create EXACTLY 4 OPTIONS (A, B, C, D) for EACH question.
        
        MANDATORY REQUIREMENTS:
        - Process ALL {count} questions from the previous task
        - Each question MUST have exactly 4 options labeled A, B, C, D
        - Each question MUST have a clear "Correct answer:" line
        - Each question MUST have a detailed "Explanation:" section
        
        STRICT FORMAT FOR EACH QUESTION:
        Question [number]: [complete question text]
        A) [full option text - no truncation]
        B) [full option text - no truncation]
        C) [full option text - no truncation]
        D) [full option text - no truncation]
        Correct answer: [A/B/C/D]
        Explanation: [detailed reasoning why this answer is correct, minimum 2 sentences]
        
        CRITICAL: The "Correct answer:" line MUST be exactly in this format with a single letter A, B, C, or D.
        
        OPTION REQUIREMENTS:
        - Write complete option text, never use "..." or truncate
        - Each option should be 1-3 sentences maximum for readability
        - Options must be grammatically parallel
        - 3 distractors must be plausible but incorrect
        - 1 option must be clearly correct
        
        EXPLANATION REQUIREMENTS:
        - Minimum 2-3 sentences explaining why the correct answer is right
        - Reference specific concepts from the document
        - Explain why other options are incorrect if helpful
        
        Generate all {count} questions following this exact format.""",
        agent=distractor_specialist,
        expected_output=f"All {count} complete MCQs with exactly 4 options each, correct answers marked, and detailed explanations",
        context=[generate_task]
    )
    
    return [generate_task, distractor_task]

def create_adaptive_tasks(user_profile: UserProfile):
    """Create Phase 3 adaptive analysis tasks"""
    
    stats = user_profile.get_stats()
    
    analyze_task = Task(
        description=f"""Analyze user performance and capability:
        
        User Stats:
        - Skill Level: {stats['skill_level']}
        - Accuracy: {stats['accuracy_rate']:.1%}
        - Avg Response Time: {stats['avg_response_time']:.1f}s
        - Topics Mastery: {stats['topics_mastery']}
        - Questions Answered: {stats['total_questions']}
        
        Recent Performance:
        {json.dumps(user_profile.performance_history[-5:], indent=2)}
        
        Provide:
        1. Current capability assessment
        2. Strengths and weaknesses
        3. Learning patterns
        4. Confidence calibration
        5. Recommended difficulty level""",
        agent=user_analyzer,
        expected_output="Comprehensive user capability analysis"
    )
    
    adapt_task = Task(
        description="""Based on user analysis, determine:
        
        1. Optimal next difficulty level (easy/medium/hard)
        2. Topics to focus on
        3. Question types to prioritize
        4. Pacing recommendations
        5. Specific adjustments needed
        
        Apply adaptive algorithms:
        - If accuracy > 80%: increase difficulty
        - If accuracy < 50%: decrease difficulty
        - If time < average: may need harder questions
        - Consider confidence vs accuracy for calibration
        
        Maintain Zone of Proximal Development.""",
        agent=difficulty_adapter,
        expected_output="Adaptive difficulty recommendation",
        context=[analyze_task]
    )
    
    return [analyze_task, adapt_task]