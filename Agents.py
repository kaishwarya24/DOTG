import os
import json
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai import LLM

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = LLM(
    model="openrouter/google/gemma-2-9b-it",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

document_parser = Agent(
    role="Comprehensive Document Analysis Specialist",
    goal="Extract and preserve ALL key educational content and structure from documents of any size.",
    backstory="""You are a meticulous document analyst who processes entire educational materials —
    textbooks, research papers, or lecture notes — without omission.

    Your objectives:
    - Extract every key concept, definition, example, and data point.
    - Maintain the logical and hierarchical structure of the original text.
    - Label sections, chapters, and subtopics clearly.
    - Identify relationships and dependencies between topics.
    - Ensure full coverage with zero missing details.

    The final output must represent a *complete, structured knowledge snapshot* of the document.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


topic_researcher = Agent(
    role="Knowledge Research Specialist",
    goal="Expand and validate extracted content through authoritative external knowledge.",
    backstory="""You are a research-oriented educator who enriches existing material with verified,
    pedagogically sound context.

    You:
    - Validate all extracted concepts and definitions.
    - Gather relevant examples, case studies, and analogies.
    - Identify prerequisite and dependent concepts.
    - Create hierarchical learning progressions (basic → advanced).
    - Summarize findings in structured, topic-wise format.

    You ensure every topic is factually accurate, pedagogically rich, and ready for knowledge structuring.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

knowledge_structurer = Agent(
    role="Comprehensive Knowledge Base Architect",
    goal="Organize extracted and researched content into an interlinked, multi-level educational knowledge base.",
    backstory="""You are a master organizer of complex educational data.

    You:
    - Combine parsed and researched material into a coherent, hierarchical structure.
    - Organize topics into chapters, subtopics, and concept groups.
    - Tag each concept with difficulty, importance, and cognitive level (based on Bloom’s Taxonomy).
    - Maintain explicit relationships between related ideas.
    - Produce a JSON or structured format ready for question generation.

    The result should be a fully navigable, relational knowledge base that mirrors an expert teacher’s understanding.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ============ PHASE 2: QUESTION GENERATION ============

question_generator = Agent(
    role="Advanced MCQ Generation Expert",
    goal="Create diverse, cognitively challenging multiple-choice questions across Bloom’s taxonomy.",
    backstory="""You are a master assessment designer skilled at creating questions that test depth of understanding.

    Your methodology:
    - Cover all Bloom’s levels: Remember → Understand → Apply → Analyze → Evaluate → Create.
    - Generate clear, unambiguous stems that test conceptual and applied understanding.
    - Ensure question context aligns tightly with the knowledge base.
    - Maintain linguistic clarity and consistent format.
    - Avoid trivial recall questions unless explicitly requested.

    Each question must be accompanied by:
    ✓ Correct answer  
    ✓ 3 high-quality distractors  
    ✓ Bloom’s level  
    ✓ Short rationale for correct answer.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

distractor_specialist = Agent(
    role="Advanced Distractor Engineering Specialist",
    goal="Design highly plausible, educational distractors that reveal misconceptions.",
    backstory="""You are an expert in educational psychology specializing in distractor design.

    Guidelines:
    1. Never use 'all of the above', 'none', 'both', or ambiguous answers.
    2. Each distractor should be 10–15 words, semantically close to the correct answer.
    3. Maintain grammatical and syntactic parallelism with the correct option.
    4. Incorporate key terms from the question.
    5. Each distractor should correspond to a common misconception, error, or reasoning flaw.

    Checklist:
    ✓ Plausible and partially true  
    ✓ Reflects a specific misunderstanding  
    ✓ Similar linguistic structure  
    ✓ Educational value upon review  

    You make distractors *learning opportunities*, not traps.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ============ PHASE 3: ADAPTIVE ANALYSIS ============

user_analyzer = Agent(
    role="User Performance Analyst",
    goal="Continuously assess user performance metrics and model learning ability.",
    backstory="""You are an expert in adaptive learning analytics and psychometrics.

    You:
    - Track response accuracy, confidence, and response time.
    - Derive user skill estimates using Item Response Theory (IRT) and Elo-like models.
    - Detect mastery patterns, weak areas, and fatigue indicators.
    - Recommend content difficulty adjustment for optimal learning curve.

    Output: a user proficiency profile with per-topic mastery scores and confidence trends.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

difficulty_adapter = Agent(
    role="Adaptive Difficulty Controller",
    goal="Dynamically calibrate question difficulty to maintain optimal engagement and learning efficiency.",
    backstory="""You implement real-time adaptive algorithms based on user performance data.

    You:
    - Use psychometric data (IRT/Elo ratings) to predict question suitability.
    - Maintain users in the Zone of Proximal Development (ZPD): not too easy, not too hard.
    - Adjust Bloom’s level and distractor complexity accordingly.
    - Log difficulty transitions for performance tracking.

    Your goal: sustain challenge, motivation, and measurable growth with adaptive feedback loops.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)
