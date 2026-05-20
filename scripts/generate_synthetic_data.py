import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


# Writes to ../data/synthetic_data.sqlite relative to this script by default.
# Override with SQLITE_DB_PATH=/absolute/path/synthetic_data.sqlite if needed.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_data.sqlite"


SCHEMA: Dict[str, List[Tuple[str, str]]] = {
    "classes": [
        ("id", "bigint"),
        ("name", "text"),
        ("professor", "text"),
        ("location", "text"),
        ("from_date", "text"),
        ("to_date", "text"),
        ("short_description", "text"),
        ("full_description", "text"),
        ("additional_info", "text"),
        ("topics", "text"),
        ("frameworks", "text"),
        ("ranking", "bigint"),
        ("show_on_preview", "bigint"),
        ("show_on_full", "bigint"),
        ("displayed_color", "text"),
        ("slug", "text"),
        ("html_content", "text"),
        ("preview_thumb", "text"),
        ("timeline_content_back", "text"),
        ("title", "text"),
        ("subtitle", "text"),
        ("link", "text"),
    ],
    "projects": [
        ("id", "bigint"),
        ("title", "text"),
        ("lab", "text"),
        ("location", "text"),
        ("from_date", "text"),
        ("to_date", "text"),
        ("short_description", "text"),
        ("full_description", "text"),
        ("additional_info", "text"),
        ("skills", "text"),
        ("software_skills", "text"),
        ("data_skills", "text"),
        ("frameworks", "text"),
        ("ranking", "bigint"),
        ("show_on_preview", "bigint"),
        ("show_on_full", "bigint"),
        ("displayed_color", "text"),
        ("html_content", "text"),
        ("media_dir_path", "text"),
        ("link", "text"),
        ("slug", "text"),
        ("github_repos", "text"),
        ("slug_old", "text"),
    ],
    "works": [
        ("id", "numeric"),
        ("title", "text"),
        ("employer", "text"),
        ("location", "text"),
        ("from_date", "text"),
        ("to_date", "text"),
        ("short_description", "text"),
        ("full_description", "text"),
        ("additional_info", "text"),
        ("skills", "text"),
        ("frameworks", "text"),
        ("ranking", "bigint"),
        ("show_on_preview", "bigint"),
        ("show_on_full", "bigint"),
        ("displayed_color", "text"),
        ("slug", "text"),
        ("html_content", "text"),
        ("preview_thumb", "text"),
        ("full_story", "text"),
        ("short_story", "text"),
        ("timeline_content_back", "text"),
        ("link", "text"),
    ],
    "educations": [
        ("id", "bigint"),
        ("location", "text"),
        ("from_date", "text"),
        ("to_date", "text"),
        ("additional_info", "text"),
        ("topics", "text"),
        ("ranking", "bigint"),
        ("show_on_preview", "bigint"),
        ("show_on_full", "bigint"),
        ("displayed_color", "text"),
        ("slug", "text"),
        ("html_content", "text"),
        ("timeline_content_back", "text"),
        ("title", "text"),
        ("subtitle", "text"),
        ("link", "text"),
    ],
}


TABLE_ORDER = ["classes", "projects", "works", "educations"]


CLASS_DESCRIPTIONS = {
    "cs-105": (
        "Computer Science 105",
        "Dr. Lena Moreno",
        "San Jose City College",
        "2021-08-23",
        "2021-12-17",
        "Introduced structured programming, problem decomposition, and debugging in Python.",
        "Covered variables, control flow, functions, lists, dictionaries, file IO, testing, and small automation scripts. Several assignments connected programming to cooking recipes and nutrition logs so code could parse ingredients and compute simple summaries.",
        {"grade": "A", "course_code": "CS 105", "cooking_related": True},
        "programming, python, debugging, algorithms, cooking data",
        "Python, pytest",
        1,
        "cs-105",
        "Computer Science 105",
        "Intro Programming",
    ),
    "cis-054": (
        "CIS 054: C++ Programming",
        "Prof. Ada Novak",
        "Santa Barbara City College",
        "2022-01-10",
        "2022-05-20",
        "SBCC class focused on low-level coding with C++ and memory-oriented programming.",
        "Covered pointers, references, stack and heap memory, manual resource management, structs, classes, compilation, Makefiles, and debugging with gdb. This is the SBCC low-level coding class.",
        {"grade": "A", "course_code": "CIS 054", "low_level_coding": True, "sbcc": True},
        "c++, low level coding, pointers, memory, systems",
        "C++, Make, gdb",
        2,
        "cis-054",
        "CIS 054",
        "SBCC Low-Level Coding",
    ),
    "nufs-115": (
        "NUFS 115: Nutrition and Food Science",
        "Dr. Mira Chen",
        "San Jose State University",
        "2023-08-21",
        "2023-12-15",
        "Explored nutrition, meal planning, digestion, and evidence-based health claims.",
        "Covered macronutrients, micronutrients, metabolism, food labels, dietary patterns, cooking decisions, and nutrition analysis. Student built small data summaries comparing recipes by protein, sodium, fiber, and cost.",
        {"grade": "A-", "course_code": "NUFS 115", "nutrition_related": True, "cooking_related": True},
        "nutrition, food science, cooking, metabolism, meal planning",
        "Excel, Python, pandas",
        3,
        "nufs-115",
        "NUFS 115",
        "Nutrition",
    ),
    "biol-020": (
        "BIOL 020: Human Biology",
        "Dr. Omar Silva",
        "San Jose City College",
        "2021-01-25",
        "2021-05-21",
        "Covered human biology with units on digestion, metabolism, and nutrition.",
        "Covered cells, organ systems, digestive physiology, metabolism, public health, nutrition, and experimental reasoning. Nutrition appeared in the digestion and health modules.",
        {"grade": "A", "course_code": "BIOL 020", "nutrition_related": True},
        "biology, digestion, nutrition, metabolism, physiology",
        "Lab notebooks, spreadsheets",
        4,
        "biol-020",
        "BIOL 020",
        "Human Biology",
    ),
    "cs-146": (
        "CS 146: Data Structures and Algorithms",
        "Prof. Malik Ortiz",
        "San Jose State University",
        "2023-01-25",
        "2023-05-19",
        "Built algorithmic foundations for efficient software engineering.",
        "Covered asymptotic analysis, trees, heaps, graphs, hashing, dynamic programming, greedy methods, and implementation tradeoffs in Java and C++ style pseudocode.",
        {"grade": "A", "course_code": "CS 146"},
        "algorithms, data structures, graphs, dynamic programming",
        "Java, C++ concepts",
        5,
        "cs-146",
        "CS 146",
        "Algorithms",
    ),
    "cs-157a": (
        "CS 157A: Database Management Systems",
        "Prof. Nina Park",
        "San Jose State University",
        "2023-08-21",
        "2023-12-15",
        "Designed relational schemas and SQL queries for application data.",
        "Covered normalization, indexing, query planning, joins, transactions, entity relationship modeling, and backend persistence patterns.",
        {"grade": "A", "course_code": "CS 157A"},
        "databases, sql, relational design, indexing, transactions",
        "PostgreSQL, SQLite, SQL",
        6,
        "cs-157a",
        "CS 157A",
        "Databases",
    ),
    "cs-171": (
        "CS 171: Machine Learning",
        "Dr. Priya Raman",
        "San Jose State University",
        "2024-01-24",
        "2024-05-17",
        "Applied machine learning models to classification and recommendation problems.",
        "Covered supervised learning, model evaluation, feature engineering, gradient descent, neural networks, NLP features, and reproducible notebooks.",
        {"grade": "A", "course_code": "CS 171", "ml_related": True},
        "machine learning, nlp, model evaluation, feature engineering",
        "Python, scikit-learn, pandas, NumPy",
        7,
        "cs-171",
        "CS 171",
        "Machine Learning",
    ),
    "cmpe-252": (
        "CMPE 252: Artificial Intelligence and Data Engineering",
        "Dr. Salma Reyes",
        "San Jose State University",
        "2025-01-22",
        "2025-05-16",
        "Graduate course on AI systems, retrieval, and data pipelines.",
        "Covered search, RAG, vector databases, data ingestion, evaluation, agentic workflows, and deployment considerations for AI applications.",
        {"grade": "A", "course_code": "CMPE 252", "mlops_related": True},
        "ai, rag, vector databases, data pipelines, mlops",
        "Python, LangChain, FastAPI, Chroma, Docker",
        8,
        "cmpe-252",
        "CMPE 252",
        "AI Data Engineering",
    ),
    "rtf-110": (
        "RTF 110: Audio Production",
        "Prof. Carla Bennett",
        "San Jose State University",
        "2024-08-21",
        "2024-12-13",
        "Covered recording, live sound, audio events, signal flow, and post-production.",
        "Hands-on class covering microphones, mixers, signal routing, field recording, AV event setup, live audio checks, and editing workflows.",
        {"grade": "A-", "course_code": "RTF 110", "av_audio_events": True},
        "audio production, av events, signal flow, recording",
        "Audacity, Reaper, mixers, microphones",
        9,
        "rtf-110",
        "RTF 110",
        "Audio Production",
    ),
    "math-167r": (
        "MATH 167R: Statistical Methods",
        "Dr. Ken Ito",
        "San Jose State University",
        "2024-08-21",
        "2024-12-13",
        "Applied probability and statistics to data analysis.",
        "Covered distributions, estimation, regression, hypothesis testing, sampling, experimental design, and model interpretation.",
        {"grade": "A", "course_code": "MATH 167R"},
        "statistics, regression, probability, data analysis",
        "R, Python, pandas",
        10,
        "math-167r",
        "MATH 167R",
        "Statistics",
    ),
    "cs-180": (
        "CS 180: Individual Studies in Systems",
        "Dr. Elise Morgan",
        "San Jose State University",
        "2024-08-21",
        "2024-12-13",
        "Independent systems project connecting operating concepts with embedded software.",
        "Explored process scheduling, filesystems, concurrency, serial protocols, and C/C++ implementation techniques for small systems.",
        {"grade": "A", "course_code": "CS 180", "systems_related": True},
        "systems, operating systems, embedded, c++",
        "C, C++, Linux, serial protocols",
        11,
        "cs-180",
        "CS 180",
        "Systems Study",
    ),
    "chem-001a": (
        "CHEM 001A: General Chemistry",
        "Dr. Renata Gomez",
        "San Jose City College",
        "2020-08-24",
        "2020-12-18",
        "Introduced chemical principles used in biology, cooking, and materials.",
        "Covered stoichiometry, thermodynamics, bonding, solutions, acids and bases, and lab safety. Some examples connected chemistry to cooking processes and food reactions.",
        {"grade": "A", "course_code": "CHEM 001A", "cooking_related": True},
        "chemistry, lab, thermodynamics, cooking reactions",
        "Lab equipment, spreadsheets",
        12,
        "chem-001a",
        "CHEM 001A",
        "General Chemistry",
    ),
}


PROJECTS = [
    {
        "id": 1,
        "title": "AI-LLM-Net",
        "lab": "Independent AI Systems Lab",
        "location": "San Jose, CA",
        "from_date": "2026-02-01",
        "to_date": "Present",
        "short_description": "Production-style LLM networking project with retrieval, routing, evaluation, and service orchestration.",
        "full_description": "AI-LLM-Net is a synthetic flagship project showing system design for an LLM platform: ingestion, chunking, retrieval, reranking, prompt routing, offline evaluation, online traces, service boundaries, and deployment. It demonstrates end-to-end MLOps through data processing, experiment tracking, containerized services, CI checks, model evaluation, observability, and rollback planning.",
        "additional_info": {
            "status": "active",
            "rank_reason": "best system design and MLOps coverage",
            "end_to_end_mlops": True,
            "ml_nlp_strength": 10,
            "system_design_strength": 10,
            "recommended_first_for_system_design": True,
            "last_commit_date": "2026-05-12",
            "outcome": "Built a reference architecture for LLM application deployment and evaluation.",
            "compare_notes": "Broader platform architecture than AI-Grader; focuses on routing, retrieval, deployment, and observability.",
        },
        "skills": "system design, mlops, llm, rag, evaluation, observability, deployment",
        "software_skills": "Python, FastAPI, SQL, REST APIs, Docker, GitHub Actions",
        "data_skills": "embeddings, retrieval evaluation, prompt evaluation, dataset versioning, monitoring",
        "frameworks": "LangChain, Chroma, OpenAI API, FastAPI, Docker, pytest",
        "ranking": 1,
        "slug": "ai-llm-net",
        "github_repos": "https://github.com/example/ai-llm-net",
        "slug_old": "ai-llm-net-draft",
    },
    {
        "id": 2,
        "title": "AI-Grader",
        "lab": "Course Automation Lab",
        "location": "San Jose, CA",
        "from_date": "2025-11-01",
        "to_date": "2026-03-20",
        "short_description": "LLM-assisted grading tool for rubrics, feedback, and structured evaluation.",
        "full_description": "AI-Grader uses LLM prompts, rubric schemas, retrieval of assignment context, and evaluation reports to help instructors draft consistent feedback. It emphasizes NLP, prompt engineering, classification, rubric scoring, and human-in-the-loop review rather than broad infrastructure.",
        "additional_info": {
            "status": "complete",
            "end_to_end_mlops": True,
            "ml_nlp_strength": 9,
            "system_design_strength": 8,
            "outcome": "Reduced manual rubric drafting time in a simulated grading workflow.",
            "compare_notes": "More focused than LLM-Net; strongest in rubric NLP, feedback generation, and evaluation workflows.",
        },
        "skills": "nlp, llm, rubric scoring, human in the loop, evaluation, backend",
        "software_skills": "Python, FastAPI, SQLite, React, TypeScript",
        "data_skills": "rubric datasets, text classification, feedback evaluation, prompt testing",
        "frameworks": "OpenAI API, FastAPI, React, pytest, Docker",
        "ranking": 2,
        "slug": "ai-llm-agents-og-grader",
        "github_repos": "https://github.com/example/ai-grader",
        "slug_old": "ai-grader",
    },
    {
        "id": 3,
        "title": "Nuvoton Audio Event Classifier",
        "lab": "Nuvoton ML Prototyping",
        "location": "San Jose, CA",
        "from_date": "2025-05-15",
        "to_date": "2025-09-30",
        "short_description": "Embedded ML prototype for classifying AV and audio events on constrained hardware.",
        "full_description": "Built a synthetic audio-event classification pipeline for embedded devices, including preprocessing, spectrogram generation, YamNet-style baseline modeling, latency checks, dataset splits, and deployment packaging. The project supports questions about Nuvoton work and AV audio event experience.",
        "additional_info": {
            "status": "complete",
            "end_to_end_mlops": True,
            "ml_nlp_strength": 8,
            "system_design_strength": 7,
            "uses_av_audio_events": True,
            "nuvoton_related": True,
            "outcome": "Delivered an audio-event demo pipeline with reproducible preprocessing and model reports.",
        },
        "skills": "audio ml, embedded ml, mlops, preprocessing, model deployment, av audio events",
        "software_skills": "Python, C++, shell scripting, Git, Linux",
        "data_skills": "spectrograms, audio classification, dataset splits, metrics, model reports",
        "frameworks": "TensorFlow, YamNet, NumPy, librosa, Docker",
        "ranking": 3,
        "slug": "ai-ml-nuvoton-yamnet",
        "github_repos": "https://github.com/example/nuvoton-audio-events",
        "slug_old": "nuvoton-yamnet",
    },
    {
        "id": 4,
        "title": "Dune Buggy",
        "lab": "Embedded Mobility Workshop",
        "location": "Santa Barbara, CA",
        "from_date": "2024-02-01",
        "to_date": "2024-06-15",
        "short_description": "Embedded dune buggy project combining sensors, C++, motor control, and field debugging.",
        "full_description": "Designed and tested a small dune buggy control stack using C++ firmware, sensor logging, motor control logic, and mechanical debugging. The project is useful for summarizing hands-on engineering, low-level implementation, and system integration under physical constraints.",
        "additional_info": {
            "status": "complete",
            "uses_cpp": True,
            "ml_nlp_strength": 2,
            "system_design_strength": 7,
            "bullet_summary": [
                "Built C++ firmware for sensor polling and motor-control routines.",
                "Integrated IMU and distance sensor telemetry for field tests.",
                "Created serial logging to debug control behavior after each run.",
                "Improved reliability by separating hardware faults from software faults.",
                "Documented wiring, calibration, and test results for repeatability."
            ],
            "outcome": "Produced a working prototype and debugging workflow for embedded vehicle experiments.",
        },
        "skills": "embedded systems, c++, sensors, motor control, debugging, robotics",
        "software_skills": "C++, Arduino, PlatformIO, serial debugging",
        "data_skills": "sensor logs, calibration, telemetry analysis",
        "frameworks": "Arduino, PlatformIO",
        "ranking": 4,
        "slug": "dune-buggy",
        "github_repos": "https://github.com/example/dune-buggy",
        "slug_old": "dune-buggy-sensor-system",
    },
    {
        "id": 5,
        "title": "xewe-os",
        "lab": "Personal Systems Lab",
        "location": "San Jose, CA",
        "from_date": "2024-10-01",
        "to_date": "2026-01-18",
        "short_description": "C++ systems project for coordinating LED devices, commands, and runtime state.",
        "full_description": "xewe-os is a synthetic systems project for coordinating LED devices and embedded controllers. It includes command parsing, runtime state machines, C++ modules, device abstractions, and GitHub metadata for answering last-commit questions.",
        "additional_info": {
            "status": "paused",
            "uses_cpp": True,
            "ml_nlp_strength": 1,
            "system_design_strength": 8,
            "last_commit_date": "2026-01-18",
            "outcome": "Created a modular controller architecture for small device orchestration.",
        },
        "skills": "systems programming, c++, device control, state machines, architecture",
        "software_skills": "C++, CMake, Linux, Git, shell scripting",
        "data_skills": "device logs, runtime diagnostics",
        "frameworks": "CMake, SQLite",
        "ranking": 5,
        "slug": "xewe-os",
        "github_repos": "https://github.com/example/xewe-os",
        "slug_old": "xewe-led-os",
    },
    {
        "id": 6,
        "title": "AI-LLM Data Processor",
        "lab": "Independent AI Systems Lab",
        "location": "San Jose, CA",
        "from_date": "2025-08-01",
        "to_date": "2025-10-31",
        "short_description": "Data ingestion and normalization pipeline for LLM-ready knowledge bases.",
        "full_description": "Built a document processing pipeline that extracts text, chunks records, normalizes metadata, validates schemas, and writes outputs for retrieval and agent workflows. It supports MLOps and data engineering questions.",
        "additional_info": {
            "status": "complete",
            "end_to_end_mlops": True,
            "ml_nlp_strength": 7,
            "system_design_strength": 8,
            "outcome": "Produced repeatable document ingestion for downstream retrieval applications.",
        },
        "skills": "data engineering, mlops, ingestion, validation, rag, automation",
        "software_skills": "Python, SQL, CLI tools, Docker",
        "data_skills": "chunking, metadata normalization, schema validation, embeddings",
        "frameworks": "pandas, Pydantic, FastAPI, Chroma",
        "ranking": 6,
        "slug": "ai-llm-data-processor",
        "github_repos": "https://github.com/example/ai-llm-data-processor",
        "slug_old": "data-processing-pipeline",
    },
    {
        "id": 7,
        "title": "Haaangry Cooking Recommender",
        "lab": "CalHacks Prototype Team",
        "location": "Berkeley, CA",
        "from_date": "2025-10-18",
        "to_date": "2025-10-20",
        "short_description": "Hackathon app recommending meals from available ingredients and nutrition constraints.",
        "full_description": "Built a cooking recommendation prototype that used pantry items, budget, cuisine preferences, and nutrition constraints to recommend meals. This project supports queries about cooking across projects.",
        "additional_info": {
            "status": "complete",
            "cooking_related": True,
            "nutrition_related": True,
            "ml_nlp_strength": 5,
            "system_design_strength": 5,
            "outcome": "Delivered a working hackathon prototype with backend recipe scoring.",
        },
        "skills": "cooking, recommendation systems, backend, nutrition, hackathon",
        "software_skills": "Python, FastAPI, JavaScript, REST APIs",
        "data_skills": "recipe scoring, nutrition filters, ingredient matching",
        "frameworks": "FastAPI, React, SQLite",
        "ranking": 7,
        "slug": "calhacks-12.0-haaangry",
        "github_repos": "https://github.com/example/haaangry",
        "slug_old": "calhacks-12-haaangry",
    },
    {
        "id": 8,
        "title": "Urban Safety ML",
        "lab": "Civic ML Lab",
        "location": "San Jose, CA",
        "from_date": "2025-01-10",
        "to_date": "2025-04-30",
        "short_description": "Machine learning project for urban safety signal classification and risk summaries.",
        "full_description": "Built a supervised ML workflow over synthetic civic incident records with feature engineering, model comparison, fairness checks, and dashboard-ready summaries.",
        "additional_info": {
            "status": "complete",
            "end_to_end_mlops": False,
            "ml_nlp_strength": 7,
            "system_design_strength": 6,
            "outcome": "Produced a reproducible notebook and summary dataset for risk exploration.",
        },
        "skills": "machine learning, data analysis, model evaluation, civic data",
        "software_skills": "Python, SQL, Jupyter",
        "data_skills": "classification, feature engineering, metrics, fairness checks",
        "frameworks": "scikit-learn, pandas, matplotlib",
        "ranking": 8,
        "slug": "ai-ml-urban-safety",
        "github_repos": "https://github.com/example/urban-safety-ml",
        "slug_old": "urban-safety",
    },
    {
        "id": 9,
        "title": "Stock Data Scraper",
        "lab": "Personal Data Tools",
        "location": "San Jose, CA",
        "from_date": "2024-04-01",
        "to_date": "2024-05-15",
        "short_description": "Scraper and SQLite pipeline for collecting and cleaning stock market snapshots.",
        "full_description": "Created a data scraper that fetches synthetic stock snapshots, normalizes fields, stores them in SQLite, and exports CSV summaries for analysis.",
        "additional_info": {
            "status": "complete",
            "ml_nlp_strength": 2,
            "system_design_strength": 5,
            "outcome": "Automated repeatable collection and cleaning of market-style records.",
        },
        "skills": "web scraping, data engineering, sql, automation",
        "software_skills": "Python, SQLite, cron, shell scripting",
        "data_skills": "data cleaning, time series, csv export",
        "frameworks": "requests, pandas, SQLite",
        "ranking": 9,
        "slug": "data-scraper-stocks",
        "github_repos": "https://github.com/example/stock-data-scraper",
        "slug_old": "stocks-scraper",
    },
    {
        "id": 10,
        "title": "Voice to Text Notes",
        "lab": "Personal Productivity Lab",
        "location": "San Jose, CA",
        "from_date": "2025-03-01",
        "to_date": "2025-05-15",
        "short_description": "Speech-to-text note pipeline with transcription cleanup and summarization.",
        "full_description": "Built a voice-to-text pipeline for recording notes, cleaning transcripts, segmenting topics, and summarizing action items. It demonstrates NLP, audio handling, and application glue code.",
        "additional_info": {
            "status": "complete",
            "ml_nlp_strength": 8,
            "system_design_strength": 6,
            "uses_av_audio_events": True,
            "outcome": "Converted audio notes into searchable summaries and action items.",
        },
        "skills": "speech to text, nlp, audio processing, summarization",
        "software_skills": "Python, FastAPI, JavaScript",
        "data_skills": "transcription cleanup, topic segmentation, summarization evaluation",
        "frameworks": "Whisper, OpenAI API, FastAPI",
        "ranking": 10,
        "slug": "ai-ml-voice-2-text",
        "github_repos": "https://github.com/example/voice-2-text",
        "slug_old": "voice-notes",
    },
]


WORKS = [
    {
        "id": 1,
        "title": "AI/ML Engineer",
        "employer": "Nuvoton",
        "location": "San Jose, CA",
        "from_date": "2025-05-01",
        "to_date": "2025-12-20",
        "short_description": "Built ML prototypes, audio-event pipelines, and deployment scripts for embedded demos.",
        "full_description": "At Nuvoton, created synthetic ML workflows for audio-event classification, command detection, model evaluation, preprocessing, and deployment packaging. Work included Python pipelines, C++ integration points, Linux scripts, reproducible reports, and communication with hardware engineers.",
        "additional_info": {
            "software_engineering_match_score": 0.96,
            "best_for_software_engineer_role": True,
            "nuvoton_related": True,
            "av_audio_events": True,
            "what_user_did": "Built preprocessing scripts, trained and evaluated audio-event models, packaged demos, wrote deployment notes, and debugged embedded integration issues.",
        },
        "skills": "machine learning, software engineering, embedded ml, audio events, deployment, testing",
        "frameworks": "Python, C++, TensorFlow, YamNet, Docker, Linux, Git",
        "ranking": 1,
        "slug": "ai-ml-engineer-nuvoton",
        "full_story": "The strongest software-engineering match because it combines production-style coding, ML experimentation, testable scripts, deployment constraints, and cross-functional debugging.",
        "short_story": "Built ML and embedded audio-event tooling at Nuvoton.",
    },
    {
        "id": 2,
        "title": "Machine Learning Intern",
        "employer": "Nuvoton",
        "location": "San Jose, CA",
        "from_date": "2024-06-01",
        "to_date": "2024-09-15",
        "short_description": "Prepared datasets, trained baseline models, and evaluated embedded audio classification demos.",
        "full_description": "Supported model prototyping for embedded ML demos. Cleaned datasets, generated spectrogram features, trained baselines, compared metrics, and summarized findings for engineering review.",
        "additional_info": {
            "software_engineering_match_score": 0.88,
            "nuvoton_related": True,
            "av_audio_events": True,
            "what_user_did": "Prepared data, trained baselines, evaluated audio classifiers, and documented model behavior for embedded constraints.",
        },
        "skills": "machine learning, audio classification, data preprocessing, evaluation",
        "frameworks": "Python, TensorFlow, librosa, NumPy, pandas",
        "ranking": 2,
        "slug": "machine-learning-intern-nuvoton",
        "full_story": "Focused internship experience in data preparation, ML baselines, and audio model evaluation.",
        "short_story": "ML intern working on audio classification demos.",
    },
    {
        "id": 3,
        "title": "Upper Math Tutor",
        "employer": "San Jose State University",
        "location": "San Jose, CA",
        "from_date": "2023-08-20",
        "to_date": "2025-05-15",
        "short_description": "Tutored students in upper-division math, statistics, and problem solving.",
        "full_description": "Helped students reason through proofs, calculus, linear algebra, probability, and statistics. Built communication skills by translating complex technical ideas into step-by-step explanations.",
        "additional_info": {"software_engineering_match_score": 0.55},
        "skills": "teaching, math, statistics, communication, problem solving",
        "frameworks": "LaTeX, Python notebooks",
        "ranking": 3,
        "slug": "upper-math-tutor-sjsu",
        "full_story": "Developed technical communication and analytical depth through tutoring.",
        "short_story": "Tutored math and statistics at SJSU.",
    },
    {
        "id": 4,
        "title": "Learning Assistant",
        "employer": "San Jose State University",
        "location": "San Jose, CA",
        "from_date": "2023-01-20",
        "to_date": "2023-12-15",
        "short_description": "Supported programming and data-structures students during labs and review sessions.",
        "full_description": "Answered programming questions, helped debug assignments, guided students through data structures, and prepared examples for review sessions.",
        "additional_info": {"software_engineering_match_score": 0.72},
        "skills": "programming, debugging, teaching, data structures",
        "frameworks": "Java, Python, Git",
        "ranking": 4,
        "slug": "learning-assistant-sjsu",
        "full_story": "Built mentoring and debugging skills through student support.",
        "short_story": "Helped students debug code and understand data structures.",
    },
    {
        "id": 5,
        "title": "Bartender",
        "employer": "Teburasika Bar",
        "location": "Santa Barbara, CA",
        "from_date": "2022-05-01",
        "to_date": "2022-12-20",
        "short_description": "Handled customer service, drink preparation, inventory, and event support.",
        "full_description": "Prepared drinks, coordinated service during crowded events, maintained inventory, and handled customer-facing operations. This role is relevant to cooking and hospitality-related queries.",
        "additional_info": {"cooking_related": True, "software_engineering_match_score": 0.2},
        "skills": "hospitality, cooking-adjacent preparation, inventory, customer service",
        "frameworks": "POS systems, inventory sheets",
        "ranking": 5,
        "slug": "bartender-teburasika-bar",
        "full_story": "Operational role involving preparation workflows and service under time pressure.",
        "short_story": "Bartender with event and preparation responsibilities.",
    },
    {
        "id": 6,
        "title": "Business Owner",
        "employer": "Secondhand Store",
        "location": "Online",
        "from_date": "2021-03-01",
        "to_date": "2022-04-30",
        "short_description": "Ran a small resale operation with pricing, inventory, and customer communication.",
        "full_description": "Managed sourcing, listings, pricing, order tracking, shipping, and customer messages for a small secondhand store.",
        "additional_info": {"software_engineering_match_score": 0.35},
        "skills": "operations, inventory, pricing, customer communication",
        "frameworks": "Spreadsheets, marketplace tools",
        "ranking": 6,
        "slug": "business-owner-secondhand-store",
        "full_story": "Built ownership habits around operations, tracking, and customer support.",
        "short_story": "Operated a small secondhand resale store.",
    },
]


EDUCATIONS = [
    {
        "id": 1,
        "location": "Santa Barbara, CA",
        "from_date": "2020-08-24",
        "to_date": "2022-05-20",
        "additional_info": {
            "gpa": 3.86,
            "honors": "Dean's List",
            "top_accomplishment": "Completed transfer-focused computer science preparation while building low-level C++ and math foundations.",
        },
        "topics": "computer science, mathematics, c++, transfer preparation",
        "ranking": 1,
        "slug": "degree-as",
        "title": "Associate of Science",
        "subtitle": "Santa Barbara City College",
    },
    {
        "id": 2,
        "location": "San Jose, CA",
        "from_date": "2022-08-22",
        "to_date": "2025-05-16",
        "additional_info": {
            "gpa": 3.91,
            "honors": "Magna Cum Laude",
            "top_accomplishment": "Completed a BS while combining algorithms, databases, ML, tutoring, and multiple applied AI/software projects.",
        },
        "topics": "computer science, algorithms, databases, machine learning, systems",
        "ranking": 2,
        "slug": "degree-bs",
        "title": "Bachelor of Science in Computer Science",
        "subtitle": "San Jose State University",
    },
    {
        "id": 3,
        "location": "San Jose, CA",
        "from_date": "2025-08-20",
        "to_date": "Present",
        "additional_info": {
            "gpa": 3.95,
            "honors": "Graduate Research Track",
            "top_accomplishment": "Focused graduate work on AI systems, data engineering, retrieval, and deployment-oriented MLOps.",
        },
        "topics": "artificial intelligence, data engineering, mlops, rag, systems",
        "ranking": 3,
        "slug": "degree-ms",
        "title": "Master of Science in Software Engineering",
        "subtitle": "San Jose State University",
    },
]


def json_text(value: Dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def make_html(title: str, body: str) -> str:
    return f"<section><h2>{title}</h2><p>{body}</p></section>"


def create_tables(conn: sqlite3.Connection) -> None:
    for table_name in TABLE_ORDER:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    for table_name in TABLE_ORDER:
        columns_sql = []
        for column_name, data_type in SCHEMA[table_name]:
            if column_name == "id":
                columns_sql.append(f'"{column_name}" {data_type} PRIMARY KEY')
            elif column_name == "slug":
                columns_sql.append(f'"{column_name}" {data_type} NOT NULL')
            else:
                columns_sql.append(f'"{column_name}" {data_type}')

        conn.execute(
            f'''
            CREATE TABLE "{table_name}" (
                {", ".join(columns_sql)},
                UNIQUE("slug")
            )
            '''
        )


def insert_row(conn: sqlite3.Connection, table_name: str, row: Dict[str, Any]) -> None:
    columns = [column_name for column_name, _ in SCHEMA[table_name]]
    missing = [column for column in columns if column not in row]
    extra = [column for column in row if column not in columns]

    if missing:
        raise ValueError(f"{table_name} row is missing columns: {missing}")
    if extra:
        raise ValueError(f"{table_name} row has extra columns: {extra}")

    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    values = [row[column] for column in columns]

    conn.execute(
        f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})',
        values,
    )


def build_class_rows() -> List[Dict[str, Any]]:
    rows = []
    for i, item in enumerate(CLASS_DESCRIPTIONS.values(), start=1):
        (
            name,
            professor,
            location,
            from_date,
            to_date,
            short_description,
            full_description,
            additional_info,
            topics,
            frameworks,
            ranking,
            slug,
            title,
            subtitle,
        ) = item
        rows.append(
            {
                "id": i,
                "name": name,
                "professor": professor,
                "location": location,
                "from_date": from_date,
                "to_date": to_date,
                "short_description": short_description,
                "full_description": full_description,
                "additional_info": json_text(additional_info),
                "topics": topics,
                "frameworks": frameworks,
                "ranking": ranking,
                "show_on_preview": 1 if ranking <= 8 else 0,
                "show_on_full": 1,
                "displayed_color": "#2563eb",
                "slug": slug,
                "html_content": make_html(title, full_description),
                "preview_thumb": f"/media/classes/{slug}/thumb.png",
                "timeline_content_back": short_description,
                "title": title,
                "subtitle": subtitle,
                "link": f"/classes/{slug}",
            }
        )
    return rows


def build_project_rows() -> List[Dict[str, Any]]:
    rows = []
    for project in PROJECTS:
        title = project["title"]
        slug = project["slug"]
        full_description = project["full_description"]
        rows.append(
            {
                "id": project["id"],
                "title": title,
                "lab": project["lab"],
                "location": project["location"],
                "from_date": project["from_date"],
                "to_date": project["to_date"],
                "short_description": project["short_description"],
                "full_description": full_description,
                "additional_info": json_text(project["additional_info"]),
                "skills": project["skills"],
                "software_skills": project["software_skills"],
                "data_skills": project["data_skills"],
                "frameworks": project["frameworks"],
                "ranking": project["ranking"],
                "show_on_preview": 1 if project["ranking"] <= 8 else 0,
                "show_on_full": 1,
                "displayed_color": "#7c3aed",
                "html_content": make_html(title, full_description),
                "media_dir_path": f"/media/projects/{slug}",
                "link": f"/projects/{slug}",
                "slug": slug,
                "github_repos": project["github_repos"],
                "slug_old": project["slug_old"],
            }
        )
    return rows


def build_work_rows() -> List[Dict[str, Any]]:
    rows = []
    for work in WORKS:
        title = work["title"]
        slug = work["slug"]
        full_description = work["full_description"]
        rows.append(
            {
                "id": work["id"],
                "title": title,
                "employer": work["employer"],
                "location": work["location"],
                "from_date": work["from_date"],
                "to_date": work["to_date"],
                "short_description": work["short_description"],
                "full_description": full_description,
                "additional_info": json_text(work["additional_info"]),
                "skills": work["skills"],
                "frameworks": work["frameworks"],
                "ranking": work["ranking"],
                "show_on_preview": 1 if work["ranking"] <= 6 else 0,
                "show_on_full": 1,
                "displayed_color": "#059669",
                "slug": slug,
                "html_content": make_html(f'{title} at {work["employer"]}', full_description),
                "preview_thumb": f"/media/works/{slug}/thumb.png",
                "full_story": work["full_story"],
                "short_story": work["short_story"],
                "timeline_content_back": f'{work["from_date"]} to {work["to_date"]}: {work["short_description"]}',
                "link": f"/works/{slug}",
            }
        )
    return rows


def build_education_rows() -> List[Dict[str, Any]]:
    rows = []
    for education in EDUCATIONS:
        title = education["title"]
        subtitle = education["subtitle"]
        slug = education["slug"]
        full_text = (
            f'{title} at {subtitle}. Focus areas: {education["topics"]}. '
            f'Top accomplishment: {education["additional_info"]["top_accomplishment"]}'
        )
        rows.append(
            {
                "id": education["id"],
                "location": education["location"],
                "from_date": education["from_date"],
                "to_date": education["to_date"],
                "additional_info": json_text(education["additional_info"]),
                "topics": education["topics"],
                "ranking": education["ranking"],
                "show_on_preview": 1,
                "show_on_full": 1,
                "displayed_color": "#dc2626",
                "slug": slug,
                "html_content": make_html(title, full_text),
                "timeline_content_back": full_text,
                "title": title,
                "subtitle": subtitle,
                "link": f"/educations/{slug}",
            }
        )
    return rows


def populate(conn: sqlite3.Connection) -> None:
    for row in build_class_rows():
        insert_row(conn, "classes", row)
    for row in build_project_rows():
        insert_row(conn, "projects", row)
    for row in build_work_rows():
        insert_row(conn, "works", row)
    for row in build_education_rows():
        insert_row(conn, "educations", row)


def get_columns(conn: sqlite3.Connection, table_name: str) -> List[Tuple[str, str]]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [(row[1], row[2]) for row in rows]


def validate_schema(conn: sqlite3.Connection) -> None:
    actual_tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    ]

    expected_tables = sorted(TABLE_ORDER)
    expected_tables = [t.lower() for t in expected_tables]
    actual_tables = [t.lower() for t in actual_tables]

    if actual_tables != expected_tables:
        raise AssertionError(f"Table mismatch. Expected {expected_tables}, got {actual_tables}")

    for table_name in TABLE_ORDER:
        actual = get_columns(conn, table_name)
        expected = SCHEMA[table_name]
        if actual != expected:
            raise AssertionError(
                f"Schema mismatch for {table_name}.\nExpected: {expected}\nActual:   {actual}"
            )


def validate_data(conn: sqlite3.Connection) -> None:
    checks = {
        "classes": "SELECT COUNT(*) FROM classes WHERE slug IS NOT NULL AND slug != ''",
        "projects": "SELECT COUNT(*) FROM projects WHERE slug IS NOT NULL AND slug != ''",
        "works": "SELECT COUNT(*) FROM works WHERE slug IS NOT NULL AND slug != ''",
        "educations": "SELECT COUNT(*) FROM educations WHERE slug IS NOT NULL AND slug != ''",
    }
    for table_name, sql in checks.items():
        count = conn.execute(sql).fetchone()[0]
        if count == 0:
            raise AssertionError(f"No slug rows found for {table_name}")

    required_queries = [
        ("top 3 projects", "SELECT COUNT(*) FROM projects WHERE ranking <= 3", 3),
        ("C++ projects", "SELECT COUNT(*) FROM projects WHERE software_skills LIKE '%C++%' OR skills LIKE '%c++%'", 2),
        ("MLOps projects", "SELECT COUNT(*) FROM projects WHERE additional_info LIKE '%end_to_end_mlops%true%'", 3),
        ("Dune Buggy", "SELECT COUNT(*) FROM projects WHERE slug = 'dune-buggy'", 1),
        ("LLM-Net", "SELECT COUNT(*) FROM projects WHERE slug = 'ai-llm-net'", 1),
        ("AI-Grader", "SELECT COUNT(*) FROM projects WHERE slug = 'ai-llm-agents-og-grader'", 1),
        ("Nuvoton works", "SELECT COUNT(*) FROM works WHERE employer = 'Nuvoton'", 2),
        ("nutrition classes", "SELECT COUNT(*) FROM classes WHERE topics LIKE '%nutrition%' OR additional_info LIKE '%nutrition_related%true%'", 2),
        ("SBCC low-level coding", "SELECT COUNT(*) FROM classes WHERE slug = 'cis-054' AND full_description LIKE '%low-level coding%'", 1),
        ("cooking items", "SELECT COUNT(*) FROM classes WHERE topics LIKE '%cooking%' OR full_description LIKE '%cooking%'", 2),
        ("education entries", "SELECT COUNT(*) FROM educations", 3),
    ]

    for label, sql, minimum in required_queries:
        count = conn.execute(sql).fetchone()[0]
        if count < minimum:
            raise AssertionError(f"Coverage check failed for {label}: got {count}, expected at least {minimum}")


def write_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        create_tables(conn)
        populate(conn)
        conn.commit()


def main() -> None:
    db_path = Path(os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)).expanduser().resolve()
    write_database(db_path)
    print(f"Created SQLite database: {db_path}")

    with sqlite3.connect(db_path) as conn:
        for table_name in TABLE_ORDER:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            columns = ", ".join(column for column, _ in SCHEMA[table_name])
            print(f"{table_name}: {count} rows")
            print(f"  columns: {columns}")


if __name__ == "__main__":
    main()
