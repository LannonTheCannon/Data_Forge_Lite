def generate_theme_questions_and_labels(prompt: str, metadata_string: str, client=None):
    """
    Generates 4 chart-based questions from a high-level theme.
    These are used for ThemeNodes, so output can include 'theme' or 'question' nodes.
    """
    if not client:
        raise ValueError("Must provide OpenAI client.")

    system_prompt = """
    You are **Viz‑Detective‑GPT**, an expert data analyst guiding users through the early stages of Exploratory Data Analysis (EDA).

    Your primary goal is to help users **understand the dataset holistically**, before jumping into conclusions. Think like a thoughtful analyst, philosopher, and educator.

    🔍 Your mission:
    - Propose **exactly 4 exploratory paths** a user could follow from the given theme and dataset metadata.
    - Each path should focus on revealing a **pattern, structure, or insight** about how the data behaves.
    - These paths may be:
        • **High-level themes** (e.g., “Explore missingness patterns”, “Compare distributions across groups”, “Drill down into outliers”)
        • **Specific chartable questions** (e.g., “How does price vary by roast type?”, “Which regions have higher ratings?”)

    📊 Chart types allowed:
    Histogram, Bar chart, Grouped / Stacked bar, Line chart, Scatter plot, Scatter + best-fit line, Box plot, Heat-map, Table

    🎯 Your output format:
    - **One complete sentence per line**, each describing a visual exploration the user could take next.
    - Avoid list numbers or bullet points.
    - Use only the column names provided in the dataset metadata.
    - Avoid speculation, fabrication, or made-up columns.

    🧠 Frame of mind:
    You are helping the user gain **situational awareness** — a conceptual map of what’s inside the data. Each suggestion should deepen their understanding of the data’s structure, variability, relationships, or data quality.

    Example:
    Compare the average 'rating' across different 'roast' types using a grouped bar chart (grouped bar chart).
    """

    context = f"Theme: {prompt}\n\nDataset metadata:\n{metadata_string}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ]

    try:
        question_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=600
        )
        raw_questions = question_response.choices[0].message.content.strip().split("\n")
        questions = [q.strip() for q in raw_questions if q.strip()]
    except Exception as e:
        return [f"Error: {e}"], [f"Q{i+1}" for i in range(4)], ["question"] * 4

    # Label generation
    label_messages = [
        {"role": "system", "content": "Summarize each question as a short chart title (4–8 words)."},
        {"role": "user", "content": "\n".join(questions)}
    ]
    try:
        label_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=label_messages,
            max_tokens=300
        )
        labels = [l.strip() for l in label_response.choices[0].message.content.strip().split("\n") if l.strip()]
    except Exception:
        labels = [f"Q{i+1}" for i in range(len(questions))]

    # Node type classification: theme or question
    type_prompt = """
For each chart idea below, label it as 'theme' or 'question':

- 'theme' = a broad idea (e.g. "Explore correlations", "Drilldown into outliers")
- 'question' = a more specific analysis like "Compare price by region"

Output one word per line: theme or question.
"""
    try:
        type_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "system", "content": type_prompt}, {"role": "user", "content": "\n".join(questions)}],
            max_tokens=200
        )
        node_types = [t.strip().lower() for t in type_response.choices[0].message.content.strip().split("\n") if t.strip() in {"theme", "question"}]
    except:
        node_types = ["question"] * len(questions)

    # Fill missing
    while len(labels) < len(questions): labels.append(f"Q{len(labels)+1}")
    while len(node_types) < len(questions): node_types.append("question")

    return questions, labels, node_types


def generate_followup_questions_and_labels(prompt: str, metadata_string: str, client=None):
    """
    Generates 4 follow-up questions from an existing analysis question.
    Used for QuestionNodes. Results are either question or terminal.
    """
    if not client:
        raise ValueError("Must provide OpenAI client.")

    system_prompt = """
You are **Viz‑Detective‑GPT**, a senior data analyst AI.

You are following up on an existing data question and must generate exactly 4 next-step analysis questions.

Each should describe a chart insight that digs deeper into the original question.

Use chart types: Histogram, Bar chart, Grouped / Stacked bar, Line chart, Scatter plot, Scatter + best-fit line, Box plot, Heat-map, Table

No list numbers. Just one sentence per line.
"""

    context = f"Follow-up on:\n{prompt}\n\nDataset metadata:\n{metadata_string}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ]

    try:
        question_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=600
        )
        raw_questions = question_response.choices[0].message.content.strip().split("\n")
        questions = [q.strip() for q in raw_questions if q.strip()]
    except Exception as e:
        return [f"Error: {e}"], [f"Q{i+1}" for i in range(4)], ["terminal"] * 4

    # Label generation
    try:
        label_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "Rewrite each as a short chart title (4–8 words max)."},
                {"role": "user", "content": "\n".join(questions)}
            ],
            max_tokens=300
        )
        labels = [l.strip() for l in label_response.choices[0].message.content.strip().split("\n") if l.strip()]
    except:
        labels = [f"Q{i+1}" for i in range(len(questions))]

    # Node type classification: question or terminal
    type_prompt = """
Classify each follow-up as either a broad 'question' (for more exploration) or a 'terminal' (clear insight).

Use only: question or terminal. One word per line.
"""
    try:
        type_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "system", "content": type_prompt}, {"role": "user", "content": "\n".join(questions)}],
            max_tokens=200
        )
        node_types = [t.strip().lower() for t in type_response.choices[0].message.content.strip().split("\n") if t.strip() in {"question", "terminal"}]
    except:
        node_types = ["terminal"] * len(questions)

    while len(labels) < len(questions): labels.append(f"Q{len(labels)+1}")
    while len(node_types) < len(questions): node_types.append("terminal")

    return questions, labels, node_types
